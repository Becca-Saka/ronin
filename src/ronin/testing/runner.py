import logging
import time
from pathlib import Path
from typing import List

from ronin import __version__
from ronin.agent.memory.trail import SessionTrail
from ronin.agent.device.session import DeviceSession
from ronin.agent.device.platform import Platform
from ronin.agent.device.readiness import ensure_device_ready
from ronin.agent.perception.detector import YOLO_REPO_ID, YOLO_REVISION
from ronin.agent.perception.blind import for_grounding
from ronin.agent.plan.planner import GROUNDING_COORDINATES
from ronin.agent.memory.replay import ReplayLedger
from ronin.errors import ArtifactWriteFailed, cause_of
from .judge.coverage import unevidenced_security_rules
from .generator import ScenarioGenerator
from .report.progress import ScenarioProgress
from .models import (
    GeneratedTestCase,
    JourneyReport,
    RunEnvironment,
    ScenarioReport,
    UserJourney,
)
from .qa_agent import QAAgent
from .report.usage_tracker import LLMUsageTracker
from ronin.utils.secrets import SecretScrubber
from .report.reports import save_junit_report, save_markdown_report

logger = logging.getLogger(__name__)


def is_session_responsive(device_control) -> bool:
    """Lightweight check to verify if the webdriver connection and instrumentation server are alive."""
    if not device_control or not device_control.driver:
        return False
    try:
        if device_control.platform == Platform.ANDROID:
            _ = device_control.driver.current_package
        else:
            _ = device_control.driver.orientation
        return True
    except Exception as e:
        logger.warning("Session responsiveness check failed: %s", e)
        return False


def tally(reports: List[ScenarioReport]) -> tuple[int, int, int]:
    """Passed, failed and skipped, where anything else that ran counts as failed."""
    passed = sum(1 for report in reports if report.status == "PASSED")
    skipped = sum(1 for report in reports if report.status == "SKIPPED")
    return passed, len(reports) - passed - skipped, skipped


def aggregate_journey_status(passed_count: int, failed_count: int) -> str:
    """A suite passes only when at least one scenario has verified evidence."""
    return "PASSED" if passed_count > 0 and failed_count == 0 else "FAILED"


class RoninTestRunner:
    """Orchestrates E2E testing by decomposing journeys, executing scenarios, and emitting JUnit reports."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
        max_steps: int = 40,
        replay: bool = False,
        reuse_app: bool = False,
        keep_app: bool = False,
        settle_time: float = 2.0,
        headless: bool = True,
        fast_model: str = None,
        record_session: bool = True,
        grounding: str = GROUNDING_COORDINATES,
    ):
        self.grounding = grounding
        self.reuse_app = reuse_app
        self.keep_app = keep_app
        self.platform = platform
        self.device_name = device_name
        self.app_path = app_path
        self.model = model
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.replay = replay
        self.settle_time = settle_time
        self.headless = headless
        self.fast_model = fast_model
        self.record_session = record_session

    def _environment(self) -> RunEnvironment:
        """What was in play for this run, so a reader can explain a difference."""
        return RunEnvironment(
            ronin_version=__version__,
            model=self.model,
            fast_model=self.fast_model,
            platform=self.platform.value,
            grounding=self.grounding,
            detector=f"{YOLO_REPO_ID}@{YOLO_REVISION}",
        )

    def run_journey(self, journey: UserJourney) -> JourneyReport:
        """Generates scenarios from the journey, executes them, and builds reports & LLM usage metrics."""
        start_time = time.time()

        session = DeviceSession(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            record_session=self.record_session,
            session_dir=self.assets_dir,
            reuse_app=self.reuse_app,
            keep_app=self.keep_app,
        )

        agent = QAAgent(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            model=self.model,
            assets_dir=self.assets_dir,
            max_steps=self.max_steps,
            settle_time=self.settle_time,
            headless=self.headless,
            fast_model=self.fast_model,
            replay_ledger=ReplayLedger() if self.replay else None,
            grounding=self.grounding,
        )

        artifact_scrubber = SecretScrubber()
        usage_tracker = LLMUsageTracker(
            output_dir=session.telemetry_dir, scrubber=artifact_scrubber
        )
        current_component = "scenario_generator"
        device_agent = agent._device_agent
        device_agent.llm.usage_callback = lambda call: usage_tracker.add_call(
            current_component, call
        )
        if device_agent.fast_llm is not device_agent.llm:
            device_agent.fast_llm.usage_callback = lambda call: usage_tracker.add_call(
                current_component, call
            )

        scenarios = []
        scenario_reports: List[ScenarioReport] = []
        progress = None
        artifact_scrubber.learn_secrets(journey.secrets)
        device_agent.scrubber.learn_secrets(journey.secrets)

        try:
            # Both before the model call. Asking adb is cheap and names what is
            # available when the device is wrong. Opening the session proves
            # the device boots, the app installs and the driver answers, and a
            # generated suite is paid for before any of the three is known.
            ensure_device_ready(self.platform, self.device_name)

            logger.info(
                "Initializing emulator and starting remote webdriver session..."
            )
            device_control = session.start(is_headless=self.headless)
            perception = for_grounding(self.grounding, session)

            # Already in priority order: the generator walks the cells that
            # way, so a run that dies part way has done the P0 cells.
            suite = ScenarioGenerator(device_agent.llm).generate_suite(journey)
            scenarios = suite.scenarios
            # A cell nothing covered is recorded before the first scenario
            # runs, so a reader sees the gap even if the run dies next.
            for gap in suite.uncovered:
                scenario_reports.append(
                    ScenarioReport(
                        test_id=gap.cell.test_id,
                        scenario_name=f"Uncovered cell {gap.cell.key}",
                        cell=gap.cell.key,
                        status=gap.status,
                        cause=gap.cause.value,
                        failure_reason=gap.reason,
                    )
                )

            logger.info(
                "Successfully decomposed journey '%s' into %d scenarios:",
                journey.id,
                len(scenarios),
            )
            for idx, s in enumerate(scenarios):
                logger.info(
                    "  %d. [%s %s] %s (%s)",
                    idx + 1,
                    s.priority.value,
                    s.cell,
                    s.scenario_name,
                    s.test_id,
                )

            progress = ScenarioProgress(
                session.telemetry_dir, journey, scenarios, scrubber=artifact_scrubber
            )
            progress.write()
            logger.info("Scenario plan: %s", progress.path)

            current_component = "ronin_agent_planner"
            total_scenarios = len(scenarios)
            for idx, scenario in enumerate(scenarios):
                # Scenario evidence must not inherit screen names, actions, or
                # navigation claims from a previous scenario.
                trail = SessionTrail()
                progress.start(scenario.test_id)
                logger.info("========================================")

                current_component = f"scenario:{scenario.test_id}"
                logger.info(
                    "Executing Scenario %d/%d: %s (%s) - Cell: %s",
                    idx + 1,
                    total_scenarios,
                    scenario.scenario_name,
                    scenario.test_id,
                    scenario.cell,
                )
                logger.info("========================================")

                if not is_session_responsive(device_control):
                    logger.warning(
                        "WebDriver/instrumentation session is unresponsive. Attempting session recovery..."
                    )
                    try:
                        # Recovery, not teardown: the scenarios still to run
                        # need the app that this session was testing.
                        session.close(final=False)
                    except Exception as close_err:
                        logger.warning(
                            "Error closing old session during recovery: %s", close_err
                        )

                    session = DeviceSession(
                        device_name=self.device_name,
                        app_path=self.app_path,
                        platform=self.platform,
                        record_session=self.record_session,
                        session_dir=self.assets_dir,
                        session_id=session.session_id,
                        reuse_app=self.reuse_app,
                        keep_app=self.keep_app,
                        # The app is on the device by now, so a fresh look
                        # would call it preexisting and leave it behind.
                        installed_by_run=session.installed_by_run,
                    )
                    logger.info(
                        "Starting a new driver connection for session %s...",
                        session.session_id,
                    )
                    device_control = session.start(is_headless=self.headless)
                    perception = for_grounding(self.grounding, session)

                    usage_tracker.output_dir = session.telemetry_dir

                scen_report = agent.execute_scenario(
                    test_case=scenario,
                    security_rules=journey.security_rules,
                    device_control=device_control,
                    perception=perception,
                    session_dir=session.screenshots_dir,
                    session_trail=trail,
                )

                progress.record(scenario.test_id, scen_report)

                scenario_reports.append(scen_report)
                self._save_checkpoint(
                    journey,
                    scenario_reports,
                    *tally(scenario_reports),
                    start_time,
                    session,
                    artifact_scrubber,
                )
                usage_tracker.checkpoint()
                try:
                    session.checkpoint_recording()
                except Exception as record_err:
                    logger.warning(
                        "Failed to save the scenario recording: %s", record_err
                    )

        except BaseException as e:
            safe_error = artifact_scrubber.scrub(str(e)) or type(e).__name__
            logger.error(
                "Fatal run interruption (%s): %s", type(e).__name__, safe_error
            )
            if progress is not None:
                progress.abort(f"Aborted due to run failure/interrupt: {safe_error}")
            self._record_aborted(scenarios, scenario_reports, e)
            if isinstance(e, KeyboardInterrupt):
                raise e
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as cleanup_err:
                    logger.warning("Session cleanup failed: %s", cleanup_err)

            try:
                usage_tracker.save_report()
            except Exception as usage_err:
                logger.error("Failed to serialize LLM usage: %s", usage_err)
                scenario_reports.append(
                    ScenarioReport(
                        test_id="usage_report",
                        scenario_name="LLM usage report",
                        cell="infrastructure",
                        status="ERROR",
                        failure_reason=f"Failed to serialize LLM usage: {usage_err}",
                    )
                )

            # persist reports even on Ctrl+C or mid-run crash
            duration = time.time() - start_time
            unevidenced = unevidenced_security_rules(
                journey.security_rules, scenario_reports
            )
            for rule in unevidenced:
                logger.warning("No scenario evidenced security rule: %s", rule)
            passed_count, failed_count, skipped_count = tally(scenario_reports)
            journey_status = aggregate_journey_status(passed_count, failed_count)
            report = JourneyReport(
                journey_id=journey.id,
                journey_name=journey.name,
                description=artifact_scrubber.scrub(journey.description),
                status=journey_status,
                total_duration=round(duration, 2),
                scenarios=scenario_reports,
                passed_count=passed_count,
                failed_count=failed_count,
                skipped_count=skipped_count,
                unevidenced_security_rules=unevidenced,
                environment=self._environment(),
            )
            try:
                self._save_markdown_report(
                    report, session=session, scrubber=artifact_scrubber
                )
            except ArtifactWriteFailed as report_err:
                logger.error("Failed to save markdown report: %s", report_err)
                self._mark_report_write_failure(report, "markdown_report", report_err)
            try:
                self._save_junit_report(
                    report, session=session, scrubber=artifact_scrubber
                )
            except ArtifactWriteFailed as junit_err:
                logger.error("Failed to save JUnit report: %s", junit_err)
                self._mark_report_write_failure(report, "junit_report", junit_err)

        return report

    @staticmethod
    def _mark_report_write_failure(
        report: JourneyReport, artifact_name: str, error: Exception
    ) -> None:
        report.status = "FAILED"
        report.failed_count += 1
        report.scenarios.append(
            ScenarioReport(
                test_id=artifact_name,
                scenario_name=artifact_name.replace("_", " ").title(),
                cell="infrastructure",
                status="ERROR",
                failure_reason=f"Failed to write {artifact_name}: {error}",
            )
        )

    @staticmethod
    def _record_aborted(
        scenarios: List[GeneratedTestCase],
        scenario_reports: List[ScenarioReport],
        error: BaseException,
    ) -> int:
        """Records a failure for everything that never produced a result.

        A run can die before any scenario exists, if generation returns
        nothing. It would otherwise report zero tests and zero failures, which
        CI reads as a pass.
        """
        error_text = str(error) or type(error).__name__
        reason = f"Aborted due to run failure/interrupt: {error_text}"
        if not scenarios:
            scenario_reports.append(
                ScenarioReport(
                    test_id="scenario_generation",
                    scenario_name="Scenario generation",
                    cell="infrastructure",
                    status="ERROR",
                    failure_reason=reason,
                )
            )
            return 1

        recorded = 0
        for scenario in scenarios:
            if any(r.test_id == scenario.test_id for r in scenario_reports):
                continue
            scenario_reports.append(
                ScenarioReport(
                    test_id=scenario.test_id,
                    scenario_name=scenario.scenario_name,
                    cell=scenario.cell,
                    status="ABORTED",
                    cause=cause_of(error).value,
                    failure_reason=reason,
                )
            )
            recorded += 1
        return recorded

    def _save_checkpoint(
        self,
        journey: UserJourney,
        reports: List[ScenarioReport],
        passed: int,
        failed: int,
        skipped: int,
        started_at: float,
        session: DeviceSession,
        scrubber: SecretScrubber,
    ) -> None:
        checkpoint = JourneyReport(
            journey_id=journey.id,
            journey_name=journey.name,
            description=scrubber.scrub(journey.description),
            status="RUNNING",
            total_duration=round(time.time() - started_at, 2),
            scenarios=list(reports),
            passed_count=passed,
            failed_count=failed,
            skipped_count=skipped,
            environment=self._environment(),
        )
        try:
            save_markdown_report(
                checkpoint, session.telemetry_dir, session, scrubber, label="partial"
            )
            save_junit_report(
                checkpoint, session.telemetry_dir, scrubber, label="partial"
            )
        except ArtifactWriteFailed as exc:
            logger.error("Failed to checkpoint partial reports: %s", exc)

    def _save_markdown_report(
        self,
        report: JourneyReport,
        session: DeviceSession = None,
        scrubber: SecretScrubber | None = None,
    ):
        output_dir = session.telemetry_dir if session is not None else self.assets_dir
        return save_markdown_report(report, output_dir, session, scrubber)

    def _save_junit_report(
        self,
        report: JourneyReport,
        session: DeviceSession = None,
        scrubber: SecretScrubber | None = None,
    ):
        output_dir = session.telemetry_dir if session is not None else self.assets_dir
        return save_junit_report(report, output_dir, scrubber)
