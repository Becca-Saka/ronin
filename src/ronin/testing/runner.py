import logging
import time
from pathlib import Path
from typing import List

from ronin import __version__
from ronin.agent.trail import SessionTrail
from ronin.agent.device.device_session import DeviceSession
from ronin.agent.device.platform import Platform
from ronin.agent.perception.image_descriptor import YOLO_REPO_ID, YOLO_REVISION
from ronin.agent.perception.perception import ScreenPerception
from ronin.agent.device_planner import GROUNDING_COORDINATES
from ronin.agent.perception.marks import DETAIL_NUMBERS
from ronin.agent.replay import ReplayLedger
from ronin.errors import ArtifactWriteFailed
from .coverage import unevidenced_security_rules
from .generator import ScenarioGenerator
from .progress import ScenarioProgress
from .models import (
    GeneratedTestCase,
    JourneyReport,
    RunEnvironment,
    ScenarioReport,
    UserJourney,
)
from .qa_agent import QAAgent
from .usage_tracker import LLMUsageTracker
from ronin.utils.secrets import SecretScrubber
from .reports import save_junit_report, save_markdown_report

logger = logging.getLogger(__name__)


CLASSIFICATION_ORDER = {
    "happy_path": 0,
    "security_compliance": 1,
    "accessibility_audit": 2,
    "edge_case": 3,
    "boundary_test": 4,
    "performance_bottleneck": 5,
    "destructive_monkey": 6,
}


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
        settle_time: float = 2.0,
        use_multimodal_vision: bool = False,
        headless: bool = True,
        fast_model: str = None,
        record_session: bool = True,
        grounding: str = GROUNDING_COORDINATES,
        mark_detail: str = DETAIL_NUMBERS,
    ):
        self.grounding = grounding
        self.mark_detail = mark_detail
        self.platform = platform
        self.device_name = device_name
        self.app_path = app_path
        self.model = model
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.replay = replay
        self.settle_time = settle_time
        self.use_multimodal_vision = use_multimodal_vision
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
        )

        agent = QAAgent(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            model=self.model,
            assets_dir=self.assets_dir,
            max_steps=self.max_steps,
            settle_time=self.settle_time,
            use_multimodal_vision=self.use_multimodal_vision,
            headless=self.headless,
            fast_model=self.fast_model,
            replay_ledger=ReplayLedger() if self.replay else None,
            grounding=self.grounding,
            mark_detail=self.mark_detail,
        )

        usage_tracker = LLMUsageTracker(output_dir=session.telemetry_dir)
        current_component = "scenario_generator"
        device_agent = agent._device_agent
        device_agent.llm.usage_callback = lambda model, cost, usage: usage_tracker.add_call(
            current_component, model, cost, usage
        )
        if device_agent.fast_llm is not device_agent.llm:
            device_agent.fast_llm.usage_callback = lambda model, cost, usage: (
                usage_tracker.add_call(current_component, model, cost, usage)
            )

        scenarios = []
        scenario_reports: List[ScenarioReport] = []
        progress = None
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        artifact_scrubber = SecretScrubber()
        artifact_scrubber.learn_secrets(journey.secrets)
        device_agent.scrubber.learn_secrets(journey.secrets)

        try:
            scenarios = ScenarioGenerator(device_agent.llm).generate_scenarios(journey)
            scenarios.sort(key=lambda s: CLASSIFICATION_ORDER.get(s.classification, 99))

            logger.info(
                "Successfully decomposed journey '%s' into %d scenarios:",
                journey.id,
                len(scenarios),
            )
            for idx, s in enumerate(scenarios):
                logger.info(
                    "  %d. [%s] %s (%s)",
                    idx + 1,
                    s.classification.upper(),
                    s.scenario_name,
                    s.test_id,
                )

            progress = ScenarioProgress(
                session.telemetry_dir, journey, scenarios, scrubber=artifact_scrubber
            )
            progress.write()
            logger.info("Scenario plan: %s", progress.path)

            current_component = "ronin_agent_planner"

            logger.info(
                "Initializing emulator and starting remote webdriver session..."
            )
            device_control = session.start(is_headless=self.headless)
            perception = ScreenPerception(session)
            total_scenarios = len(scenarios)
            for idx, scenario in enumerate(scenarios):
                # Scenario evidence must not inherit screen names, actions, or
                # navigation claims from a previous scenario.
                trail = SessionTrail()
                progress.start(scenario.test_id)
                logger.info("========================================")

                current_component = f"scenario:{scenario.test_id}"
                logger.info(
                    "Executing Scenario %d/%d: %s (%s) - Category: %s",
                    idx + 1,
                    total_scenarios,
                    scenario.scenario_name,
                    scenario.test_id,
                    scenario.classification.upper(),
                )
                logger.info("========================================")

                if not is_session_responsive(device_control):
                    logger.warning(
                        "WebDriver/instrumentation session is unresponsive. Attempting session recovery..."
                    )
                    try:
                        session.close()
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
                    )
                    logger.info(
                        "Starting a new driver connection for session %s...",
                        session.session_id,
                    )
                    device_control = session.start(is_headless=self.headless)
                    perception = ScreenPerception(session)

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
                if scen_report.status == "PASSED":
                    passed_count += 1
                elif scen_report.status == "SKIPPED":
                    skipped_count += 1
                else:
                    failed_count += 1
                self._save_checkpoint(
                    journey,
                    scenario_reports,
                    passed_count,
                    failed_count,
                    skipped_count,
                    start_time,
                    session,
                    artifact_scrubber,
                )

        except BaseException as e:
            safe_error = artifact_scrubber.scrub(str(e)) or type(e).__name__
            logger.error(
                "Fatal run interruption (%s): %s", type(e).__name__, safe_error
            )
            if progress is not None:
                progress.abort(f"Aborted due to run failure/interrupt: {safe_error}")
            failed_count += self._record_aborted(scenarios, scenario_reports, e)
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
                        classification="infrastructure",
                        status="ERROR",
                        failure_reason=f"Failed to serialize LLM usage: {usage_err}",
                    )
                )
                failed_count += 1

            # persist reports even on Ctrl+C or mid-run crash
            duration = time.time() - start_time
            unevidenced = unevidenced_security_rules(
                journey.security_rules, scenario_reports
            )
            for rule in unevidenced:
                logger.warning("No scenario evidenced security rule: %s", rule)
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
                classification="infrastructure",
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

        A run that dies before any scenario exists — generation returning
        nothing, for instance — would otherwise report zero tests and zero
        failures, which CI reads as a pass.
        """
        error_text = str(error) or type(error).__name__
        reason = f"Aborted due to run failure/interrupt: {error_text}"
        if not scenarios:
            scenario_reports.append(
                ScenarioReport(
                    test_id="scenario_generation",
                    scenario_name="Scenario generation",
                    classification="happy_path",
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
                    classification=scenario.classification,
                    status="ABORTED",
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
