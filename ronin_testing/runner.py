import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from agent.trail import SessionTrail
from agent.device.device_session import DeviceSession
from agent.device.platform import Platform
from agent.perception.perception import ScreenPerception
from .generator import ScenarioGenerator
from .models import UserJourney, JourneyReport, ScenarioReport
from .qa_agent import QAAgent
from .usage_tracker import LLMUsageTracker

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
        settle_time: float = 2.0,
        use_multimodal_vision: bool = False,
        headless: bool = True,
        fast_model: str = None,
        record_session: bool = True,
    ):
        self.platform = platform
        self.device_name = device_name
        self.app_path = app_path
        self.model = model
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.settle_time = settle_time
        self.use_multimodal_vision = use_multimodal_vision
        self.headless = headless
        self.fast_model = fast_model
        self.record_session = record_session

    async def run_journey(self, journey: UserJourney) -> JourneyReport:
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
        passed_count = 0
        failed_count = 0
        skipped_count = 0

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

            current_component = "ronin_agent_planner"

            logger.info(
                "Initializing emulator and starting remote webdriver session..."
            )
            device_control = session.start(is_headless=self.headless)
            perception = ScreenPerception(session)
            trail = SessionTrail()

            total_scenarios = len(scenarios)
            for idx, scenario in enumerate(scenarios):
                logger.info("========================================")
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

                scenario_reports.append(scen_report)
                if scen_report.status == "PASSED":
                    passed_count += 1
                elif scen_report.status == "SKIPPED":
                    skipped_count += 1
                else:
                    failed_count += 1

        except BaseException as e:
            logger.exception(
                "Fatal exception or interrupt encountered during run: %s", e
            )
            for scenario in scenarios:
                if not any(r.test_id == scenario.test_id for r in scenario_reports):
                    scenario_reports.append(
                        ScenarioReport(
                            test_id=scenario.test_id,
                            scenario_name=scenario.scenario_name,
                            classification=scenario.classification,
                            status="FAILED",
                            failure_reason=f"Aborted due to run failure/interrupt: {str(e)}",
                        )
                    )
                    failed_count += 1
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

            # persist reports even on Ctrl+C or mid-run crash
            duration = time.time() - start_time
            journey_status = "PASSED" if failed_count == 0 else "FAILED"
            report = JourneyReport(
                journey_id=journey.id,
                journey_name=journey.name,
                description=journey.description,
                status=journey_status,
                total_duration=round(duration, 2),
                scenarios=scenario_reports,
                passed_count=passed_count,
                failed_count=failed_count,
                skipped_count=skipped_count,
            )
            try:
                self._save_markdown_report(report, session=session)
            except Exception as report_err:
                logger.error("Failed to save markdown report: %s", report_err)
            try:
                self._save_junit_report(report, session=session)
            except Exception as junit_err:
                logger.error("Failed to save JUnit report: %s", junit_err)

        return report

    def _save_markdown_report(
        self, report: JourneyReport, session: DeviceSession = None
    ):
        """Saves a structured human-readable markdown report of the journey run."""
        output_dir = session.telemetry_dir if session is not None else self.assets_dir
        filepath = (
            output_dir
            / f"report_{report.journey_id}_{time.strftime('%Y%m%d_%H%M%S')}.md"
        )
        lines = [
            f"# Ronin QA Test Report: {report.journey_name}",
            f"",
            f"* **Journey ID**: `{report.journey_id}`",
            f"* **Status**: {'🟢 PASSED' if report.status == 'PASSED' else '🔴 FAILED'}",
            f"* **Scenarios Run**: {len(report.scenarios)} ({report.passed_count} passed, {report.failed_count} failed, {report.skipped_count} skipped)",
            f"* **Duration**: {report.total_duration}s",
        ]
        if session:
            lines.extend(
                [
                    f"* **Execution Log**: [ronin_session_{session.session_id}.log](file://{session.logs_dir.absolute()}/ronin_session_{session.session_id}.log)",
                    f"* **Appium Server Log**: [appium_session_{session.session_id}.log](file://{session.logs_dir.absolute()}/appium_session_{session.session_id}.log)",
                ]
            )
        lines.extend(
            [
                f"",
                f"## Journey Description",
                f"{report.description}",
                f"",
                f"---",
                f"",
            ]
        )

        for idx, scen in enumerate(report.scenarios):
            if scen.status == "PASSED":
                status_icon = "🟢"
            elif scen.status == "SKIPPED":
                status_icon = "🟡"
            else:
                status_icon = "🔴"
            lines.extend(
                [
                    f"## Scenario {idx + 1}: {scen.scenario_name} ({scen.test_id})",
                    f"* **Classification**: `{scen.classification}`",
                    f"* **Status**: {status_icon} {scen.status}",
                    f"* **Steps Executed**: {scen.steps_executed}",
                    f"* **Duration**: {scen.duration}s",
                    f"",
                ]
            )

            if scen.failure_reason:
                lines.append(f"* ❌ **Failure Reason**: `{scen.failure_reason}`")

            actionable_verifications = [
                v for v in scen.rule_verifications if v.status != "untested"
            ]
            if actionable_verifications:
                lines.append("### Verifications:")
                for rule in actionable_verifications:
                    rule_status_icon = (
                        "[x] Passed" if rule.status == "passed" else "[ ] Failed"
                    )
                    screen_suffix = (
                        f" on `{rule.screen_name}`"
                        if getattr(rule, "screen_name", None)
                        else ""
                    )
                    lines.append(
                        f"  * {rule_status_icon}: {rule.rule}{screen_suffix} (Reasoning: {rule.reasoning})"
                    )
                lines.append("")

            if scen.detected_issues:
                lines.append("### Detected Issues:")
                for iss in scen.detected_issues:
                    lines.append(
                        f"  * ⚠️ **{iss.severity.upper()} ({iss.issue_type})**: {iss.description} on screen `{iss.current_screen}`"
                    )
                lines.append("")

            lines.append("---")

        try:
            filepath.write_text("\n".join(lines))
            logger.info("Markdown test report successfully written to %s", filepath)
        except Exception as e:
            logger.error("Failed to write Markdown test report: %s", e)

    def _save_junit_report(self, report: JourneyReport, session: DeviceSession = None):
        """Generates standard JUnit XML report for CI systems integration."""
        output_dir = session.telemetry_dir if session is not None else self.assets_dir
        filepath = (
            output_dir
            / f"junit_{report.journey_id}_{time.strftime('%Y%m%d_%H%M%S')}.xml"
        )

        testsuites_el = ET.Element(
            "testsuites",
            {
                "name": "Ronin User Journeys",
                "tests": str(len(report.scenarios)),
                "failures": str(report.failed_count),
                "skipped": str(report.skipped_count),
                "time": str(report.total_duration),
            },
        )

        testsuite_el = ET.SubElement(
            testsuites_el,
            "testsuite",
            {
                "name": report.journey_name,
                "tests": str(len(report.scenarios)),
                "failures": str(report.failed_count),
                "skipped": str(report.skipped_count),
                "time": str(report.total_duration),
                "id": report.journey_id,
            },
        )

        for scen in report.scenarios:
            testcase_el = ET.SubElement(
                testsuite_el,
                "testcase",
                {
                    "name": scen.scenario_name,
                    "classname": f"{report.journey_id}.{scen.test_id}",
                    "time": str(scen.duration),
                },
            )

            if scen.status == "FAILED":
                msg = scen.failure_reason or "Verification failed"
                detail_lines = []
                for iss in scen.detected_issues:
                    detail_lines.append(
                        f"[{iss.severity.upper()}] {iss.issue_type}: {iss.description} on {iss.current_screen}"
                    )

                failure_el = ET.SubElement(
                    testcase_el, "failure", {"message": msg, "type": "QAAssertionError"}
                )
                failure_el.text = "\n".join(detail_lines) or msg
            elif scen.status == "SKIPPED":
                msg = scen.failure_reason or "Scenario skipped"
                ET.SubElement(testcase_el, "skipped", {"message": msg})

        try:
            tree = ET.ElementTree(testsuites_el)
            with open(filepath, "wb") as f:
                tree.write(f, encoding="utf-8", xml_declaration=True)
            logger.info("JUnit XML test report successfully written to %s", filepath)
        except Exception as e:
            logger.error("Failed to write JUnit XML test report: %s", e)
