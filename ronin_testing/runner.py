import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from device.device_session import DeviceSession
from device.platform import Platform
from perception.perception import ScreenPerception
from .generator import ScenarioGenerator
from .models import UserJourney, JourneyReport, ScenarioReport, GeneratedTestCase
from .usage_tracker import LLMUsageTracker

logger = logging.getLogger(__name__)

# Fixed execution order priority for test scenario classifications:
CLASSIFICATION_ORDER = {
    "happy_path": 0,
    "security_compliance": 1,
    "accessibility_audit": 2,
    "edge_case": 3,
    "boundary_test": 4,
    "performance_bottleneck": 5,
    "destructive_monkey": 6,
}


class RoninTestRunner:
    """Orchestrates E2E testing by decomposing journeys, executing scenarios, and emitting JUnit reports."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
        max_steps: int = 15,
        settle_time: float = 2.0,
        use_multimodal_vision: bool = False,
        headless: bool = True,
        fast_model: str = None,
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

    async def run_journey(self, journey: UserJourney) -> JourneyReport:
        """Generates scenarios from the journey, executes them, and builds reports & LLM usage metrics."""
        start_time = time.time()

        # 1. Instantiate the Agent
        from ronin_agent import RoninAgent

        agent = RoninAgent(
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

        # 2. Setup Telemetry immediately to track scenario decomposition
        usage_tracker = LLMUsageTracker(output_dir=self.assets_dir)
        current_component = "scenario_generator"
        agent.llm.usage_callback = lambda model, cost, usage: usage_tracker.add_call(
            current_component, model, cost, usage
        )
        if hasattr(agent, "fast_llm") and agent.fast_llm is not agent.llm:
            agent.fast_llm.usage_callback = lambda model, cost, usage: (
                usage_tracker.add_call(current_component, model, cost, usage)
            )

        scenarios = []
        scenario_reports: List[ScenarioReport] = []
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        session = None

        try:
            # 3. Decompose Journey into Test Scenarios
            scenarios = ScenarioGenerator(agent.llm).generate_scenarios(journey)

            # Sort scenarios in the fixed execution order
            scenarios.sort(key=lambda s: CLASSIFICATION_ORDER.get(s.classification, 99))

            # Switch telemetry context to planner for scenarios
            current_component = "ronin_agent_planner"

            # 4. Device Session orchestration
            session = DeviceSession(
                device_name=self.device_name,
                app_path=self.app_path,
                platform=self.platform,
                record_session=True,
                session_dir=self.assets_dir,
            )

            logger.info(
                "Initializing emulator and starting remote webdriver session..."
            )
            device_control = session.start(is_headless=self.headless)
            perception = ScreenPerception(session)

            # Execute each generated test case
            for scenario in scenarios:
                logger.info("========================================")
                logger.info(
                    "Executing Scenario: %s (%s)",
                    scenario.scenario_name,
                    scenario.test_id,
                )
                logger.info("========================================")

                scen_report = agent.execute_test_scenario(
                    test_case=scenario,
                    security_rules=journey.security_rules,
                    device_control=device_control,
                    perception=perception,
                    session_dir=self.assets_dir,
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
            # Fill missing scenarios as failed reports
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
            # Shutdown session safely
            if session is not None:
                try:
                    session.close()
                except Exception as cleanup_err:
                    logger.warning("Session cleanup failed: %s", cleanup_err)

            # Intercept and write LLM Usage Tracker JSON
            try:
                usage_tracker.save_report()
            except Exception as usage_err:
                logger.error("Failed to serialize LLM usage: %s", usage_err)

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

        # Save HTML/Markdown & JUnit Reports
        self._save_markdown_report(report)
        self._save_junit_report(report)

        return report

    def _save_markdown_report(self, report: JourneyReport):
        """Saves a structured human-readable markdown report of the journey run."""
        filepath = (
            self.assets_dir
            / f"report_{report.journey_id}_{time.strftime('%Y%m%d_%H%M%S')}.md"
        )
        lines = [
            f"# Ronin QA Test Report: {report.journey_name}",
            f"",
            f"* **Journey ID**: `{report.journey_id}`",
            f"* **Status**: {'🟢 PASSED' if report.status == 'PASSED' else '🔴 FAILED'}",
            f"* **Scenarios Run**: {len(report.scenarios)} ({report.passed_count} passed, {report.failed_count} failed, {report.skipped_count} skipped)",
            f"* **Duration**: {report.total_duration}s",
            f"",
            f"## Journey Description",
            f"{report.description}",
            f"",
            f"---",
            f"",
        ]

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

    def _save_junit_report(self, report: JourneyReport):
        """Generates standard JUnit XML report for CI systems integration."""
        filepath = (
            self.assets_dir
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
