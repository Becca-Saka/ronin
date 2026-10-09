"""
QAAgent — testing layer built on top of DeviceAgent.
Injects QAPlanner per-scenario and collects QA outputs via the step hook.
Returns ScenarioReport with all verifications, detected issues, and status.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from ronin.agent.device_agent import DeviceAgent, DeviceRunResult
from ronin.agent.device_planner import DeviceStepResponse, DoneAction
from ronin.agent.trail import SessionTrail
from ronin.agent.device.platform import Platform
from ronin.llm.llm_client import LLMClient
from ronin.testing.models import (
    GeneratedTestCase,
    QAIssue,
    RuleVerification,
    ScenarioReport,
)
from ronin.testing.qa_planner import QAPlanner, QAStepResponse

logger = logging.getLogger(__name__)


class QAAgent:
    """
    Orchestrates a single QA test scenario using DeviceAgent + QAPlanner.

    One QAAgent instance is reused across all scenarios in a journey — LLM
    clients are created once. A fresh QAPlanner is created per scenario to
    carry scenario-specific context (instructions, expected outcomes, rules).
    """

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
    ):
        self._use_multimodal_vision = use_multimodal_vision

        self._device_agent = DeviceAgent(
            device_name=device_name,
            app_path=app_path,
            platform=platform,
            model=model,
            assets_dir=assets_dir,
            max_steps=max_steps,
            settle_time=settle_time,
            use_multimodal_vision=use_multimodal_vision,
            headless=headless,
            fast_model=fast_model,
        )
        self._fast_llm: LLMClient = self._device_agent.fast_llm

    def execute_scenario(
        self,
        test_case: GeneratedTestCase,
        security_rules: List[str],
        device_control,
        perception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
    ) -> ScenarioReport:
        """Execute one test scenario and return a complete ScenarioReport."""
        start_time = time.time()

        qa_outputs: List[QAStepResponse] = []
        stop_reason: dict = {"value": None}  # mutable so the nested hook can write out

        def step_hook(step_response: DeviceStepResponse) -> Optional[bool]:
            if not isinstance(step_response, QAStepResponse):
                return None

            qa_outputs.append(step_response)

            if any(i.issue_type == "crash_screen" for i in step_response.detected_issues):
                logger.error("App crash screen detected — stopping scenario.")
                stop_reason["value"] = "App crashed: crash screen detected."
                return False

            if not step_response.is_scenario_feasible:
                logger.warning(
                    "Scenario flagged infeasible: %s", step_response.infeasibility_reason
                )
                stop_reason["value"] = (
                    f"Scenario is infeasible: {step_response.infeasibility_reason or 'No reason provided'}"
                )
                return False

            return None

        qa_planner = QAPlanner(
            llm_client=self._fast_llm,
            use_multimodal_vision=self._use_multimodal_vision,
            test_case=test_case,
            security_rules=security_rules,
        )

        run_result: DeviceRunResult = self._device_agent.run(
            goal=test_case.steps_instructions,
            device_control=device_control,
            perception=perception,
            session_dir=session_dir,
            session_trail=session_trail,
            step_hook=step_hook,
            planner_override=qa_planner,
        )

        return self._build_report(
            test_case=test_case,
            run_result=run_result,
            qa_outputs=qa_outputs,
            stop_reason=stop_reason["value"],
            duration=time.time() - start_time,
        )

    def _build_report(
        self,
        test_case: GeneratedTestCase,
        run_result: DeviceRunResult,
        qa_outputs: List[QAStepResponse],
        stop_reason: Optional[str],
        duration: float,
    ) -> ScenarioReport:
        all_verifications: List[RuleVerification] = []
        all_issues: List[QAIssue] = []

        for step_resp in qa_outputs:
            for verify in step_resp.rule_verifications:
                if not verify.screen_name:
                    verify.screen_name = step_resp.current_screen
                all_verifications.append(verify)

            for issue in step_resp.detected_issues:
                issue.current_screen = step_resp.current_screen
                all_issues.append(issue)
                logger.info(
                    "QA issue [%s/%s] on '%s': %s",
                    issue.severity, issue.issue_type,
                    step_resp.current_screen, issue.description,
                )

        status, failure_reason = self._determine_status(
            run_result=run_result,
            stop_reason=stop_reason,
            all_verifications=all_verifications,
            all_issues=all_issues,
            qa_outputs=qa_outputs,
        )

        return ScenarioReport(
            test_id=test_case.test_id,
            scenario_name=test_case.scenario_name,
            classification=test_case.classification,
            status=status,
            duration=round(duration, 2),
            steps_executed=run_result.steps_executed,
            rule_verifications=all_verifications,
            detected_issues=all_issues,
            failure_reason=failure_reason,
        )

    def _determine_status(
        self,
        run_result: DeviceRunResult,
        stop_reason: Optional[str],
        all_verifications: List[RuleVerification],
        all_issues: List[QAIssue],
        qa_outputs: List[QAStepResponse],
    ):
        from ronin.utils.redactor import redact_text

        if run_result.status == "hook_stop" and stop_reason and "infeasible" in stop_reason.lower():
            return "SKIPPED", stop_reason

        if run_result.status == "exception":
            return "FAILED", redact_text(run_result.failure_reason or "Execution exception")

        if run_result.status in ("loop_detected", "step_limit"):
            return "FAILED", run_result.failure_reason

        if any(i.issue_type == "crash_screen" for i in all_issues):
            return "FAILED", "App crashed: crash screen detected."

        critical_issue = next(
            (i for i in all_issues if i.severity == "critical"), None
        )
        if critical_issue:
            return "FAILED", f"Critical QA issue: {critical_issue.description}"

        failed_rule = next(
            (v for v in all_verifications if v.status == "failed"), None
        )
        if failed_rule:
            return "FAILED", f"Rule validation failed: {failed_rule.rule}"

        if qa_outputs and not qa_outputs[-1].is_scenario_feasible:
            return "SKIPPED", qa_outputs[-1].infeasibility_reason or "Scenario infeasible"

        return "PASSED", None
