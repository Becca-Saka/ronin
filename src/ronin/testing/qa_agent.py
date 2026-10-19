"""
QAAgent — testing layer built on top of DeviceAgent.
Injects QAPlanner per-scenario and collects QA outputs via the step hook.
Returns ScenarioReport with all verifications, detected issues, and status.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

from ronin.agent.run.agent import DeviceAgent, DeviceRunResult
from ronin.agent.plan.planner import DeviceStepResponse
from ronin.agent.memory.trail import SessionTrail
from ronin.agent.device.platform import Platform
from ronin.agent.memory.replay import ReplayLedger

from ronin.defaults import STEP_BUDGET_FLOOR, STEP_BUDGET_PER_INSTRUCTION
from ronin.llm.llm_client import LLMClient
from ronin.testing.models import (
    ExpectedOutcome,
    GeneratedTestCase,
    QAIssue,
    RuleVerification,
    ScenarioReport,
    SensitiveFinding,
)
from ronin.agent.device import restoration
from ronin.testing.capabilities import (
    target_of,
    unconfirmed_capabilities,
    unsupported_reason,
)
from ronin.agent.plan.planner import GROUNDING_COORDINATES
from ronin.testing import rules
from ronin.testing.verdicts import (
    _missing_required_ids,
    _terminal_outcome,
)
from ronin.testing.qa_auditor import Assessment, QAAuditor, ScreenWatch
from ronin.testing.qa_planner import QAPlanner, QAStepResponse
from ronin.errors import Cause
from ronin.testing import taxonomy, transitions

from ronin.agent.actuation.actions import WaitAction

logger = logging.getLogger(__name__)

LAUNCH_RECHECK_WAIT = 1.5


def _defer_first_look(step_response, assessment, first_look: bool) -> bool:
    """Withdraws a terminal verdict taken before the app may have finished drawing.

    A launching app shows an empty screen for a moment, which reads as both
    crashed and infeasible. One more step costs little. A false crash report
    costs a red build.

    The response's own action has to be withdrawn too. It is almost always
    `done`, and the run loop acts on it the moment this returns. Clearing the
    verdict alone would change the label and nothing else.

    Returns whether a verdict was withdrawn.
    """
    if not first_look:
        return False

    deferred = False
    if any(i.issue_type == "crash_screen" for i in assessment.detected_issues):
        logger.warning("Crash screen on the first look — re-checking next step.")
        assessment.detected_issues = [
            i for i in assessment.detected_issues if i.issue_type != "crash_screen"
        ]
        deferred = True
    if not step_response.is_scenario_feasible:
        logger.warning("Infeasible on the first look — re-checking next step.")
        step_response.is_scenario_feasible = True
        deferred = True

    if deferred:
        # Whatever the action was, it was chosen from the same screen the
        # verdict came from. Acting on it would trust a reading we just said
        # not to trust — and `done` would end the run before the second look.
        step_response.action = WaitAction(
            duration=LAUNCH_RECHECK_WAIT,
            reason="Looking again before judging a screen the app may still be drawing.",
        )
    return deferred


class QAAgent:
    """Runs one QA scenario, and keeps the clients a journey shares.

    One QAAgent serves every scenario in a journey, so the model clients are
    built once. Each scenario gets its own QAPlanner and its own QAAuditor,
    because each carries state that must not reach the next scenario.
    """

    # Class-level so an agent built without __init__ still answers sensibly.
    _grounding = GROUNDING_COORDINATES

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
        max_steps: int = 40,
        settle_time: float = 2.0,
        headless: bool = True,
        fast_model: str = None,
        replay_ledger: Optional[ReplayLedger] = None,
        grounding: str = GROUNDING_COORDINATES,
    ):
        self._grounding = grounding

        self._device_agent = DeviceAgent(
            model=model,
            max_steps=max_steps,
            settle_time=settle_time,
            fast_model=fast_model,
            replay_ledger=replay_ledger,
            grounding=grounding,
        )
        self._fast_llm: LLMClient = self._device_agent.fast_llm

    def _step_budget(self, test_case: GeneratedTestCase) -> int:
        """How many agent steps a scenario may use.

        Scenarios differ widely in size, and a short check cannot share one
        number with a long end-to-end order. So each gets a budget from its own
        instructions, under the configured ceiling.
        """
        wanted = len(test_case.steps_instructions) * STEP_BUDGET_PER_INSTRUCTION
        ceiling = self._device_agent.max_steps
        return min(ceiling, max(STEP_BUDGET_FLOOR, wanted))

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
        assessments: List[Assessment] = []
        stop_reason: dict = {"value": None}  # mutable so the nested hook can write out

        # Only what one screen can settle is put to the auditor. Asked whether
        # the app hung, it answers from pixels that cannot carry the fact.
        auditor = QAAuditor(
            self._fast_llm,
            test_case.expected_outcomes,
            security_rules,
            scrubber=self._device_agent.scrubber,
        )

        watch = ScreenWatch(auditor)

        def step_hook(step_response: DeviceStepResponse, snapshot) -> Optional[bool]:
            if not isinstance(step_response, QAStepResponse):
                return None

            qa_outputs.append(step_response)
            # The screen is audited by something that was not told the plan, so
            # a verdict here is capable of contradicting the action above it.
            fresh = watch.look(snapshot)
            for assessment in fresh:
                assessment.screen_name = (
                    assessment.screen_name or step_response.current_screen
                )
            assessments.extend(fresh)

            if fresh and _defer_first_look(
                step_response, fresh[-1], len(qa_outputs) == 1
            ):
                # The withdrawal promised another look; this is what keeps it.
                watch.look_again()
                return None

            if any(
                issue.issue_type == "crash_screen"
                for assessment in fresh
                for issue in assessment.detected_issues
            ):
                logger.error("App crash screen detected — stopping scenario.")
                stop_reason["value"] = "App crashed: crash screen detected."
                return False

            if not step_response.is_scenario_feasible:
                logger.warning(
                    "Scenario flagged infeasible: %s",
                    self._device_agent.scrubber.scrub(step_response.infeasibility_reason),
                )
                stop_reason["value"] = (
                    "Scenario is infeasible: "
                    + self._device_agent.scrubber.scrub(
                        step_response.infeasibility_reason or "No reason provided"
                    )
                )
                return False

            return None

        # Read against the device in front of us, so a scenario whose
        # capability this target cannot apply is skipped rather than failing a
        # step inside itself.
        unsupported = unsupported_reason(test_case, target_of(device_control))
        if unsupported:
            return ScenarioReport(
                test_id=test_case.test_id,
                scenario_name=test_case.scenario_name,
                cell=test_case.cell,
                status="UNSUPPORTED",
                cause=Cause.SCENARIO.value,
                duration=round(time.time() - start_time, 2),
                failure_reason=unsupported,
            )

        # OCR caches are useful within a scenario but are evidence from the
        # wrong run once the app has been restarted.
        perception.reset()

        qa_planner = QAPlanner(
            llm_client=self._fast_llm,
            setup=test_case.setup,
            scrubber=self._device_agent.scrubber,
            grounding=self._grounding,
        )

        replay = self._device_agent.replay_ledger
        # Replay skips perception and auditing on the screens it walks, so a
        # cell whose claims apply mid-navigation must not use it.
        audits_every_screen = taxonomy.audits_every_screen(
            test_case.category, test_case.scenario_type
        )
        if audits_every_screen:
            # Replayed screens are never perceived or audited, which is the
            # whole point of these scenarios.
            self._device_agent.replay_ledger = None

        step_budget = self._step_budget(test_case)
        logger.info(
            "Scenario %s: %d instruction(s), %d step budget",
            test_case.test_id,
            len(test_case.steps_instructions),
            step_budget,
        )

        try:
            # Scenarios share one device, and SetupMode restores the app and
            # not the system around it. Without this, one scenario that goes
            # offline decides what every later scenario was testing.
            with restoration.preserved(device_control):
                run_result: DeviceRunResult = self._device_agent.run(
                    goal=test_case.steps_instructions,
                    device_control=device_control,
                    perception=perception,
                    session_dir=session_dir,
                    session_trail=session_trail,
                    step_hook=step_hook,
                    planner_override=qa_planner,
                    max_steps=step_budget,
                    setup=test_case.setup,
                )
        finally:
            self._device_agent.replay_ledger = replay

        # Nothing ever leaves the screen a scenario ended on, so it is read here.
        assessments.extend(watch.close())

        return self._build_report(
            test_case=test_case,
            run_result=run_result,
            qa_outputs=qa_outputs,
            assessments=assessments,
            stop_reason=stop_reason["value"],
            duration=time.time() - start_time,
            security_rules=security_rules,
        )

    def _build_report(
        self,
        test_case: GeneratedTestCase,
        run_result: DeviceRunResult,
        qa_outputs: List[QAStepResponse],
        assessments: List[Assessment],
        stop_reason: Optional[str],
        duration: float,
        security_rules: List[str],
    ) -> ScenarioReport:
        all_verifications: List[RuleVerification] = []
        all_issues: List[QAIssue] = []
        all_findings: List[SensitiveFinding] = []

        scrub = self._device_agent.scrubber.scrub

        for assessment in assessments:
            for verify in assessment.rule_verifications:
                if not verify.screen_name:
                    verify.screen_name = assessment.screen_name
                # An audit naming the value it found — "password123 is visible
                # unmasked" — would otherwise write it into logs and reports.
                verify.rule = scrub(verify.rule)
                verify.reasoning = scrub(verify.reasoning)
                all_verifications.append(verify)

            for finding in assessment.sensitive_findings:
                finding.screen_name = assessment.screen_name
                finding.quoted = scrub(finding.quoted)
                all_findings.append(finding)

            for issue in assessment.detected_issues:
                issue.current_screen = assessment.screen_name
                issue.description = scrub(issue.description)
                all_issues.append(issue)
                logger.info(
                    "QA issue [%s/%s] on '%s': %s",
                    issue.severity, issue.issue_type,
                    assessment.screen_name, issue.description,
                )

        all_verifications.extend(
            self._settled_transitions(test_case, assessments)
        )

        status, failure_reason, cause = self._determine_status(
            run_result=run_result,
            stop_reason=stop_reason,
            all_verifications=all_verifications,
            all_issues=all_issues,
            qa_outputs=qa_outputs,
            assessments=assessments,
            required_outcomes=test_case.expected_outcomes,
            required_capabilities=test_case.required_capabilities,
            security_rules=security_rules,
        )

        return ScenarioReport(
            test_id=test_case.test_id,
            scenario_name=test_case.scenario_name,
            cell=test_case.cell,
            status=status,
            cause=cause.value if cause else None,
            duration=round(duration, 2),
            steps_executed=run_result.steps_executed,
            rule_verifications=all_verifications,
            detected_issues=all_issues,
            failure_reason=scrub(failure_reason or "") or None,
        )

    def _settled_transitions(
        self, test_case: GeneratedTestCase, assessments: List[Assessment]
    ) -> List[RuleVerification]:
        """Evidence for the outcomes no single screenshot could carry.

        The gate requires every expected outcome, and the auditor is never
        shown the ones about reaching a screen. Without this they are demanded
        and unanswerable, and a scenario that did everything asked of it
        reports missing evidence.

        A failure here leaves them unevidenced, which is the honest answer:
        the judgement did not happen, so nothing was judged.
        """
        try:
            return transitions.settle(
                self._fast_llm,
                test_case.expected_outcomes,
                assessments,
                scrubber=self._device_agent.scrubber,
            )
        except Exception as exc:  # noqa: BLE001 - model boundary
            logger.warning("Could not judge the transitions of this scenario: %s", exc)
            return []

    def _determine_status(
        self,
        run_result: DeviceRunResult,
        stop_reason: Optional[str],
        all_verifications: List[RuleVerification],
        all_issues: List[QAIssue],
        qa_outputs: List[QAStepResponse],
        assessments: List[Assessment],
        required_outcomes: Sequence[ExpectedOutcome | str],
        security_rules: Sequence[str],
        all_findings: Optional[List[SensitiveFinding]] = None,
        required_capabilities: Sequence = (),
    ):
        # Feasibility stays the planner's call: it is a statement about whether
        # the agent can proceed, not about whether the app is correct.
        terminal = _terminal_outcome(
            run_result,
            stop_reason,
            all_verifications,
            all_issues,
            qa_outputs,
            all_findings,
        )
        if terminal:
            return terminal

        if run_result.status != "goal_reached" or not assessments:
            return (
                "UNVERIFIED",
                "The scenario did not produce verified completion evidence.",
                Cause.AGENT,
            )

        # Completion is the planner's call, because it is the only one told what
        # the scenario set out to do. The auditor is deliberately blind to the
        # plan, so it cannot judge whether a plan finished.
        #
        # Independence is not lost. The expected outcomes reach the auditor as
        # rules, and the checks below refuse the scenario unless the auditor
        # evidenced every one of them on a screen it judged without the plan.
        unprovable = rules.unprovable(required_outcomes)
        if unprovable:
            return (
                "UNVERIFIED",
                "Nothing recorded during a run can settle these outcomes: "
                + ", ".join(unprovable)
                + ". They ask how the app performed, and Ronin measures no part of that.",
                Cause.SCENARIO,
            )

        # A declared capability that never reached the device leaves every
        # outcome below it describing a state the run never entered.
        unconfirmed = (
            unconfirmed_capabilities(
                required_capabilities, run_result.capability_receipts
            )
            if required_capabilities
            else []
        )
        if unconfirmed:
            return (
                "UNVERIFIED",
                "The scenario declared device states it never had confirmed: "
                + ", ".join(unconfirmed),
                Cause.EVIDENCE,
            )

        missing = _missing_required_ids(
            all_verifications, required_outcomes, security_rules
        )
        if missing:
            return (
                "UNVERIFIED",
                "Required verification evidence is missing or untested: " + ", ".join(missing),
                Cause.EVIDENCE,
            )

        # Named last so a specific gap is reported in preference to this.
        if not required_outcomes:
            return (
                "UNVERIFIED",
                "The scenario declared no expected outcome, so reaching the goal "
                "rests on the planner's own word.",
                Cause.SCENARIO,
            )

        return "PASSED", None, None
