"""
QAPlanner — extends DevicePlanner with QA testing concerns.
Adds scenario instructions, expected outcomes, security rules, and the
full QA audit prompt to each planning call. Returns a QAStepResponse
that carries both the device action and QA audit results.

This is the only LLM call for QA scenarios (single combined call).
The architecture is already split: DevicePlanner has no knowledge of QA.
The two-call optimization (separate cheap-model audit) can be wired in
by splitting QAPlanner into DevicePlanner + QAAuditor when needed.
"""

import logging
from typing import List, Optional

from pydantic import Field, model_validator

from ronin.agent.perception.marks import DETAIL_NUMBERS
from ronin.agent.device_planner import (
    GROUNDING_COORDINATES,
    GROUNDING_MARKS,
    DevicePlanner,
    DeviceStepResponse,
    MarksDeviceAction,
)
from ronin.llm.llm_client import LLMClient
from ronin.llm.messages import MultiPartMessage
from ronin.testing.models import QAIssue, RuleVerification

logger = logging.getLogger(__name__)


class QAStepResponse(DeviceStepResponse):
    """DeviceStepResponse extended with QA audit outputs."""

    rule_verifications: List[RuleVerification] = Field(
        default_factory=list,
        description=(
            "Evaluations of expected outcomes, custom security rules, and dynamic screen assertions "
            "against the current screen."
        ),
    )
    detected_issues: List[QAIssue] = Field(
        default_factory=list,
        description=(
            "Visual bugs, layout overflows, crashed screens, or security data exposures "
            "found on the current screen."
        ),
    )
    is_scenario_feasible: bool = Field(
        default=True,
        description=(
            "Set False only when the scenario can never be carried out — it contradicts the app, "
            "or the screen it needs cannot be reached from here. Not yet being on the right screen "
            "is not infeasible: navigate there instead."
        ),
    )
    infeasibility_reason: Optional[str] = Field(
        default=None,
        description="Detailed reason why the scenario is infeasible. None if feasible.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_common_provider_shapes(cls, value):
        """Normalize equivalent shapes produced despite advisory JSON schemas."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        verifications = normalized.get("rule_verifications")
        if isinstance(verifications, dict):
            normalized["rule_verifications"] = [
                {
                    "rule": rule,
                    "status": status,
                    "reasoning": "Provider returned a compact rule-status mapping.",
                }
                for rule, status in verifications.items()
            ]
        for verification in normalized.get("rule_verifications") or []:
            if isinstance(verification, dict) and "reasoning" not in verification:
                verification["reasoning"] = verification.pop(
                    "details", "No reasoning supplied by the provider."
                )
        for issue in normalized.get("detected_issues") or []:
            if not isinstance(issue, dict):
                continue
            if "description" not in issue and "issue" in issue:
                issue["description"] = issue.pop("issue")
            issue.setdefault("issue_type", "qa_issue")
        return normalized


class MarksQAStepResponse(QAStepResponse):
    """The QA step response used in marks mode, where targets have no coordinates."""

    action: MarksDeviceAction = Field(
        ..., description="The single next action, naming its target by number."
    )


class QAPlanner(DevicePlanner):
    """
    Planning + QA audit in one LLM call.
    Instantiated per-scenario by QAAgent, carrying scenario-specific context.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_multimodal_vision: bool,
        test_case,
        security_rules: List[str] = None,
        scrubber=None,
        grounding: str = GROUNDING_COORDINATES,
        mark_detail: str = DETAIL_NUMBERS,
    ):
        super().__init__(llm_client, use_multimodal_vision, grounding, mark_detail)
        self._test_case = test_case
        from ronin.utils.secrets import SecretScrubber

        # Shared with the device agent so a value declared once is masked everywhere.
        self._scrubber = scrubber or SecretScrubber()
        self._security_rules = security_rules or []

    def step_response_model(self):
        """The QA schema the model answers in, matched to the grounding mode."""
        if self.grounding == GROUNDING_MARKS:
            return MarksQAStepResponse
        return QAStepResponse

    def plan_step(
        self,
        goal: List[str] | str,
        gui_elements=None,
        history: str = "",
        trail: Optional[str] = None,
        drift_warning: Optional[str] = None,
        loop_nudge: Optional[str] = None,
        current_image_bytes: bytes = b"",
        visual_history: List[bytes] = None,
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
    ) -> QAStepResponse:
        from ronin.utils.redactor import redact_elements, redact_text

        if gui_elements:
            gui_elements = redact_elements(gui_elements)

        if isinstance(goal, str):
            goal_lines = [goal]
        else:
            goal_lines = goal or []

        prompt = self._build_qa_prompt(
            goal_lines=goal_lines,
            gui_elements=gui_elements,
            history=history,
            trail=trail,
            drift_warning=drift_warning,
            loop_nudge=loop_nudge,
            keyboard_open=keyboard_open,
            marks=marks,
            screen_height=screen_height,
        )

        message = MultiPartMessage()
        # EXPERIMENT: visual history disabled — testing whether stale screenshot coordinates
        # cause LLM hallucination. Rich REACTION lines in action history replace visual context.
        # for idx, img_bytes in enumerate(visual_history or []):
        #     message.add_text(f"Screenshot from step {idx + 1}:")
        #     message.add_image(img_bytes)
        message.add_text("Current screenshot (newest):")
        message.add_image(current_image_bytes)
        message.add_text(prompt)

        logger.info("Querying QAPlanner for next step with QA auditing...")
        response: QAStepResponse = self.llm.generate_response(
            [message], self.step_response_model()
        )

        def clean(text):
            return self._scrubber.scrub(redact_text(text))

        response.reasoning = clean(response.reasoning)
        response.observation = clean(response.observation)
        response.evaluation_previous_goal = clean(response.evaluation_previous_goal)
        response.infeasibility_reason = clean(response.infeasibility_reason)
        for issue in response.detected_issues:
            issue.description = clean(issue.description)
        for verify in response.rule_verifications:
            verify.reasoning = clean(verify.reasoning)
            verify.rule = clean(verify.rule)

        return response

    def _build_qa_prompt(
        self,
        goal_lines: List[str],
        gui_elements,
        history: str,
        trail: Optional[str],
        drift_warning: Optional[str],
        loop_nudge: Optional[str],
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
    ) -> str:
        expected_outcomes = self._test_case.expected_outcomes or []
        security_rules = self._security_rules

        prompt = f"""You are a closed-loop mobile E2E testing agent performing QA validation on a mobile app.
Your task is to execute the following test scenario and verify all expected outcomes and security rules.
Each scenario begins with a fresh app launch. The app always starts on the Login Screen.
Complete the full user journey from the beginning — do not assume any state or progress from a previous scenario.

### Test Scenario Instructions
{chr(10).join(f"- {step}" for step in goal_lines)}

### Expected Outcomes / Assertions
{chr(10).join(f"- [expected:{index}] {o}" for index, o in enumerate(expected_outcomes)) if expected_outcomes else "None specified."}

### Custom Security Rules to Enforce
{chr(10).join(f"- [security:{index}] {r}" for index, r in enumerate(security_rules)) if security_rules else "None specified."}

### Current UI Elements Detected {self._target_heading()}
{self.format_targets(gui_elements, marks, screen_height)}
{"" if keyboard_open is None else chr(10) + "### Keyboard State" + chr(10) + ("VISIBLE — the soft keyboard is currently displayed. Elements in the lower ~40% of the screen may be keyboard keys, not app UI." if keyboard_open else "HIDDEN — the soft keyboard is not visible. All detected elements are app UI.")}

### Action History (with system-observed outcomes)
{history or "No actions executed yet."}
"""
        if drift_warning:
            prompt += f"""
### CONTEXT DRIFT WARNING
{drift_warning}
"""
        if trail:
            prompt += f"""
### Session Trail — App Knowledge From This Session
{trail}
"""
        if loop_nudge:
            prompt += f"""
### LOOP DETECTION ADVISORY
{loop_nudge}
"""
        prompt += """
### QA Auditing Instructions
Inspect the visual screenshots and audit the current screen for defects:
1. **Visual/Layout Bugs**: Scan for diagonal black-and-yellow overflow stripes, clipped text, overlapping buttons, misaligned headers, or empty image containers.
2. **Security Vulnerabilities**: Scan for plain-text passwords, unmasked PINs, exposed auth tokens, credit card digits, or accessible admin/debug routes.
3. **Dynamic Screen Assertions**: Generate context-specific assertions for the current layout (e.g. "Login button is visible and enabled", "Password field masks input"). Verify each as passed, failed, or untested.
4. **Rule Verifications**: For each expected outcome and custom security rule, copy its bracketed ID into `rule_id` and determine its status: "passed", "failed", or "untested".
   - Report what THIS screen shows. "failed" means you can see the rule broken here. A rule you have not reached yet is "untested", never "failed" — you will be asked again on every screen, and the last answer is the one that counts.
   - Set `applicable` to false when this screen has nothing the rule could apply to, and use "untested": a card-masking rule on a screen with no card field is not evidence of anything. Never mark a rule "passed" because its subject is absent.
   - Dynamic assertions may omit `rule_id`.
5. **Scenario Feasibility**: A scenario starts from a freshly launched app, so the screen it targets usually needs navigating to first. If its elements are not here yet, work towards them — sign in, open the relevant section — and keep going. Mark `is_scenario_feasible = False` only when the scenario is genuinely impossible: it contradicts how the app works, or its target cannot be reached from here at all.
"""
        prompt += self._action_vocabulary() + self._action_constraints()
        prompt += """
### Response Instructions
1. Set `current_screen` to the descriptive name of the current screen.
2. Set `evaluation_previous_goal`: honestly assess the previous action outcome. Empty on first step.
3. Set `observation`: describe only what is currently visible on the screen (UI state, visible fields, keyboard, overlays). No action plans here.
4. Set `reasoning`: explain why you chose the next action based on the observation.
5. Audit QA issues and populate `detected_issues` (each must include `severity`: low, medium, high, or critical).
6. Evaluate all expected outcomes, security rules, and dynamic assertions; populate `rule_verifications`.
7. If the scenario's target screen has not been reached yet, navigate towards it. Set `is_scenario_feasible = False` with an `infeasibility_reason` and action `done` only when it is genuinely unreachable or contradicts the app.
8. If all objectives are observably met, set `goal_satisfied = true`, describe the visible proof in `goal_evidence`, and set action to `done`. Never infer completion from the session trail or prior scenarios.
9. Otherwise, set `action` to the single next action to execute.
10. Set `external_navigation = True` if the action will temporarily leave the app.
"""
        return prompt
