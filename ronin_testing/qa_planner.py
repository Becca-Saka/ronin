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

from pydantic import Field

from agent.device_planner import DevicePlanner, DeviceStepResponse
from llm.llm_client import LLMClient
from llm.messages import MultiPartMessage
from ronin_testing.models import QAIssue, RuleVerification

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
            "Set False if the scenario instructions are invalid or impossible for the current screen "
            "(e.g. references fields that do not exist). Otherwise True."
        ),
    )
    infeasibility_reason: Optional[str] = Field(
        default=None,
        description="Detailed reason why the scenario is infeasible. None if feasible.",
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
    ):
        super().__init__(llm_client, use_multimodal_vision)
        self._test_case = test_case
        self._security_rules = security_rules or []

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
    ) -> QAStepResponse:
        from utils.redactor import redact_elements, redact_text

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
        )

        message = MultiPartMessage()
        for idx, img_bytes in enumerate(visual_history or []):
            message.add_text(f"Screenshot from step {idx + 1}:")
            message.add_image(img_bytes)
        message.add_text("Current screenshot (newest):")
        message.add_image(current_image_bytes)
        message.add_text(prompt)

        logger.info("Querying QAPlanner for next step with QA auditing...")
        response: QAStepResponse = self.llm.generate_response([message], QAStepResponse)

        response.reasoning = redact_text(response.reasoning)
        response.evaluation_previous_goal = redact_text(
            response.evaluation_previous_goal
        )
        for issue in response.detected_issues:
            issue.description = redact_text(issue.description)
        for verify in response.rule_verifications:
            verify.reasoning = redact_text(verify.reasoning)

        return response

    def _build_qa_prompt(
        self,
        goal_lines: List[str],
        gui_elements,
        history: str,
        trail: Optional[str],
        drift_warning: Optional[str],
        loop_nudge: Optional[str],
    ) -> str:
        from agent.actuation.actions import CapabilityType, CAPABILITY_SCHEMAS

        expected_outcomes = self._test_case.expected_outcomes or []
        security_rules = self._security_rules

        caps_desc = "\n".join(
            f"   - {cap.value}: {CAPABILITY_SCHEMAS.get(cap, '{}')}"
            for cap in CapabilityType
        )

        prompt = f"""You are a closed-loop mobile E2E testing agent performing QA validation on a mobile app.
Your task is to execute the following test scenario and verify all expected outcomes and security rules.
Each scenario begins with a fresh app launch. The app always starts on the Login Screen.
Complete the full user journey from the beginning — do not assume any state or progress from a previous scenario.

### Test Scenario Instructions
{chr(10).join(f"- {step}" for step in goal_lines)}

### Expected Outcomes / Assertions
{chr(10).join(f"- {o}" for o in expected_outcomes) if expected_outcomes else "None specified."}

### Custom Security Rules to Enforce
{chr(10).join(f"- {r}" for r in security_rules) if security_rules else "None specified."}

### Current UI Elements Detected (YOLO/OCR)
{self._format_elements(gui_elements) if gui_elements else "None detected."}

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
        prompt += f"""
### QA Auditing Instructions
Inspect the visual screenshots and audit the current screen for defects:
1. **Visual/Layout Bugs**: Scan for diagonal black-and-yellow overflow stripes, clipped text, overlapping buttons, misaligned headers, or empty image containers.
2. **Security Vulnerabilities**: Scan for plain-text passwords, unmasked PINs, exposed auth tokens, credit card digits, or accessible admin/debug routes.
3. **Dynamic Screen Assertions**: Generate context-specific assertions for the current layout (e.g. "Login button is visible and enabled", "Password field masks input"). Verify each as passed, failed, or untested.
4. **Rule Verifications**: For each expected outcome, custom security rule, and dynamic assertion, determine its status on the current screen: "passed", "failed", or "untested".
5. **Scenario Feasibility**: Verify the scenario instructions are valid for the current screen. If they require elements that do not exist, set `is_scenario_feasible = False` and fill `infeasibility_reason`.

### Allowed Action Types
Each action must have a `type` field set to one of:
- "tap": Tap at coordinates. Fields: `type`, `coordinates` (x, y), `reason`.
- "type": Type text into a field. Fields: `type`, `text`, `coordinates` (x, y), `reason`.
- "clear": Clear a text field. Fields: `type`, `coordinates` (x, y), `reason`.
- "double_tap": Double-tap. Fields: `type`, `coordinates` (x, y), `reason`.
- "long_press": Long press. Fields: `type`, `duration` (float seconds), `coordinates` (x, y), `reason`.
- "drag_and_drop": Drag. Fields: `type`, `source_x`, `source_y`, `destination_x`, `destination_y`, `reason`.
- "key_event": Trigger a key event. Fields: `type`, `key` (4=back, 3=home, 66=enter, 67=backspace), `reason`.
- "swipe": Swipe. Fields: `type`, `start_x`, `start_y`, `end_x`, `end_y`, `reason`.
- "capability": System-level control. Fields: `type`, `capability`, `parameters` (dict), `reason`.
  Available capabilities:
{caps_desc}
- "done": Goal complete or must stop. Fields: `type`, `reason`.

CRITICAL CONSTRAINTS:
1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
2. For "type", provide both `text` AND `coordinates` of the target field.
3. Before typing, first send a "tap" to focus the field.
4. The soft keyboard creates a touch-blocking overlay across the bottom portion of the screen. YOLO may still detect elements in that area, but tapping their coordinates will hit keyboard keys instead of the element. If the keyboard is visible and your next tap target is in the lower half of the screen, dismiss the keyboard first with key_event 4 — on Android, back always dismisses the soft keyboard without navigating away.
5. Do NOT repeat an action already in history unless outcome or self-assessment shows it failed or was undone. Inability to see a result visually is NOT evidence of failure.
7. When you tap a text field and the keyboard appears, the screen layout shifts and YOLO-detected element positions change. For a TYPE action into the same field you just tapped, use the EXACT same coordinates as the TAP — do not re-derive coordinates from the post-keyboard YOLO scan.


"""
        if self.use_multimodal_vision:
            prompt += """
6. Prefer element center coordinates. If multimodal override is ENABLED, you may use custom coordinates based on visual inspection when YOLO missed an element.
"""
        else:
            prompt += """
6. MUST use center coordinates of the detected UI elements. Do not use coordinates outside the detected bounding boxes.
"""
        prompt += """
### Response Instructions
1. Set `current_screen` to the descriptive name of the current screen.
2. Set `evaluation_previous_goal`: honestly assess the previous action outcome. Empty on first step.
3. Set `reasoning`: explain what you observe and why you chose the next action.
4. Audit QA issues and populate `detected_issues`.
5. Evaluate all expected outcomes, security rules, and dynamic assertions; populate `rule_verifications`.
6. If scenario instructions reference elements that do not exist here, set `is_scenario_feasible = False`, fill `infeasibility_reason`, set action to `done`.
7. If all objectives are met or a critical crash terminates execution, set action to `done`.
8. Otherwise, set `action` to the single next action to execute.
9. Set `external_navigation = True` if the action will temporarily leave the app.
"""
        return prompt
