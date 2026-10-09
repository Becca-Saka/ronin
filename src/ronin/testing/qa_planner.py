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

from ronin.agent.device_planner import DevicePlanner, DeviceStepResponse
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
        keyboard_open: Optional[bool] = None,
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
        response: QAStepResponse = self.llm.generate_response([message], QAStepResponse)

        response.reasoning = redact_text(response.reasoning)
        response.observation = redact_text(response.observation)
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
        keyboard_open: Optional[bool] = None,
    ) -> str:
        from ronin.agent.actuation.actions import CAPABILITY_SCHEMAS, CapabilityType

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
1. Action vocabulary is STRICT:
   - Use "tap" for all touch interactions
   - Use "type" only for text input
   - Never use "click", "send_keys", or any alternative synonyms

2. TYPE action requirements:
   - Must include BOTH `text` and `coordinates`
   - Coordinates MUST refer to the input field (not keyboard keys or overlays)

3. REQUIRED SEQUENCE FOR TEXT INPUT:
   Step A: tap(input_field_coordinates)
   Step B: wait for keyboard to appear
   Step C: type(text, same input_field_coordinates)

4. KEYBOARD STATE RULE:
   Assume keyboard is VISIBLE if:
   - A previous "tap" targeted a text field, OR
   - history shows a recent "type", OR
   - UI bottom region is partially occluded in detection

5. KEYBOARD BLOCKING RULE:
   If keyboard is visible:
   - Any element in the lower 45% of the screen is NOT interactable
   - Taps in this region will hit keyboard keys instead of UI elements
   - Before tapping such elements, dismiss keyboard using key_event 4

6. KEYBOARD DISMISSAL RULE:
   Use key_event 4 ONLY when:
   - Target element is in lower screen region, OR
   - Current action requires full-screen interaction (submit, login, continue buttons near bottom)

   After key_event 4:
   - Treat next UI element list as a NEW layout (coordinates fully refreshed)

7. NO ACTION REPEATS:
   Do not repeat any action in history unless:
   - previous attempt explicitly failed, OR
   - system outcome confirms no state change occurred

   Lack of visual confirmation is NOT failure evidence.

8. COORDINATE AUTHORITY RULE:
   - Always use coordinates from CURRENT UI ELEMENTS ONLY
   - Do NOT reuse coordinates from previous screens after any UI change event (keyboard open/close, navigation, modal, scroll)
"""
        # CRITICAL CONSTRAINTS:
        # 1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
        # 2. For "type", provide both `text` AND `coordinates` of the target field.
        # 3. Before typing, first send a "tap" to focus the field.
        # 4. The soft keyboard blocks taps on anything behind it. If your next target is a button or control in the lower half of the screen (e.g. Login, Submit), dismiss the keyboard first with key_event 4 — on Android, back dismisses the soft keyboard without leaving the app. After dismissal, use the fresh element list for coordinates.
        # 5. Do NOT repeat an action already in history unless outcome or self-assessment shows it failed or was undone. Inability to see a result visually is NOT evidence of failure.
        # 6. When the keyboard is visible and you need to TYPE into a field, use text_field/input elements from the CURRENT scan in the upper portion of the screen (above the keyboard). Ignore bottom-portion detections — those are keyboard keys.
        # """
        #         if self.use_multimodal_vision:
        #             prompt += """
        # 7. Prefer element center coordinates from the Current UI Elements list. If a target is missing because the keyboard obscures it, dismiss the keyboard (rule 4) first. You may also derive coordinates from older screenshots when YOLO missed a visible element.
        # """
        #         else:
        #             prompt += """
        # 7. MUST use center coordinates from the Current UI Elements list. If a target is missing because the keyboard obscures it, do not guess coordinates from older screenshots — dismiss the keyboard (rule 4) and plan from the next scan. Older screenshots are context only.
        # """
        prompt += """
### Response Instructions
1. Set `current_screen` to the descriptive name of the current screen.
2. Set `evaluation_previous_goal`: honestly assess the previous action outcome. Empty on first step.
3. Set `observation`: describe only what is currently visible on the screen (UI state, visible fields, keyboard, overlays). No action plans here.
4. Set `reasoning`: explain why you chose the next action based on the observation.
5. Audit QA issues and populate `detected_issues` (each must include `severity`: low, medium, high, or critical).
6. Evaluate all expected outcomes, security rules, and dynamic assertions; populate `rule_verifications`.
7. If scenario instructions reference elements that do not exist here, set `is_scenario_feasible = False`, fill `infeasibility_reason`, set action to `done`.
8. If all objectives are met or a critical crash terminates execution, set action to `done`.
9. Otherwise, set `action` to the single next action to execute.
10. Set `external_navigation = True` if the action will temporarily leave the app.
"""
        return prompt
