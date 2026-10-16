"""
DevicePlanner — core device-control planning layer.
Builds prompts and calls the LLM to decide the next action.
No QA concerns: those live in ronin/testing/qa_planner.py.
"""

import logging
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field

from ronin.agent.actuation.actions import (
    CAPABILITY_SCHEMAS,
    Action,
    CapabilityType,
    MarksAction,
)
from ronin.agent.perception.marks import DETAIL_NUMBERS
from ronin.llm.llm_client import LLMClient
from ronin.llm.messages import MultiPartMessage

logger = logging.getLogger(__name__)

GROUNDING_COORDINATES = "coordinates"
GROUNDING_MARKS = "marks"


class DoneAction(BaseModel):
    """Signals that the goal is complete or that execution cannot continue."""

    type: Literal["done"] = "done"
    reason: str = Field(
        ..., description="Why the goal is complete, or why execution must stop."
    )


DeviceAction = Annotated[Action | DoneAction, Field(discriminator="type")]
MarksDeviceAction = Annotated[MarksAction | DoneAction, Field(discriminator="type")]


class DeviceStepResponse(BaseModel):
    """Single planning step response from the device agent."""

    current_screen: str = Field(
        ...,
        description="Descriptive name of the current screen/page (e.g. 'Login Screen', 'Home Feed').",
    )
    evaluation_previous_goal: str = Field(
        default="",
        description=(
            "Brief honest assessment of whether the previous action succeeded, failed, or produced "
            "an uncertain result. Empty string on the very first step. "
            "Examples: 'Success — screen navigated to Home', "
            "'Failed — button did not respond', "
            "'Uncertain — field unchanged but masked content cannot be confirmed visually'."
        ),
    )
    observation: str = Field(
        ...,
        description="Description of what is currently visible on the screen.",
    )
    reasoning: str = Field(
        ...,
        description="Explain why the chosen action is the best next step.",
    )
    goal_satisfied: bool = Field(
        default=False,
        description=(
            "True when the goal's stated outcome is observable on the current screen, before "
            "any further action. Judge the goal only — ignore follow-on work such as onboarding "
            "or permission prompts."
        ),
    )
    goal_evidence: str = Field(
        default="",
        description="What on the current screen shows the goal is met. Empty unless goal_satisfied.",
    )
    action: DeviceAction = Field(
        ...,
        description="The single action to execute next. Use done() when the goal is complete.",
    )
    external_navigation: bool = Field(
        default=False,
        description=(
            "Set True if this action is expected to temporarily navigate outside the SUT app "
            "(e.g. opening a browser link, system dialog, or external payment screen)."
        ),
    )


class MarksStepResponse(DeviceStepResponse):
    """The step response used in marks mode, where targets have no coordinates."""

    action: MarksDeviceAction = Field(
        ..., description="The single next action, naming its target by number."
    )


class DevicePlanner:
    # Class-level so a planner built without __init__ still answers sensibly.
    grounding = GROUNDING_COORDINATES
    mark_detail = DETAIL_NUMBERS

    """Builds prompts and calls the LLM to determine the next device action."""

    def __init__(
        self,
        llm_client: LLMClient,
        use_multimodal_vision: bool = False,
        grounding: str = GROUNDING_COORDINATES,
        mark_detail: str = DETAIL_NUMBERS,
    ):
        self.llm = llm_client
        self.use_multimodal_vision = use_multimodal_vision
        # "marks" offers numbered elements and no coordinates, so a target the
        # planner cannot find becomes a refusal rather than an invented point.
        self.grounding = grounding
        # What accompanies the numbered screenshot. See marks.describe.
        self.mark_detail = mark_detail

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
    ) -> DeviceStepResponse:
        """Plan the next action given current screen state and accumulated history."""
        from ronin.utils.redactor import redact_elements, redact_text

        if isinstance(goal, str):
            goal_lines = [goal]
        else:
            goal_lines = goal or []

        if gui_elements:
            gui_elements = redact_elements(gui_elements)

        prompt = self._build_prompt(
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

        logger.info("Querying DevicePlanner for next step...")
        response: DeviceStepResponse = self.llm.generate_response(
            [message], self.step_response_model()
        )

        response.reasoning = redact_text(response.reasoning)
        response.evaluation_previous_goal = redact_text(
            response.evaluation_previous_goal
        )
        return response

    def _action_vocabulary(self) -> str:
        """The action schema both planners offer the model.

        Defined once: QAPlanner subclasses this planner, and when the two kept
        their own copies they drifted into contradicting each other.
        """
        caps_desc = "\n".join(
            f"   - {cap.value}: {CAPABILITY_SCHEMAS.get(cap, '{}')}"
            for cap in CapabilityType
        )
        if self.grounding == GROUNDING_MARKS:
            targeting = """
### Allowed Action Types
Each action must have a `type` field set to one of:
- "tap": Tap an element. Fields: `type`, `mark` (int), `reason`.
- "type": Type text into an element. Fields: `type`, `text`, `mark` (int), `reason`.
- "clear": Clear a text field. Fields: `type`, `mark` (int), `reason`.
- "double_tap": Double-tap an element. Fields: `type`, `mark` (int), `reason`.
- "long_press": Long press an element. Fields: `type`, `duration` (float seconds), `mark` (int), `reason`.
- "target_not_listed": The element you need is not numbered on the screenshot. Fields: `type`, `description` (what it looks like and roughly where), `reason`."""
        else:
            targeting = """
### Allowed Action Types
Each action must have a `type` field set to one of:
- "tap": Tap at coordinates. Fields: `type`, `coordinates` as {"x": int, "y": int}, `reason`.
- "type": Type text into a field. Fields: `type`, `text`, `coordinates` as {"x": int, "y": int}, `reason`.
- "clear": Clear a text field. Fields: `type`, `coordinates` as {"x": int, "y": int}, `reason`.
- "double_tap": Double-tap. Fields: `type`, `coordinates` as {"x": int, "y": int}, `reason`.
- "long_press": Long press. Fields: `type`, `duration` (float seconds), `coordinates` as {"x": int, "y": int}, `reason`."""

        return (
            targeting
            + """
- "drag_and_drop": Drag from source to destination. Fields: `type`, `source_x`, `source_y`, `destination_x`, `destination_y`, `reason`.
- "key_event": Trigger a key event. Fields: `type`, `key` (int keycode: 4=back, 3=home, 66=enter, 67=backspace), `reason`.
- "swipe": Swipe from start to end. Fields: `type`, `start_x`, `start_y`, `end_x`, `end_y`, `reason`.
- "wait": Let the screen finish loading or animating. Fields: `type`, `duration` (float seconds), `reason`. Use this instead of giving up when the screen looks empty or half-drawn.
- "capability": System-level control. Fields: `type`, `capability`, `parameters` (dict), `reason`.
  Available capabilities:
"""
            + caps_desc
            + """
- "done": Goal complete or execution must stop. Fields: `type`, `reason`.
"""
        )

    def _action_constraints(self) -> str:
        """The rules for choosing an action. Shared with QAPlanner — see above."""
        if self.grounding == GROUNDING_MARKS:
            return """
CRITICAL CONSTRAINTS:
1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
2. Every element on the screenshot carries a number. Name the number in `mark`. Never send coordinates — they are not accepted.
3. "type" focuses the element itself — do not spend a separate "tap" first. Only tap when you need focus without entering text.
4. Choose only a number that appears in the list below. If what you need is not numbered, use "target_not_listed" and describe it. Do not pick the nearest number instead, and do not guess.
5. The soft keyboard blocks taps on anything behind it. If your next target is in the lower region (e.g. Login, Submit), dismiss the keyboard first with key_event 4 — on Android, back dismisses the soft keyboard without leaving the app.
6. Do NOT repeat an action that appears in your history unless the system outcome or your self-assessment shows it failed or was undone. Inability to visually confirm a result is NOT evidence of failure.
7. The numbers are redrawn every step. Use the numbers from the CURRENT screenshot and list, never one you remember from an earlier step.
"""

        return """
CRITICAL CONSTRAINTS:
1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
2. For "type", you MUST provide both `text` AND `coordinates` of the target field.
3. "type" focuses the field at those coordinates itself — do not spend a separate "tap" first. Only tap when you need focus without entering text.
4. The soft keyboard blocks taps on anything behind it. If your next target is a button or control in the lower half of the screen (e.g. Login, Submit), dismiss the keyboard first with key_event 4 — on Android, back dismisses the soft keyboard without leaving the app. After dismissal, use the fresh element list for coordinates.
5. Do NOT repeat an action that appears in your history unless the system outcome or your self-assessment shows it failed or was undone. Inability to visually confirm a result is NOT evidence of failure.
6. When the keyboard is visible and you need to TYPE into a field, use text_field/input elements from the CURRENT scan in the upper portion of the screen (above the keyboard). Ignore bottom-portion detections — those are keyboard keys.
7. You MUST use center coordinates from the Current UI Elements list. If a target you need is missing because the keyboard obscures it, do not guess coordinates from older screenshots — dismiss the keyboard (rule 4) and plan from the next scan. Older screenshots in the message are context only; they are not a coordinate source unless multimodal override is enabled.
8. After any event that changes the layout — keyboard opening or closing, navigation, a modal, a scroll — treat the next element list as a new layout and re-read every coordinate from it.
"""

    def _build_prompt(
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
        kb_line = ""
        if keyboard_open is True:
            kb_line = "\n### Keyboard State\nVISIBLE — the soft keyboard is currently displayed. Elements in the lower ~40% of the screen may be keyboard keys, not app UI.\n"
        elif keyboard_open is False:
            kb_line = "\n### Keyboard State\nHIDDEN — the soft keyboard is not visible. All detected elements are app UI.\n"

        prompt = f"""You are an AI agent controlling a mobile app to achieve a goal.
Your job is to execute the steps below by interacting with the app one action at a time.

### Goal / Instructions
{chr(10).join(f"- {step}" for step in goal_lines)}

### Current UI Elements Detected {self._target_heading()}
{self.format_targets(gui_elements, marks, screen_height)}
{kb_line}
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
        prompt += self._action_vocabulary() + self._action_constraints()
        prompt += """
### Response Instructions
1. Set `current_screen` to the descriptive name of the screen you currently see.
2. Set `evaluation_previous_goal`: honestly assess whether your PREVIOUS action succeeded, failed, or is uncertain. Leave empty on the first step.
3. Set `observation`: describe only what is currently visible.
4. Set `reasoning`: explain why the chosen action is the best next step.
5. Set `goal_satisfied` True when the goal's outcome is already visible on this screen, and put the evidence in `goal_evidence`. Judge the goal as stated: a post-login onboarding or permission screen still means a sign-in goal is met.
6. Set `action`: choose exactly ONE action. If the goal is fully complete, use `done`.
7. Set `external_navigation` to True if the action will temporarily leave the app.
"""
        return prompt

    ROW_BAND = 60
    DUPLICATE_OVERLAP = 0.9

    @staticmethod
    def _center(el) -> tuple[int, int]:
        box = el.bounding_box
        return (box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2

    @staticmethod
    def _overlap(a, b) -> float:
        """Intersection over union of two bounding boxes."""
        left = max(a.x_min, b.x_min)
        top = max(a.y_min, b.y_min)
        right = min(a.x_max, b.x_max)
        bottom = min(a.y_max, b.y_max)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        areas = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (
            b.y_max - b.y_min
        )
        union = areas - intersection
        return intersection / union if union else 0.0

    def _deduplicate(self, elements) -> list:
        """Drops repeat detections of the same thing, which only add noise."""
        kept = []
        for el in elements:
            if any(
                el.text == other.text
                and self._overlap(el.bounding_box, other.bounding_box) >= self.DUPLICATE_OVERLAP
                for other in kept
            ):
                continue
            kept.append(el)
        return kept

    ROW_LABEL_LIMIT = 3

    def _row_label(self, el, elements) -> Optional[str]:
        """The texts sharing a row with an element, naming which row it is in.

        List screens repeat the same label on every row — eight 'Menu' buttons
        for eight restaurants — and coordinates alone have proven easy to
        mismatch by one row. All the row's text is offered rather than the
        nearest one, since the closest line is often a subtitle while the
        identifying name sits a few pixels further away.
        """
        _, cy = self._center(el)
        neighbours = []
        for other in elements:
            if other is el or not other.text or other.text == el.text:
                continue
            ox, oy = self._center(other)
            if abs(oy - cy) <= self.ROW_BAND:
                neighbours.append((abs(oy - cy), ox, other.text))
        if not neighbours:
            return None
        neighbours.sort()
        return ", ".join(f"'{text}'" for _, _, text in neighbours[: self.ROW_LABEL_LIMIT])

    def step_response_model(self):
        """The schema the model answers in, which differs by grounding mode."""
        if self.grounding == GROUNDING_MARKS:
            return MarksStepResponse
        return DeviceStepResponse

    def _target_heading(self) -> str:
        """Says where the targets are, which differs by mode and must not lie."""
        if self.grounding == GROUNDING_MARKS:
            return "(numbered on the screenshot)"
        return "(YOLO/OCR)"

    def format_targets(self, elements, marks=None, screen_height: int = 0) -> str:
        """The element list for the prompt, in whichever form the mode calls for."""
        if self.grounding == GROUNDING_MARKS:
            from ronin.agent.perception import marks as marks_module

            return marks_module.describe(marks or [], screen_height, self.mark_detail)
        return self._format_elements(elements)

    def _format_elements(self, elements) -> str:
        if not elements:
            return "None detected."
        elements = self._deduplicate(elements)
        elements = sorted(
            elements,
            key=lambda el: (self._center(el)[1] // 25, self._center(el)[0]),
        )

        repeated = {
            el.text
            for el in elements
            if el.text and sum(1 for other in elements if other.text == el.text) > 1
        }

        lines = []
        for el in elements:
            cx, cy = self._center(el)
            if el.text:
                line = f"- {el.element} | text: '{el.text}' | center: ({cx}, {cy})"
                if el.text in repeated:
                    row = self._row_label(el, elements)
                    if row:
                        line += f" | in row with: {row}"
            else:
                line = f"- {el.element} | center: ({cx}, {cy})"
            lines.append(line)
        return "\n".join(lines)
