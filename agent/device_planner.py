"""
DevicePlanner — core device-control planning layer.
Builds prompts and calls the LLM to decide the next action.
No QA concerns: those live in ronin_testing/qa_planner.py.
"""

import logging
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field

from agent.actuation.actions import (
    CAPABILITY_SCHEMAS,
    Action,
    ActionType,
    CapabilityType,
)
from llm.llm_client import LLMClient
from llm.messages import MultiPartMessage

logger = logging.getLogger(__name__)


class DoneAction(BaseModel):
    """Signals that the goal is complete or that execution cannot continue."""

    type: Literal["done"] = "done"
    reason: str = Field(
        ..., description="Why the goal is complete, or why execution must stop."
    )


DeviceAction = Annotated[Action | DoneAction, Field(discriminator="type")]


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


class DevicePlanner:
    """Builds prompts and calls the LLM to determine the next device action."""

    def __init__(self, llm_client: LLMClient, use_multimodal_vision: bool = False):
        self.llm = llm_client
        self.use_multimodal_vision = use_multimodal_vision

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
    ) -> DeviceStepResponse:
        """Plan the next action given current screen state and accumulated history."""
        from utils.redactor import redact_elements, redact_text

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
        )

        message = MultiPartMessage()
        for idx, img_bytes in enumerate(visual_history or []):
            message.add_text(f"Screenshot from step {idx + 1}:")
            message.add_image(img_bytes)
        message.add_text("Current screenshot (newest):")
        message.add_image(current_image_bytes)
        message.add_text(prompt)

        logger.info("Querying DevicePlanner for next step...")
        response: DeviceStepResponse = self.llm.generate_response(
            [message], DeviceStepResponse
        )

        response.reasoning = redact_text(response.reasoning)
        response.evaluation_previous_goal = redact_text(
            response.evaluation_previous_goal
        )
        return response

    def _build_prompt(
        self,
        goal_lines: List[str],
        gui_elements,
        history: str,
        trail: Optional[str],
        drift_warning: Optional[str],
        loop_nudge: Optional[str],
    ) -> str:
        caps_desc = "\n".join(
            f"   - {cap.value}: {CAPABILITY_SCHEMAS.get(cap, '{}')}"
            for cap in CapabilityType
        )

        prompt = f"""You are an AI agent controlling a mobile app to achieve a goal.
Your job is to execute the steps below by interacting with the app one action at a time.

### Goal / Instructions
{chr(10).join(f"- {step}" for step in goal_lines)}

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
### Allowed Action Types
Each action must have a `type` field set to one of:
- "tap": Tap at coordinates. Fields: `type`, `coordinates` (x, y), `reason`.
- "type": Type text into a field. Fields: `type`, `text`, `coordinates` (x, y), `reason`.
- "clear": Clear a text field. Fields: `type`, `coordinates` (x, y), `reason`.
- "double_tap": Double-tap. Fields: `type`, `coordinates` (x, y), `reason`.
- "long_press": Long press. Fields: `type`, `duration` (float seconds), `coordinates` (x, y), `reason`.
- "drag_and_drop": Drag from source to destination. Fields: `type`, `source_x`, `source_y`, `destination_x`, `destination_y`, `reason`.
- "key_event": Trigger a key event. Fields: `type`, `key` (int keycode: 4=back, 3=home, 66=enter, 67=backspace), `reason`.
- "swipe": Swipe from start to end. Fields: `type`, `start_x`, `start_y`, `end_x`, `end_y`, `reason`.
- "capability": System-level control. Fields: `type`, `capability`, `parameters` (dict), `reason`.
  Available capabilities:
{caps_desc}
- "done": Goal complete or execution must stop. Fields: `type`, `reason`.

CRITICAL CONSTRAINTS:
1. Use "tap" for tapping/clicking and "type" for entering text. Never use "click" or "send_keys".
2. For "type", you MUST provide both `text` AND `coordinates` of the target field.
3. Before typing into a field, first send a "tap" action to focus it.
4. The soft keyboard creates a touch-blocking overlay across the bottom portion of the screen. YOLO may still detect elements in that area, but tapping their coordinates will hit keyboard keys instead of the element. If the keyboard is visible and your next tap target is in the lower half of the screen, dismiss the keyboard first with key_event 4 — on Android, back always dismisses the soft keyboard without navigating away.
5. Do NOT repeat an action that appears in your history unless the system outcome or your self-assessment shows it failed or was undone. Inability to visually confirm a result is NOT evidence of failure.
6. When you tap a text field and the keyboard appears, the screen layout shifts and YOLO-detected element positions change. For a TYPE action into the same field you just tapped, use the EXACT same coordinates as the TAP — do not re-derive coordinates from the post-keyboard YOLO scan.
7. You MUST use center coordinates of the detected UI elements listed above. Do not use coordinates outside the detected bounding boxes.
"""
        prompt += """
### Response Instructions
1. Set `current_screen` to the descriptive name of the screen you currently see.
2. Set `evaluation_previous_goal`: honestly assess whether your PREVIOUS action succeeded, failed, or is uncertain. Leave empty on the first step.
3. Set `observation`: describe only what is currently visible.
4. Set `reasoning`: explain why the chosen action is the best next step.
4. Set `action`: choose exactly ONE action. If the goal is fully complete, use `done`.
5. Set `external_navigation` to True if the action will temporarily leave the app.
"""
        return prompt

    def _format_elements(self, elements) -> str:
        lines = []
        for el in elements:
            cx = (el.bounding_box.x_min + el.bounding_box.x_max) // 2
            cy = (el.bounding_box.y_min + el.bounding_box.y_max) // 2
            if el.text:
                lines.append(
                    f"- {el.element} | text: '{el.text}' | center: ({cx}, {cy})"
                )
            else:
                lines.append(f"- {el.element} | center: ({cx}, {cy})")
        return "\n".join(lines)
