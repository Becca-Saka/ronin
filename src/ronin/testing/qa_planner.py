"""QAPlanner — DevicePlanner, told what the scenario is trying to do.

It decides the next action and nothing else. QAAuditor judges the screen, and
reads neither the plan nor this prompt.

The scenario's own context stays here: instructions, expected outcomes and
security rules. Deciding what to do next needs all three. Judging does not, and
that is why the auditor does not receive them.
"""

import logging
from typing import List, Optional

from pydantic import Field

from ronin.agent.device_planner import (
    GROUNDING_COORDINATES,
    GROUNDING_MARKS,
    DevicePlanner,
    DeviceStepResponse,
    MarksDeviceAction,
)
from ronin.llm.llm_client import LLMClient
from ronin.llm.messages import MultiPartMessage
from ronin.testing import rules

logger = logging.getLogger(__name__)


class QAStepResponse(DeviceStepResponse):
    """What the planner answers with: an action, and whether it can proceed.

    The verdicts belong to ronin.testing.qa_auditor, which judges the screen
    without reading the plan.
    """

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
    ):
        super().__init__(llm_client, use_multimodal_vision, grounding)
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
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
    ) -> QAStepResponse:
        from ronin.utils.redactor import redact_elements, safe_text

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
        message.add_text("Current screenshot (newest):")
        message.add_image(current_image_bytes)
        message.add_text(prompt)

        logger.info("Querying QAPlanner for next step with QA auditing...")
        response: QAStepResponse = self.llm.generate_response(
            [message], self.step_response_model()
        )

        def clean(text):
            return safe_text(text, self._scrubber)

        response.reasoning = clean(response.reasoning)
        response.observation = clean(response.observation)
        response.evaluation_previous_goal = clean(response.evaluation_previous_goal)
        response.infeasibility_reason = clean(response.infeasibility_reason)

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
{rules.listing(expected_outcomes, rules.EXPECTED)}

### Custom Security Rules to Enforce
{rules.listing(security_rules, rules.SECURITY)}

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
### Scenario Feasibility
A scenario starts from a freshly launched app, so the screen it targets usually
needs navigating to first. If its elements are not here yet, work towards them —
sign in, open the relevant section — and keep going. Mark `is_scenario_feasible`
false only when the scenario is genuinely impossible: it contradicts how the app
works, or its target cannot be reached from here at all.

Do not grade the app. Whether the expected outcomes hold, whether a security
rule is met, and whether anything on screen looks wrong are judged separately,
by something that is not told what you decided to do.
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
