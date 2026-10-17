"""The shape of one planning answer, apart from the prompt that asks for it."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ronin.agent.actuation.actions import Action, MarksAction


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
