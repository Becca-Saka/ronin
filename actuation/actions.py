from enum import Enum

from pydantic import BaseModel, Field


# --- Basic Building Blocks ---
class Coordinates(BaseModel):
    x: int = Field(..., description="The x coordinate in pixels.")
    y: int = Field(..., description="The y coordinate in pixels.")


class ActionType(Enum):
    TYPE = "type"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    DRAG_AND_DROP = "drag_and_drop"
    KEY_EVENT = "key_event"  # key events (4=back, 3=home, 66=enter, 67=backspace)
    SWIPE = "swipe"


# --- Action Definitions ---


class TypeAction(BaseModel):
    type: ActionType = ActionType.TYPE
    text: str = Field(..., description="The text of the action.")
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates to move to."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class TapAction(BaseModel):
    type: ActionType = ActionType.TAP
    coordinates: Coordinates = Field(..., description="The screen coordinates to tap.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class DoubleTapAction(BaseModel):
    type: ActionType = ActionType.DOUBLE_TAP
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates to double tap."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class LongPressAction(BaseModel):
    type: ActionType = ActionType.LONG_PRESS
    duration: float = Field(
        ..., description="The duration of the long press in seconds."
    )
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates to long press."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class DragAndDropAction(BaseModel):
    type: ActionType = ActionType.DRAG_AND_DROP
    source_x: int = Field(..., description="The x coordinate to drag from.")
    source_y: int = Field(..., description="The y coordinate to drag from.")
    destination_x: int = Field(..., description="The x coordinate to drag to.")
    destination_y: int = Field(..., description="The y coordinate to drag to.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class KeyEventAction(BaseModel):
    type: ActionType = ActionType.KEY_EVENT
    key: int = Field(..., description="The key code to press.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class SwipeAction(BaseModel):
    type: ActionType = ActionType.SWIPE
    start_x: int = Field(..., description="The x coordinate to start the swipe.")
    start_y: int = Field(..., description="The y coordinate to start the swipe.")
    end_x: int = Field(..., description="The x coordinate to end the swipe.")
    end_y: int = Field(..., description="The y coordinate to end the swipe.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


# --- The Union of All Actions ---
Action = (
    TypeAction
    | TapAction
    | DoubleTapAction
    | LongPressAction
    | DragAndDropAction
    | KeyEventAction
    | SwipeAction
)


# --- The Final Output Schema ---
class ActionPlan(BaseModel):
    instruction: str = Field(
        ..., description="The normalized instruction to achieve the goal."
    )
    actions: list[Action] = Field(
        ...,
        description="The sequence of machine-executable actions to achieve the goal.",
    )
