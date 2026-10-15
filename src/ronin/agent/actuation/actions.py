from enum import Enum
from typing import Any, Literal, Annotated

from pydantic import BaseModel, Field, model_validator


# --- Basic Building Blocks ---
class Coordinates(BaseModel):
    x: int = Field(..., description="The x coordinate in pixels.")
    y: int = Field(..., description="The y coordinate in pixels.")

    @model_validator(mode="before")
    @classmethod
    def _accept_pairs(cls, value: Any) -> Any:
        """Accepts [x, y] as well as {"x": .., "y": ..}.

        Models routinely answer with a pair even when the schema asks for an
        object, and the request cannot insist: provider-side schema enforcement
        is off. Rejecting the pair fails the whole step response and kills the
        scenario, so the pair is read for what it plainly is.
        """
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"x": value[0], "y": value[1]}
        return value


class ActionType(str, Enum):
    TYPE = "type"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    DRAG_AND_DROP = "drag_and_drop"
    KEY_EVENT = "key_event"  # key events (4=back, 3=home, 66=enter, 67=backspace)
    SWIPE = "swipe"
    CAPABILITY = "capability"
    CLEAR = "clear"
    WAIT = "wait"


class CapabilityType(str, Enum):
    LOCATION = "location"
    NETWORK = "network"
    BATTERY = "battery"
    TIMEZONE = "timezone"
    LOCALE = "locale"
    PERMISSION = "permission"
    INTERRUPT = "interrupt"
    BIOMETRIC = "biometric"
    CLIPBOARD = "clipboard"
    DEEPLINK = "deeplink"
    APP_STATE = "app_state"
    MEDIA_INJECT = "media_inject"
    ACCESSIBILITY = "accessibility"


CAPABILITY_SCHEMAS = {
    CapabilityType.LOCATION: "{'latitude': float, 'longitude': float}",
    CapabilityType.NETWORK: "{'mode': 'offline' | 'online' | 'wifi_only' | 'data_only' | 'airplane'}",
    CapabilityType.BATTERY: "{'level': int, 'charging': bool}",
    CapabilityType.TIMEZONE: "{'timezone': str (e.g., 'America/New_York')}",
    CapabilityType.LOCALE: "{'language': str (e.g., 'fr'), 'country': str (e.g., 'FR')}",
    CapabilityType.PERMISSION: "{'permission': 'camera' | 'location' | 'notifications' | 'photos' | 'contacts', 'grant': bool}",
    CapabilityType.INTERRUPT: "{'type': 'low_memory' | 'incoming_call', 'phone': str}",
    CapabilityType.CLIPBOARD: "{'text': str}",
    CapabilityType.BIOMETRIC: "{'action': 'match' | 'unmatch' | 'enroll', 'finger_id': int}",
    CapabilityType.DEEPLINK: "{'url': str}",
    CapabilityType.APP_STATE: "{'action': 'clear_data' | 'force_stop'}",
    CapabilityType.MEDIA_INJECT: "{'local_path': str}",
    CapabilityType.ACCESSIBILITY: "{'enabled': bool}"
}



# --- Action Definitions ---


class TypeAction(BaseModel):
    type: Literal[ActionType.TYPE] = ActionType.TYPE
    text: str = Field(default="", description="The text of the action.")
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates to move to."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class TapAction(BaseModel):
    type: Literal[ActionType.TAP] = ActionType.TAP
    coordinates: Coordinates = Field(..., description="The screen coordinates to tap.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class ClearAction(BaseModel):
    type: Literal[ActionType.CLEAR] = ActionType.CLEAR
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates of the text field to clear."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class DoubleTapAction(BaseModel):
    type: Literal[ActionType.DOUBLE_TAP] = ActionType.DOUBLE_TAP
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates to double tap."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class LongPressAction(BaseModel):
    type: Literal[ActionType.LONG_PRESS] = ActionType.LONG_PRESS
    duration: float = Field(
        ..., description="The duration of the long press in seconds."
    )
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates to long press."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class DragAndDropAction(BaseModel):
    type: Literal[ActionType.DRAG_AND_DROP] = ActionType.DRAG_AND_DROP
    source_x: int = Field(..., description="The x coordinate to drag from.")
    source_y: int = Field(..., description="The y coordinate to drag from.")
    destination_x: int = Field(..., description="The x coordinate to drag to.")
    destination_y: int = Field(..., description="The y coordinate to drag to.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class KeyEventAction(BaseModel):
    type: Literal[ActionType.KEY_EVENT] = ActionType.KEY_EVENT
    key: int = Field(..., description="The key code to press.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class SwipeAction(BaseModel):
    type: Literal[ActionType.SWIPE] = ActionType.SWIPE
    start_x: int = Field(..., description="The x coordinate to start the swipe.")
    start_y: int = Field(..., description="The y coordinate to start the swipe.")
    end_x: int = Field(..., description="The x coordinate to end the swipe.")
    end_y: int = Field(..., description="The y coordinate to end the swipe.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class CapabilityAction(BaseModel):
    type: Literal[ActionType.CAPABILITY] = ActionType.CAPABILITY
    capability: CapabilityType = Field(..., description="The system capability to control.")
    parameters: dict = Field(
        default_factory=dict,
        description="Key-value parameters for the capability (e.g. {'latitude': 37.77, 'longitude': -122.41} or {'mode': 'offline'} or {'permission': 'camera', 'grant': True})."
    )
    reason: str = Field(..., description="How the capability control helps achieve the test goal.")


class WaitAction(BaseModel):
    """Waits before the next observation so transient rendering can finish."""

    type: Literal[ActionType.WAIT] = ActionType.WAIT
    duration: float = Field(
        default=1.0, description="Seconds to wait before looking at the screen again."
    )
    reason: str = Field(..., description="What the screen is expected to finish doing.")


# --- The Union of All Actions ---
Action = Annotated[
    TypeAction
    | TapAction
    | DoubleTapAction
    | LongPressAction
    | DragAndDropAction
    | KeyEventAction
    | SwipeAction
    | CapabilityAction
    | ClearAction
    | WaitAction,
    Field(discriminator="type")
]


# --- The Final Output Schema ---
class ActionPlan(BaseModel):
    instruction: str = Field(
        ..., description="The normalized instruction to achieve the goal."
    )
    actions: list[Action] = Field(
        ...,
        description="The sequence of machine-executable actions to achieve the goal.",
    )
    external_navigation: bool = Field(
        default=False,
        description="Set to True if the next step is expected to temporarily navigate outside the SUT app (e.g., opening an external browser page, map, system dialog, or payment screen).",
    )
