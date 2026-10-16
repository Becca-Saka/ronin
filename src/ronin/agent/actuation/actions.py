from enum import Enum
from typing import Any, Literal, Annotated

from pydantic import BaseModel, Field, create_model, model_validator


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
    TARGET_NOT_LISTED = "target_not_listed"


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


class TargetNotListedAction(BaseModel):
    """Says the target is not among the numbered elements, rather than guessing.

    For when the element a step needs is absent from the numbered targets.
    Never executed: the step stops and the screen is recorded, so a detection
    gap becomes a counted event instead of a tap into empty space.
    """

    type: Literal[ActionType.TARGET_NOT_LISTED] = ActionType.TARGET_NOT_LISTED
    description: str = Field(
        ..., description="What the target looks like and roughly where it is."
    )
    reason: str = Field(..., description="What you were trying to achieve.")


class WaitAction(BaseModel):
    """Waits before the next observation so transient rendering can finish."""

    type: Literal[ActionType.WAIT] = ActionType.WAIT
    duration: float = Field(
        default=1.0, description="Seconds to wait before looking at the screen again."
    )
    reason: str = Field(..., description="What the screen is expected to finish doing.")


def _marks_variant(model, name: str):
    """The same action with coordinates removed and `mark` required.

    Marks mode is enforced by the schema rather than by prose plus a check
    afterwards. An optional field is one the model may omit — and it does,
    intermittently, which turns every omission into a wasted step. A field it
    cannot omit and a field that does not exist leave nothing to get wrong.
    """
    fields = {
        field_name: (info.annotation, info)
        for field_name, info in model.model_fields.items()
        if field_name != "coordinates"
    }
    fields["mark"] = (
        int,
        Field(..., description="Number of the target element on the screenshot."),
    )
    variant = create_model(name, **fields)
    # Resolution turns the number back into this, once we know where it points.
    variant.coordinate_form = model
    return variant


TypeMarkAction = _marks_variant(TypeAction, "TypeMarkAction")
TapMarkAction = _marks_variant(TapAction, "TapMarkAction")
ClearMarkAction = _marks_variant(ClearAction, "ClearMarkAction")
DoubleTapMarkAction = _marks_variant(DoubleTapAction, "DoubleTapMarkAction")
LongPressMarkAction = _marks_variant(LongPressAction, "LongPressMarkAction")


# Actions that touch a specific place, and so need a target resolved first.
TARGETED_ACTIONS = frozenset(
    {
        ActionType.TAP,
        ActionType.TYPE,
        ActionType.CLEAR,
        ActionType.DOUBLE_TAP,
        ActionType.LONG_PRESS,
    }
)


# Actions whose target must be a text field, and which therefore prove
# whether it is one.
TEXT_FIELD_ACTIONS = frozenset({ActionType.TYPE, ActionType.CLEAR})


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
    | WaitAction
    | TargetNotListedAction,
    Field(discriminator="type")
]


# The same vocabulary, but targets are named by number only.
MarksAction = Annotated[
    TypeMarkAction
    | TapMarkAction
    | DoubleTapMarkAction
    | LongPressMarkAction
    | ClearMarkAction
    | DragAndDropAction
    | KeyEventAction
    | SwipeAction
    | CapabilityAction
    | WaitAction
    | TargetNotListedAction,
    Field(discriminator="type"),
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
