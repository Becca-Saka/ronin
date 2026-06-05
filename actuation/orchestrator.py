import logging
import time

from actuation.actions import ActionPlan, ActionType
from device.device_control import DeviceControl

logger = logging.getLogger(__name__)


class ActionOrchestrator:
    def __init__(self, device_control: DeviceControl):
        self.device_control = device_control

    def orchestrate(self, action_plan: ActionPlan):
        for action in action_plan.actions:
            logger.info("Executing action: %s because %s", action.type, action.reason)
            match action.type:
                case ActionType.TYPE:
                    coordinates = action.coordinates
                    self.device_control.type(action.text, coordinates.x, coordinates.y)
                case ActionType.TAP:
                    coordinates = action.coordinates
                    self.device_control.tap(coordinates.x, coordinates.y)
                case ActionType.DOUBLE_TAP:
                    coordinates = action.coordinates
                    self.device_control.double_tap(coordinates.x, coordinates.y)
                case ActionType.LONG_PRESS:
                    coordinates = action.coordinates
                    self.device_control.long_press(
                        coordinates.x, coordinates.y, action.duration
                    )
                case ActionType.DRAG_AND_DROP:
                    self.device_control.drag_and_drop(
                        action.source_x,
                        action.source_y,
                        action.destination_x,
                        action.destination_y,
                    )
                case ActionType.KEY_EVENT:
                    self.device_control.key_event(action.key)
                case ActionType.SWIPE:
                    self.device_control.swipe(
                        action.start_x, action.start_y, action.end_x, action.end_y
                    )
            time.sleep(1.5)
