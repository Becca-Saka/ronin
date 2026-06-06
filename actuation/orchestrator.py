import logging
import time

from actuation.actions import ActionPlan, ActionType
from device.device_control import DeviceControl
from device.capability_controller import CapabilityController

logger = logging.getLogger(__name__)


class ActionOrchestrator:
    def __init__(self, device_control: DeviceControl):
        self.device_control = device_control
        self.capability_controller = CapabilityController(device_control)

    def orchestrate(self, action_plan: ActionPlan):
        for action in action_plan.actions:
            match action.type:
                case ActionType.TYPE:
                    coordinates = action.coordinates
                    logger.info(
                        "Executing action: %s at (%d, %d) — %s",
                        action.type, coordinates.x, coordinates.y, action.reason,
                    )
                    self.device_control.type(action.text, coordinates.x, coordinates.y)
                case ActionType.CLEAR:
                    coordinates = action.coordinates
                    logger.info(
                        "Executing action: %s at (%d, %d) — %s",
                        action.type, coordinates.x, coordinates.y, action.reason,
                    )
                    self.device_control.clear(coordinates.x, coordinates.y)
                case ActionType.TAP:
                    coordinates = action.coordinates
                    logger.info(
                        "Executing action: %s at (%d, %d) — %s",
                        action.type, coordinates.x, coordinates.y, action.reason,
                    )
                    self.device_control.tap(coordinates.x, coordinates.y, with_feedback=True)
                case ActionType.DOUBLE_TAP:
                    coordinates = action.coordinates
                    logger.info(
                        "Executing action: %s at (%d, %d) — %s",
                        action.type, coordinates.x, coordinates.y, action.reason,
                    )
                    self.device_control.double_tap(coordinates.x, coordinates.y, with_feedback=True)
                case ActionType.LONG_PRESS:
                    coordinates = action.coordinates
                    logger.info(
                        "Executing action: %s at (%d, %d) duration=%s — %s",
                        action.type, coordinates.x, coordinates.y, action.duration, action.reason,
                    )
                    self.device_control.long_press(
                        coordinates.x, coordinates.y, action.duration
                    )
                case ActionType.DRAG_AND_DROP:
                    logger.info(
                        "Executing action: %s from (%d, %d) to (%d, %d) — %s",
                        action.type,
                        action.source_x, action.source_y,
                        action.destination_x, action.destination_y,
                        action.reason,
                    )
                    self.device_control.drag_and_drop(
                        action.source_x,
                        action.source_y,
                        action.destination_x,
                        action.destination_y,
                    )
                case ActionType.KEY_EVENT:
                    logger.info(
                        "Executing action: %s key=%s — %s",
                        action.type, action.key, action.reason,
                    )
                    self.device_control.key_event(action.key)
                case ActionType.SWIPE:
                    logger.info(
                        "Executing action: %s from (%d, %d) to (%d, %d) — %s",
                        action.type,
                        action.start_x, action.start_y,
                        action.end_x, action.end_y,
                        action.reason,
                    )
                    self.device_control.swipe(
                        action.start_x, action.start_y, action.end_x, action.end_y
                    )
                case ActionType.CAPABILITY:
                    logger.info(
                        "Executing action: %s capability=%s — %s",
                        action.type, action.capability, action.reason,
                    )
                    self.capability_controller.execute(action.capability, action.parameters)
            time.sleep(1.5)
