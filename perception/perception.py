from device.device_session import DeviceSession
from perception.image_descriptor import ImageDescriptor


class ScreenPerception:
    def __init__(self, device_session: DeviceSession):
        self.device_session = device_session
        self.image_descriptor = ImageDescriptor()

    def perceive(self):
        png_bytes = self.device_session.device_control.get_screenshot()
        gui_elements, annotated_img = self.image_descriptor.describe(png_bytes)
        return gui_elements, annotated_img
