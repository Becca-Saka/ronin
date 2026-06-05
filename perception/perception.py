"""
Perception module coordinating screenshot acquisition and UI element parsing.
"""

from device.device_session import DeviceSession
from perception.image_descriptor import ImageDescriptor


class ScreenPerception:
    """Manages screen perception by combining screenshots and vision models."""

    def __init__(self, device_session: DeviceSession):
        """Initializes ScreenPerception with an active device session.

        Args:
            device_session: The active DeviceSession instance.
        """
        self.device_session = device_session
        self.image_descriptor = ImageDescriptor()

    def perceive(self):
        """Captures the current screen and runs element detection and OCR on it.

        Returns:
            A tuple of (gui_elements, annotated_image) where gui_elements is a list
            of GUIElement detections, and annotated_image is a PIL Image with boxes drawn.
        """
        png_bytes = self.device_session.device_control.get_screenshot()
        gui_elements, annotated_img = self.image_descriptor.describe(png_bytes)
        return gui_elements, annotated_img

