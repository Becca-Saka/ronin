import io
import logging
import imagehash
from PIL import Image

from device.device_session import DeviceSession
from perception.image_descriptor import ImageDescriptor

logger = logging.getLogger(__name__)


class ScreenPerception:
    """Manages screen perception by combining screenshots and vision models."""

    def __init__(self, device_session: DeviceSession):
        """Initializes ScreenPerception with an active device session.

        Args:
            device_session: The active DeviceSession instance.
        """
        self.device_session = device_session
        self.image_descriptor = ImageDescriptor()
        self._cache = {}  # Map: imagehash.ImageHash -> list of GUIElement
        self._last_hash = None

    def clear_cache(self):
        """Clears the perception cache and resets the last seen hash."""
        self._cache.clear()
        self._last_hash = None
        logger.info("Perception cache cleared.")

    def perceive(self):
        """Captures the current screen and runs element detection and OCR on it.

        Returns:
            A tuple of (gui_elements, annotated_image) where gui_elements is a list
            of GUIElement detections, and annotated_image is a PIL Image with boxes drawn.
        """
        png_bytes = self.device_session.device_control.get_screenshot()
        img = Image.open(io.BytesIO(png_bytes))
        current_hash = imagehash.phash(img)

        # Check if screen is visually identical to last step (Hamming distance < 8)
        if self._last_hash and (current_hash - self._last_hash) < 8:
            logger.info("pHash cache hit! Bypassing YOLOv8 and OCR.")
            return self._cache[self._last_hash], img

        gui_elements, annotated_img = self.image_descriptor.describe(png_bytes)
        self._cache[current_hash] = gui_elements
        self._last_hash = current_hash
        return gui_elements, annotated_img

