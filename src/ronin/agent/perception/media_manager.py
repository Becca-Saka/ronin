import io
import logging
from pathlib import Path

from PIL import Image, ImageDraw

from ronin.agent.device.device_control import DeviceControl

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "View": "blue",
    "ImageView": "orange",
    "Text": "green",
    "Line": "purple",
}


class MediaManager:
    def __init__(self, scrubber=None):
        """`scrubber` holds the journey's declared secrets; without one, nothing is masked."""
        self.scrubber = scrubber

    def redact_artifact_image(self, image: Image.Image, elements: list) -> Image.Image:
        """Covers declared secret values before a screenshot is persisted.

        Detection text is never masked on the way in — the planner needs it,
        and a security audit cannot find an exposed card number it was not
        allowed to see — so the mask is applied here, on the way out.
        """
        if self.scrubber is None:
            return image
        safe = image.copy()
        draw = ImageDraw.Draw(safe)
        for element in elements:
            if not self.scrubber.contains_secret(element.text):
                continue
            box = element.bounding_box
            draw.rectangle(
                (box.x_min, box.y_min, box.x_max, box.y_max), fill="black"
            )
        return safe

    def save_redacted_image(self, image: Image.Image, elements: list, path: Path) -> None:
        self.redact_artifact_image(image, elements).save(path)

    def save_redacted_bytes(self, data: bytes, elements: list, path: Path) -> None:
        image = Image.open(io.BytesIO(data))
        self.save_redacted_image(image, elements, path)

    def save_redacted_pair(
        self,
        annotated: Image.Image,
        raw: bytes,
        elements: list,
        annotated_path: Path,
        raw_path: Path,
    ) -> None:
        self.save_redacted_image(annotated, elements, annotated_path)
        self.save_redacted_bytes(raw, elements, raw_path)

    def save_redacted_action_annotation(
        self, annotated: Image.Image, elements: list, action, path: Path
    ) -> bool:
        return self.save_action_annotation(
            self.redact_artifact_image(annotated, elements), action, path
        )

    def annotate(self, png_bytes, x, y, label, color="red"):
        image = Image.open(io.BytesIO(png_bytes))
        draw = ImageDraw.Draw(image)
        px, py = int(x), int(y)
        radius = 40
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius], outline=color, width=6
        )
        draw.line([px - radius, py, px + radius, py], fill=color, width=3)
        draw.line([px, py - radius, px, py + radius], fill=color, width=3)
        draw.text((px + radius + 5, py - 10), f"{label} ({x:.0f}, {y:.0f})", fill=color)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def save_screenshot(
        self, png_bytes, name: str, screenshot_dir: Path = Path("screenshots")
    ):
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        filepath = screenshot_dir / f"{name}.png"
        Image.open(io.BytesIO(png_bytes)).save(filepath)
        logger.info("Saved screenshot to %s/%s.png", screenshot_dir, name)

    def capture_screenshot(
        self,
        device_control: DeviceControl,
        name: str,
        screenshot_dir: Path = Path("screenshots"),
    ):
        png_bytes = device_control.get_screenshot()
        self.save_screenshot(png_bytes, name, screenshot_dir)

    def save_annotated_screenshot(
        self,
        x,
        y,
        device_control: DeviceControl,
        name: str,
        screenshot_dir: Path = Path("screenshots"),
        label: str = "tap",
        color: str = "red",
    ):
        png_bytes = device_control.get_screenshot()
        annotated_png_bytes = self.annotate(png_bytes, x, y, label, color)
        self.save_screenshot(annotated_png_bytes, name, screenshot_dir)
        return annotated_png_bytes

    def save_action_annotation(
        self,
        annotated_img: Image.Image,
        action_plan,
        filepath: Path,
    ) -> bool:
        """Annotates the planned action (tap, type, swipe, drag) on the image and saves it.

        Returns:
            True if the image was annotated and saved, False otherwise.
        """
        if not action_plan:
            return False

        action_img = annotated_img.copy()
        draw = ImageDraw.Draw(action_img)
        first_action = action_plan
        action_type_str = first_action.type.value
        highlighted = False

        if hasattr(first_action, "coordinates") and first_action.coordinates:
            px, py = int(first_action.coordinates.x), int(first_action.coordinates.y)
            radius = 25
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                outline="red",
                width=5,
            )
            draw.line([px - radius, py, px + radius, py], fill="red", width=3)
            draw.line([px, py - radius, px, py + radius], fill="red", width=3)
            draw.text(
                (px + radius + 5, py - 10),
                f"{action_type_str.upper()} ({px}, {py})",
                fill="red",
            )
            highlighted = True
        elif action_type_str == "swipe" and hasattr(first_action, "start_x"):
            sx, sy = int(first_action.start_x), int(first_action.start_y)
            ex, ey = int(first_action.end_x), int(first_action.end_y)
            draw.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill="red")
            draw.line([sx, sy, ex, ey], fill="red", width=5)
            draw.text((sx + 15, sy - 10), f"SWIPE START ({sx}, {sy})", fill="red")
            highlighted = True
        elif action_type_str == "drag_and_drop" and hasattr(first_action, "source_x"):
            sx, sy = int(first_action.source_x), int(first_action.source_y)
            ex, ey = int(first_action.destination_x), int(first_action.destination_y)
            draw.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill="red")
            draw.line([sx, sy, ex, ey], fill="red", width=5)
            draw.text((sx + 15, sy - 10), f"DRAG START ({sx}, {sy})", fill="red")
            highlighted = True

        if highlighted:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            action_img.save(filepath)
            logger.info("Saved action location annotation to %s", filepath)
            return True

        return False
