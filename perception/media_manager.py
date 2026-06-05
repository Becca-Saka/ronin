import io
from pathlib import Path

from PIL import Image, ImageDraw

from device.device_control import DeviceControl

CLASS_COLORS = {
    "View": "blue",
    "ImageView": "orange",
    "Text": "green",
    "Line": "purple",
}


class MediaManager:
    def __init__(self):
        pass

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
        print(f"  ↳ saved → {screenshot_dir}/{name}.png")

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
