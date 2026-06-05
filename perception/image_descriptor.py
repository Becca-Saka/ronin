import io

import easyocr
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from ultralytics import YOLO


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __iter__(self):
        yield self.x_min
        yield self.y_min
        yield self.x_max
        yield self.y_max


class TextElement(BaseModel):
    element: str
    bounding_box: BoundingBox  # [x_min, y_min, x_max, y_max]
    text: str
    type: str


class GUIElement(BaseModel):
    element: str
    bounding_box: BoundingBox  # [x_min, y_min, x_max, y_max]
    confidence: float
    type: str
    text: str = ""

    def model_dump(self):
        return {
            "element": self.element,
            "bounding_box": self.bounding_box,
            "confidence": self.confidence,
            "type": self.type,
            "text": self.text,
        }


class ImageDescriptor:
    def __init__(self):
        # --- load model ---
        print("Loading deki-yolo model...")
        model_path = hf_hub_download(repo_id="orasul/deki-yolo", filename="best.pt")
        self.yolo = YOLO(model_path)
        print("✓ Model loaded")
        self.names = self.yolo.names
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        print("✓ OCR loaded")

    def describe(
        self,
        png_bytes: bytes,
        box_color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> tuple[list[GUIElement], Image.Image]:
        """
        Describes the image, annotates it, saves to disk, and returns GUI elements.
        """
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        results = self.yolo(image)

        result = results[0]
        names = result.names if hasattr(result, "names") else self.names
        gui_elements = self.get_yolo_gui_elements(result, names)
        OCR_RELEVANT_TYPES = {"Text", "Button", "EditText"}  # only OCR relevant types
        for el in gui_elements:
            if el.element in OCR_RELEVANT_TYPES:
                el_text = self.extract_text_from_element(image, el)
                el.text = el_text  # attach to element
        annotated_img = self.draw_gui_elements(
            image, gui_elements, box_color, thickness
        )

        return gui_elements, annotated_img

    def extract_text_from_element(self, image: Image.Image, element: GUIElement) -> str:
        """Crop the bounding box and run OCR on it."""
        x1, y1, x2, y2 = element.bounding_box

        # Add small padding
        pad = 4
        w, h = image.size
        crop = image.crop(
            (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(w, x2 + pad),
                min(h, y2 + pad),
            )
        )

        results = self.reader.readtext(np.array(crop), detail=0)
        return " ".join(results).strip()

    def get_yolo_gui_elements(self, result, names: dict) -> list[GUIElement]:
        """
        Parses YOLO result object and returns a list of Pydantic GUIElements.
        """
        elements = []

        if result.boxes is None:
            return elements

        # Move data to CPU and numpy
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(xyxy, clss, confs):
            x_min, y_min, x_max, y_max = box

            # Get class name
            class_name = (
                names.get(cls_id, str(cls_id))
                if isinstance(names, dict)
                else names[cls_id]
            )
            element = GUIElement(
                element=class_name,
                bounding_box=BoundingBox(
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_max),
                    y_max=int(y_max),
                ),
                confidence=float(conf),
                type="yolo_detection",
            )

            elements.append(element)

        return elements

    def draw_gui_elements(
        self,
        image: Image.Image,
        elements: list[GUIElement],
        box_color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> Image.Image:
        """
        Draws boxes from a list of GUIElement objects using PIL ImageDraw only.
        """
        image = image.copy()
        draw = ImageDraw.Draw(image)

        w, h = image.size
        scale_ref = max(w, h)
        t = max(1, int(thickness * (scale_ref / 1000.0)))
        font_size = max(10, int(14 * (scale_ref / 1200.0)))

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for item in elements:
            x1, y1, x2, y2 = item.bounding_box
            label_text = f"{item.element} {item.confidence:.2f}"

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=t)

            # Measure text
            bbox = draw.textbbox((0, 0), label_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Top-right positioning
            text_x = max(0, x2 - tw - 3)
            text_y = max(3, y1 - th - 5)

            # Background box
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + tw + 2, text_y + th + 2],
                fill=(0, 0, 0),
            )

            # Draw text
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)

        return image
