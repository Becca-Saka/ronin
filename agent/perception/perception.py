import io
import logging

import imagehash
import numpy as np
from PIL import Image

from agent.device.device_session import DeviceSession
from agent.perception.image_descriptor import (
    OCR_RELEVANT_TYPES,
    GUIElement,
    ImageDescriptor,
)
from agent.perception.scan_dump import ScanInfo

logger = logging.getLogger(__name__)

# --- Tunable thresholds (adjust per device/app after calibration) ---
# pHash-256 distance above which the screen is treated as a full navigation event.
NAVIGATION_THRESHOLD = 80
# Per-element mean absolute pixel error (0–255) above which an element is dirty.
DIRTY_THRESHOLD = 4.0
# Minimum IoU to consider two same-class elements the same element across frames.
IOU_MATCH_THRESHOLD = 0.70
# If fewer than this fraction of previous elements match, treat as layout shift.
LAYOUT_SHIFT_THRESHOLD = 0.50
# MAE on non-YOLO screen regions above which an uncovered visual change is flagged.
UNCOVERED_THRESHOLD = 2.0
# Consecutive dirty steps with unchanged OCR output before animation suppression kicks in.
ANIMATION_SUPPRESS_AFTER = 3


class ScreenPerception:
    """Manages screen perception using YOLO, OCR, and per-element pixel diff.

    Replaces the old pHash-only cache gate with a three-tier strategy:
      E — 256-bit pHash as a coarse navigation detector only
      C — per-element bounding-box MAE to detect subtle element-level changes
      D — raw MAE on YOLO-uncovered regions to flag canvas/WebView changes
    """

    def __init__(self, device_session: DeviceSession):
        self.device_session = device_session
        self.image_descriptor = ImageDescriptor()

        self._last_hash: imagehash.ImageHash | None = None
        self._last_screenshot: Image.Image | None = None
        self._last_elements: list[GUIElement] | None = None

        self._consecutive_dirty: dict[str, int] = {}
        self._last_ocr_text: dict[
            str, str
        ] = {}  # suppress repeated OCR on animated elements
        self.last_scan_info: ScanInfo | None = None

    @property
    def current_hash(self) -> "imagehash.ImageHash | None":
        return self._last_hash

    def reset(self):
        """Resets all perception state. Call between scenarios."""
        self._last_hash = None
        self._last_screenshot = None
        self._last_elements = None
        self._consecutive_dirty.clear()
        self._last_ocr_text.clear()
        self.last_scan_info = None
        logger.info("Perception state reset.")

    def perceive(self) -> tuple[list[GUIElement], Image.Image, Image.Image]:
        """Captures the current screen and returns up-to-date GUI elements.

        Returns:
            (gui_elements, raw_img, annotated_img)
            - gui_elements: detected elements with fresh text where changed
            - raw_img: clean screenshot with no overlays — send this to the LLM
            - annotated_img: YOLO bounding boxes drawn on screenshot — save to disk
        """
        png_bytes = self.device_session.device_control.get_screenshot()
        raw_img = Image.open(io.BytesIO(png_bytes))
        current_hash = imagehash.phash(raw_img, hash_size=16)
        phash_hex = str(current_hash)
        w, h = raw_img.size
        distance = (
            current_hash - self._last_hash if self._last_hash is not None else None
        )

        if self._last_screenshot is None:
            logger.info("First perception pass — running full YOLO + OCR.")
            return self._full_scan(
                png_bytes,
                raw_img,
                current_hash,
                reason="first_pass",
                phash_distance=distance,
            )

        logger.debug(
            "pHash distance: %d (threshold: %d)", distance, NAVIGATION_THRESHOLD
        )
        if distance >= NAVIGATION_THRESHOLD:
            logger.info(
                "pHash distance %d ≥ %d — navigation detected, full re-scan.",
                distance,
                NAVIGATION_THRESHOLD,
            )
            return self._full_scan(
                png_bytes,
                raw_img,
                current_hash,
                reason="navigation",
                phash_distance=distance,
            )

        curr_elements, annotated_img = self.image_descriptor.describe(png_bytes)

        matched_pairs, new_elements, removed = self._match_elements(
            self._last_elements, curr_elements
        )
        match_rate = len(matched_pairs) / max(len(self._last_elements), 1)
        logger.debug(
            "Element diff: %d matched, %d new, %d removed (match rate %.0f%%).",
            len(matched_pairs),
            len(new_elements),
            len(removed),
            match_rate * 100,
        )

        if match_rate < LAYOUT_SHIFT_THRESHOLD:
            logger.info(
                "IoU match rate %.0f%% < %.0f%% — layout shift detected, full re-scan.",
                match_rate * 100,
                LAYOUT_SHIFT_THRESHOLD * 100,
            )
            return self._full_scan(
                png_bytes,
                raw_img,
                current_hash,
                reason="layout_shift",
                phash_distance=distance,
                matched=len(matched_pairs),
                new=len(new_elements),
                removed=len(removed),
                match_rate=match_rate,
            )

        dirty_elements = []
        for prev_el, curr_el in matched_pairs:
            curr_el.text = prev_el.text
            mae = self._element_mae(
                self._last_screenshot, raw_img, prev_el.bounding_box
            )
            element_key = self._element_key(prev_el)

            if mae > DIRTY_THRESHOLD:
                if self._is_animation_suppressed(element_key, prev_el.text):
                    logger.debug(
                        "Animation suppressed (MAE=%.2f): %s", mae, element_key
                    )
                else:
                    curr_el.text = ""
                    dirty_elements.append(curr_el)
                    logger.debug("Element dirty (MAE=%.2f): %s", mae, element_key)
            else:
                self._consecutive_dirty.pop(element_key, None)

        # New elements that weren't in the previous step need fresh OCR too
        ocr_targets = dirty_elements + new_elements
        ocr_eligible = [el for el in ocr_targets if el.element in OCR_RELEVANT_TYPES]
        if ocr_targets:
            logger.info(
                "Re-running OCR on %d dirty/new element(s) (%d OCR-eligible).",
                len(ocr_targets),
                len(ocr_eligible),
            )
            self.image_descriptor.run_ocr_on_elements(raw_img, ocr_targets)
            # Update animation suppression state for dirty elements
            for el in dirty_elements:
                key = self._element_key(el)
                self._update_animation_tracking(key, el.text)

        uncovered_mae = self._uncovered_region_mae(
            self._last_screenshot, raw_img, curr_elements
        )
        if uncovered_mae > UNCOVERED_THRESHOLD:
            logger.warning(
                "Visual change (MAE=%.2f) in region not covered by YOLO elements. "
                "Check the screenshot for canvas/WebView changes.",
                uncovered_mae,
            )

        if not ocr_targets:
            logger.info("Screen stable — no dirty/new elements, returning cached OCR.")

        self._update_state(raw_img, curr_elements, current_hash)
        self.last_scan_info = ScanInfo(
            mode="incremental",
            reason="ocr_refresh" if ocr_targets else "stable",
            phash=phash_hex,
            phash_distance=distance,
            screen_width=w,
            screen_height=h,
            element_count=len(curr_elements),
            matched=len(matched_pairs),
            new=len(new_elements),
            removed=len(removed),
            match_rate=round(match_rate, 4),
            dirty_ocr_count=len(dirty_elements),
            uncovered_mae=round(uncovered_mae, 4),
            thresholds=self._scan_thresholds(),
        )
        return curr_elements, raw_img, annotated_img

    def _scan_thresholds(self) -> dict[str, float | int]:
        return {
            "navigation_phash": NAVIGATION_THRESHOLD,
            "dirty_mae": DIRTY_THRESHOLD,
            "iou_match": IOU_MATCH_THRESHOLD,
            "layout_shift_match_rate": LAYOUT_SHIFT_THRESHOLD,
            "uncovered_mae": UNCOVERED_THRESHOLD,
        }

    def _full_scan(
        self,
        png_bytes: bytes,
        raw_img: Image.Image,
        current_hash: imagehash.ImageHash,
        reason: str,
        phash_distance: int | None = None,
        matched: int | None = None,
        new: int | None = None,
        removed: int | None = None,
        match_rate: float | None = None,
    ) -> tuple[list[GUIElement], Image.Image, Image.Image]:
        """Runs the complete YOLO + OCR pipeline and updates state."""
        gui_elements, annotated_img = self.image_descriptor.describe(png_bytes)
        logger.info("Full scan complete: %d elements detected.", len(gui_elements))
        self._consecutive_dirty.clear()
        self._update_state(raw_img, gui_elements, current_hash)
        w, h = raw_img.size
        self.last_scan_info = ScanInfo(
            mode="full",
            reason=reason,
            phash=str(current_hash),
            phash_distance=phash_distance,
            screen_width=w,
            screen_height=h,
            element_count=len(gui_elements),
            matched=matched,
            new=new,
            removed=removed,
            match_rate=round(match_rate, 4) if match_rate is not None else None,
            thresholds=self._scan_thresholds(),
        )
        return gui_elements, raw_img, annotated_img

    def _update_state(
        self,
        raw_img: Image.Image,
        elements: list[GUIElement],
        current_hash: imagehash.ImageHash,
    ) -> None:
        self._last_screenshot = raw_img
        self._last_elements = elements
        self._last_hash = current_hash

    def _match_elements(
        self,
        prev: list[GUIElement],
        curr: list[GUIElement],
    ) -> tuple[list[tuple[GUIElement, GUIElement]], list[GUIElement], list[GUIElement]]:
        """Greedy IoU-based matching of elements across frames by class + position.

        Returns:
            matched_pairs: (prev_el, curr_el) for elements present in both frames
            new_elements:  curr elements with no match in prev (appeared)
            removed:       prev elements with no match in curr (disappeared)
        """
        matched_pairs = []
        unmatched_curr = list(curr)
        unmatched_prev = list(prev)

        for prev_el in prev:
            best_iou, best_match = 0.0, None
            for curr_el in unmatched_curr:
                if prev_el.element != curr_el.element:
                    continue
                iou = self._compute_iou(prev_el.bounding_box, curr_el.bounding_box)
                if iou > best_iou:
                    best_iou, best_match = iou, curr_el

            if best_iou >= IOU_MATCH_THRESHOLD and best_match is not None:
                matched_pairs.append((prev_el, best_match))
                unmatched_curr.remove(best_match)
                unmatched_prev.remove(prev_el)

        return matched_pairs, unmatched_curr, unmatched_prev

    def _compute_iou(self, a, b) -> float:
        x_left = max(a.x_min, b.x_min)
        y_top = max(a.y_min, b.y_min)
        x_right = min(a.x_max, b.x_max)
        y_bottom = min(a.y_max, b.y_max)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
        area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _element_mae(self, prev_img: Image.Image, curr_img: Image.Image, bbox) -> float:
        """Mean absolute pixel error within an element's bounding box crop."""
        box = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        prev_arr = np.array(prev_img.crop(box)).astype(float)
        curr_arr = np.array(curr_img.crop(box)).astype(float)
        if prev_arr.size == 0:
            return 0.0
        return float(np.mean(np.abs(prev_arr - curr_arr)))

    def _uncovered_region_mae(
        self, prev_img: Image.Image, curr_img: Image.Image, elements: list[GUIElement]
    ) -> float:
        """MAE across screen pixels not covered by any YOLO bounding box (Option D)."""
        prev_arr = np.array(prev_img).astype(float)
        curr_arr = np.array(curr_img).astype(float)
        h, w = prev_arr.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        for el in elements:
            bb = el.bounding_box
            mask[bb.y_min : bb.y_max, bb.x_min : bb.x_max] = False
        if not mask.any():
            return 0.0
        diff = np.abs(prev_arr - curr_arr)
        return float(np.mean(diff[mask]))

    def _element_key(self, el: GUIElement) -> str:
        return f"{el.element}_{el.bounding_box.x_min}_{el.bounding_box.y_min}"

    def _is_animation_suppressed(self, element_key: str, cached_text: str) -> bool:
        """Returns True if this element has been consistently dirty with the same
        OCR output for ANIMATION_SUPPRESS_AFTER steps — indicating an animation."""
        count = self._consecutive_dirty.get(element_key, 0)
        last_text = self._last_ocr_text.get(element_key)
        if count >= ANIMATION_SUPPRESS_AFTER and last_text == cached_text:
            logger.debug("Animation suppression active for %s", element_key)
            return True
        return False

    def _update_animation_tracking(self, element_key: str, new_text: str) -> None:
        last_text = self._last_ocr_text.get(element_key)
        if last_text == new_text:
            self._consecutive_dirty[element_key] = (
                self._consecutive_dirty.get(element_key, 0) + 1
            )
        else:
            self._consecutive_dirty[element_key] = 1
            self._last_ocr_text[element_key] = new_text
