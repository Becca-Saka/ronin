import json
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Optional

from ronin.agent.perception.image_descriptor import GUIElement


def _to_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return value


@dataclass
class ScanInfo:
    mode: str
    reason: str
    phash: str
    phash_distance: Optional[int] = None
    screen_width: int = 0
    screen_height: int = 0
    element_count: int = 0
    matched: Optional[int] = None
    new: Optional[int] = None
    removed: Optional[int] = None
    match_rate: Optional[float] = None
    dirty_ocr_count: Optional[int] = None
    uncovered_mae: Optional[float] = None
    thresholds: dict[str, float | int] = field(default_factory=dict)


def element_to_dict(el: GUIElement, index: int, scrubber=None) -> dict[str, Any]:
    """Serializes one element, masking declared secrets in its detected text.

    A scan dump is a file, so a password the app happens to be displaying would
    otherwise be written out verbatim. The planner reads the element itself,
    never this.
    """
    cx = (el.bounding_box.x_min + el.bounding_box.x_max) // 2
    cy = (el.bounding_box.y_min + el.bounding_box.y_max) // 2
    text = el.text or ""
    return {
        "index": index,
        "element": el.element,
        "type": el.type,
        "confidence": round(el.confidence, 4),
        "text": scrubber.scrub(text) if scrubber else text,
        "bounding_box": el.bounding_box.model_dump(),
        "center": {"x": cx, "y": cy},
    }


def build_screen_scan_record(
    *,
    step: int,
    timestamp: str,
    scan_info: ScanInfo,
    gui_elements: list[GUIElement],
    screenshot_paths: dict[str, str],
    prev_phash: Optional[str] = None,
    scrubber=None,
) -> dict[str, Any]:
    return {
        "step": step,
        "timestamp": timestamp,
        "screenshots": screenshot_paths,
        "phash": scan_info.phash,
        "prev_phash": prev_phash,
        "phash_distance": scan_info.phash_distance,
        "screen": {
            "width": scan_info.screen_width,
            "height": scan_info.screen_height,
        },
        "scan": {
            "mode": scan_info.mode,
            "reason": scan_info.reason,
            "element_count": scan_info.element_count,
            "matched": scan_info.matched,
            "new": scan_info.new,
            "removed": scan_info.removed,
            "match_rate": scan_info.match_rate,
            "dirty_ocr_count": scan_info.dirty_ocr_count,
            "uncovered_mae": scan_info.uncovered_mae,
            "thresholds": scan_info.thresholds,
        },
        "gui_elements": [
            element_to_dict(el, idx, scrubber) for idx, el in enumerate(gui_elements)
        ],
    }


def write_screen_scan_dump(
    path: Path,
    record: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(record), f, indent=2)
    return path
