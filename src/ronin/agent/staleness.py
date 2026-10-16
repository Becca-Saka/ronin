"""Checking that the screen still looks like the one the plan was made from.

Several seconds pass between reading the screen and acting on it — a model call
is not instant. In that gap a keyboard opens, a list populates, a banner lands,
and the coordinates the planner chose now point at something else. The tap is
accurate and lands on the wrong thing.

So we look once more immediately before acting, at the target only. Comparing
the whole screen would trip on every animation elsewhere and throw away good
plans; what matters is whether the thing we are about to touch has moved.
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from ronin.defaults import STALE_TARGET_MAE, STALE_TARGET_PATCH

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def _point(action) -> Optional[Tuple[int, int]]:
    """The single point an action is aimed at, or None for gestures and others."""
    coordinates = getattr(action, "coordinates", None)
    if coordinates is None:
        return None
    return coordinates.x, coordinates.y


def has_target(action) -> bool:
    """Whether the action is aimed at one point, and so has something to go stale.

    Gestures, key events and waits are aimed at no single place: there is
    nothing about them for a changing screen to invalidate.
    """
    return _point(action) is not None


def target_region(action, gui_elements, screen: Tuple[int, int]) -> Optional[Box]:
    """The area to re-check: the element being targeted, or a patch around the point.

    A point inside no detected element still gets checked — that is exactly the
    invented-coordinate case, and a patch is the only thing left to compare.
    """
    point = _point(action)
    if point is None:
        return None
    x, y = point

    for element in gui_elements or []:
        box = element.bounding_box
        if box.x_min <= x <= box.x_max and box.y_min <= y <= box.y_max:
            return box.x_min, box.y_min, box.x_max, box.y_max

    half = STALE_TARGET_PATCH // 2
    width, height = screen
    return (
        max(0, x - half),
        max(0, y - half),
        min(width, x + half),
        min(height, y + half),
    )


def _mean_absolute_error(before: Image.Image, after: Image.Image, box: Box) -> float:
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return 0.0
    first = np.asarray(before.crop(box), dtype=float)
    second = np.asarray(after.crop(box), dtype=float)
    if first.shape != second.shape:
        return float("inf")
    return float(np.abs(first - second).mean())


def target_moved(before_png: bytes, after_png: bytes, action, gui_elements) -> bool:
    """Whether the action's target changed between planning and now.

    Returns False whenever the question cannot be answered — an unreadable
    screenshot or an action with no single target is not evidence of movement,
    and refusing to act on no evidence would strand the run.
    """
    try:
        before = Image.open(io.BytesIO(before_png)).convert("RGB")
        after = Image.open(io.BytesIO(after_png)).convert("RGB")
    except Exception:
        return False

    if before.size != after.size:
        return False

    box = target_region(action, gui_elements, before.size)
    if box is None:
        return False

    error = _mean_absolute_error(before, after, box)
    if error > STALE_TARGET_MAE:
        logger.info(
            "Target changed since planning (MAE=%.1f > %.1f) — re-reading the screen "
            "instead of acting on it.",
            error,
            STALE_TARGET_MAE,
        )
        return True
    return False
