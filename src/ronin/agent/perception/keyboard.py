"""Telling the soft keyboard's own keys from the app's controls.

YOLO has no class for a keyboard, so this is geometry: low on the screen with
a label too short to be a sentence.
"""

import logging
from copy import copy
from typing import List, Optional

from ronin.agent.perception.elements import GUIElement

logger = logging.getLogger(__name__)

KEYBOARD_TOP_FRACTION = 0.60
"""Where the keyboard starts, as a fraction of screen height."""

KEY_LABEL_LENGTH = 2
"""A label this short on a key-shaped box is a key, not a control."""


def split_keyboard(
    elements: List[GUIElement], keyboard_open: Optional[bool], screen_height: Optional[int]
) -> tuple[List[GUIElement], List[GUIElement]]:
    """Separates the soft keyboard's own keys from the app's controls.

    An open keyboard is most of what is on screen. Numbering every letter would
    bury the handful of app controls under a wall of badges, and the agent never
    needs them. Text arrives through `type`, and Enter and Back are key events.

    YOLO has no class for a keyboard, so this is geometry: low on the screen
    with a label too short to be a sentence.
    """
    if not keyboard_open or not screen_height:
        return list(elements or []), []

    keyboard_top = int(screen_height * KEYBOARD_TOP_FRACTION)
    kept, keys = [], []
    for element in elements or []:
        box = element.bounding_box
        center_y = (box.y_min + box.y_max) // 2
        looks_like_key = (
            center_y >= keyboard_top
            and len((element.text or "").strip()) <= KEY_LABEL_LENGTH
        )
        (keys if looks_like_key else kept).append(element)

    # A second pass by position, because the label rule misses the wide keys —
    # "?123", "space", "return". Anything sitting within the keyboard's own
    # extent belongs to it whatever its label says.
    if keys:
        top = min(key.bounding_box.y_min for key in keys)
        bottom = max(key.bounding_box.y_max for key in keys)
        remaining = []
        for element in kept:
            center_y = (element.bounding_box.y_min + element.bounding_box.y_max) // 2
            (keys if top <= center_y <= bottom else remaining).append(element)
        kept = remaining
        logger.info("Keyboard is open — %d key(s) left unnumbered.", len(keys))
    return kept, keys


def without_keyboard_keys(
    elements: List[GUIElement], keyboard_open: Optional[bool], screen_height: Optional[int]
) -> List[GUIElement]:
    """The app's controls, with the soft keyboard's keys removed."""
    return split_keyboard(elements, keyboard_open, screen_height)[0]


def keyboard_band(keys: List[GUIElement], screen_width: int) -> Optional[tuple]:
    """The region the keyboard occupies, as one rectangle.

    Drawn so that the bare strip carrying no numbers is explained. Nothing here
    carries a number because the keyboard covers it. Anything behind it needs
    the keyboard dismissed first.
    """
    if not keys:
        return None
    boxes = [key.bounding_box for key in keys]
    return (
        0,
        min(box.y_min for box in boxes),
        screen_width,
        max(box.y_max for box in boxes),
    )


def annotate_keyboard_keys(
    elements: List[GUIElement], keyboard_open: Optional[bool], screen_height: Optional[int]
) -> List[GUIElement]:
    """Prefixes likely keyboard keys with [key] so the planner can tell them apart.

    Used where the keys are still offered as targets. Where they are numbered
    instead, see without_keyboard_keys — there they are not offered at all.
    """
    if not keyboard_open or not screen_height:
        return elements

    keyboard_top = int(screen_height * KEYBOARD_TOP_FRACTION)
    annotated = []
    for element in elements:
        center_y = (element.bounding_box.y_min + element.bounding_box.y_max) // 2
        text = (element.text or "").strip()
        if center_y >= keyboard_top and len(text) <= KEY_LABEL_LENGTH:
            element = copy(element)
            element.text = f"[key] {text}" if text else "[key]"
        annotated.append(element)
    return annotated
