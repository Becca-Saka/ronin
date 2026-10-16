"""Whether the text about to be typed will land in the field that was tapped.

Text goes to whatever holds focus. A tap can miss, because the keyboard
reflowed the layout or the coordinates are a step old. It would then write into
the previously focused field, with nothing to show that it had. So these
functions read focus back and check it against the point asked for.

The caller passes the screen area in. A focused element can cover the whole
screen (Flutter, WebView). It then contains every point, and cannot answer the
question.
"""

import logging
import time

from ronin.defaults import FOCUS_POLL_INTERVAL, FOCUS_POLL_TIMEOUT
from ronin.errors import ActionRejected

logger = logging.getLogger(__name__)

FULL_SCREEN_SHARE = 0.8


def focused_element(driver):
    """The element holding focus, or None when the driver cannot say.

    Broad at this boundary. An unreachable session and a page with no focus
    mean the same thing here: nothing can be verified.
    """
    try:
        return driver.switch_to.active_element
    except Exception as exc:  # noqa: BLE001 - driver boundary
        logger.debug("Could not read the focused element: %s", exc)
        return None


def contains(driver, screen_area, x, y) -> bool:
    """Whether the active element contains the requested coordinates.

    A focused element covering the screen (Flutter, WebView) answers for
    every point, so it cannot stand in for a real field.
    """
    element = focused_element(driver)
    if element is None:
        return False
    try:
        rect = element.rect
    except Exception as exc:  # noqa: BLE001 - driver boundary
        logger.debug("Could not read focused element bounds: %s", exc)
        return False

    width, height = rect.get("width", 0), rect.get("height", 0)
    if screen_area and width * height >= FULL_SCREEN_SHARE * screen_area:
        return False

    return (
        rect.get("x", 0) <= x <= rect.get("x", 0) + width
        and rect.get("y", 0) <= y <= rect.get("y", 0) + height
    )


def await_focus(driver, screen_area, x, y) -> None:
    """Polls until focus lands on (x, y), instead of sleeping a flat second."""
    for _ in range(max(1, int(FOCUS_POLL_TIMEOUT / FOCUS_POLL_INTERVAL))):
        if contains(driver, screen_area, x, y):
            return
        time.sleep(FOCUS_POLL_INTERVAL)


def assert_at(driver, screen_area, x, y, action: str) -> None:
    """Raises unless focus is on the element at (x, y).

    Text goes to whatever holds focus. A tap can miss, because the keyboard
    shifted the layout or the coordinates are stale. It would then write into
    the previously focused field without any sign.

    A change of focus is not evidence that the right field was hit. Any text
    field opens the keyboard, so a tap on the wrong one looks identical to the
    layout reflowing. The planner reads the screen again and tries again. That
    costs a step and cannot type into the wrong field.
    """
    element = focused_element(driver)
    if element is None:
        return
    try:
        rect = element.rect
    except Exception as exc:  # noqa: BLE001 - driver boundary
        logger.debug("Could not read focused element bounds: %s", exc)
        return

    width, height = rect.get("width", 0), rect.get("height", 0)
    logger.debug("Focused element after tap: %s at %s", element.id, rect)

    if screen_area and width * height >= FULL_SCREEN_SHARE * screen_area:
        # A single full-screen surface (Flutter, WebView) tells us nothing.
        logger.debug("Focused element covers the screen; skipping the focus check.")
        return

    inside = (
        rect.get("x", 0) <= x <= rect.get("x", 0) + width
        and rect.get("y", 0) <= y <= rect.get("y", 0) + height
    )
    if not inside:
        raise ActionRejected(
            f"{action} at ({x}, {y}) did not focus anything there — focus is on the "
            f"element at x={rect.get('x')},y={rect.get('y')} "
            f"({width}x{height}). Nothing was entered. The layout has probably moved "
            "since these coordinates were read; re-check the field's current position."
        )
