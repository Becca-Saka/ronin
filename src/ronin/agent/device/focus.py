"""Whether the text about to be typed will land in the field that was tapped.

Text goes to whatever holds focus. A tap can miss, because the keyboard
reflowed the layout or the coordinates are a step old. It would then write into
the previously focused field, with nothing to show that it had. So Focus reads
focus back and checks it against the point asked for.

Focus holds the screen area as well as the driver. A focused element can cover
the whole screen (Flutter, WebView). It then contains every point, and cannot
answer the question.
"""

import logging
import time

from ronin.defaults import FOCUS_POLL_INTERVAL, FOCUS_POLL_TIMEOUT
from ronin.errors import ActionRejected

logger = logging.getLogger(__name__)

FULL_SCREEN_SHARE = 0.8


class Focus:
    """Reads focus back from one driver, against one screen size."""

    def __init__(self, driver, screen_area: int):
        self.driver = driver
        self.screen_area = screen_area

    def element(self):
        """The element holding focus, or None when the driver cannot say.

        Broad at this boundary. An unreachable session and a page with no focus
        mean the same thing here: nothing can be verified.
        """
        try:
            return self.driver.switch_to.active_element
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.debug("Could not read the focused element: %s", exc)
            return None

    def _rect(self):
        """The focused element's bounds, or None when there are none to read."""
        element = self.element()
        if element is None:
            return None
        try:
            return element.rect
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.debug("Could not read focused element bounds: %s", exc)
            return None

    def _covers_screen(self, rect) -> bool:
        width, height = rect.get("width", 0), rect.get("height", 0)
        return bool(self.screen_area) and width * height >= FULL_SCREEN_SHARE * self.screen_area

    @staticmethod
    def _holds(rect, x, y) -> bool:
        width, height = rect.get("width", 0), rect.get("height", 0)
        return (
            rect.get("x", 0) <= x <= rect.get("x", 0) + width
            and rect.get("y", 0) <= y <= rect.get("y", 0) + height
        )

    def contains(self, x, y) -> bool:
        """Whether the active element contains the requested coordinates.

        A focused element covering the screen (Flutter, WebView) answers for
        every point, so it cannot stand in for a real field.
        """
        rect = self._rect()
        if rect is None or self._covers_screen(rect):
            return False
        return self._holds(rect, x, y)

    def wait_for(self, x, y) -> None:
        """Polls until focus lands on (x, y), instead of sleeping a flat second."""
        for _ in range(max(1, int(FOCUS_POLL_TIMEOUT / FOCUS_POLL_INTERVAL))):
            if self.contains(x, y):
                return
            time.sleep(FOCUS_POLL_INTERVAL)

    def assert_at(self, x, y, action: str) -> None:
        """Raises unless focus is on the element at (x, y).

        Text goes to whatever holds focus. A tap can miss, because the keyboard
        shifted the layout or the coordinates are stale. It would then write
        into the previously focused field without any sign.

        A change of focus is not evidence that the right field was hit. Any
        text field opens the keyboard, so a tap on the wrong one looks
        identical to the layout reflowing. The planner reads the screen again
        and tries again. That costs a step and cannot type into the wrong
        field.
        """
        rect = self._rect()
        if rect is None:
            return

        logger.debug("Focused element after tap: %s", rect)
        if self._covers_screen(rect):
            # A single full-screen surface (Flutter, WebView) tells us nothing.
            logger.debug("Focused element covers the screen, so the check is skipped.")
            return

        if self._holds(rect, x, y):
            return

        width, height = rect.get("width", 0), rect.get("height", 0)
        raise ActionRejected(
            f"{action} at ({x}, {y}) did not focus anything there — focus is on the "
            f"element at x={rect.get('x')},y={rect.get('y')} "
            f"({width}x{height}). Nothing was entered. The layout has probably moved "
            "since these coordinates were read, so re-check the field's position."
        )
