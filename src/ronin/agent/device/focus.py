"""Whether the text about to be typed will land in the field that was tapped.

Text goes to whatever holds focus. A tap can miss, because the keyboard
reflowed the layout or the coordinates are a step old. It would then write into
the previously focused field, with nothing to show that it had. So Focus reads
focus back and checks it against the point asked for, and refuses the action
when it cannot.

Focus holds the screen area as well as the driver, because some apps render as
one surface and focus it whole. An element that covers the screen contains
every point, so it cannot answer the question. It passes unchecked rather than
blocking every type in such an app.
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

        An element covering the screen answers for every point, so it cannot
        stand in for a real field.
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

    def rect(self):
        """The focused element's bounds, for comparing focus before and after."""
        return self._rect()

    def assert_at(self, x, y, action: str, before=None) -> None:
        """Raises unless focus is on the element at (x, y).

        A change of focus is not evidence that the right field was hit. Any
        text field opens the keyboard, so a tap on the wrong one looks
        identical to the layout reflowing.

        `before` is focus as it stood before the tap. Unchanged focus means the
        tap landed on something that cannot hold text, which is a different
        problem from a layout that moved, and needs a different answer.
        """
        rect = self._rect()
        if rect is None:
            raise ActionRejected(
                f"{action} at ({x}, {y}) could not be checked, because nothing "
                "reported focus. Nothing was entered. Read the screen again and "
                "choose the field to use."
            )

        logger.debug("Focused element after tap: %s", rect)
        if self._covers_screen(rect):
            logger.debug("Focused element covers the screen, so the check is skipped.")
            return

        if self._holds(rect, x, y):
            return

        raise ActionRejected(f"{action} {self._why(rect, before)} Nothing was entered.")

    @staticmethod
    def _why(rect, before) -> str:
        """What went wrong, in terms of the next thing to try.

        The planner reads this and chooses again, so it has to name a different
        move. Told only that coordinates went stale, one run retried the same
        target five times: in marks mode the planner never sees a coordinate,
        so it could do nothing with that.
        """
        if before is not None and rect == before:
            return (
                "did not reach the target you chose, which cannot hold text — a "
                "label, or a picture of a field rather than the field itself. "
                "Focus never moved. Choose the element that takes the typing, or "
                "scroll if it is off screen or behind the keyboard."
            )
        return (
            "landed on a different element than the one you chose, so the screen "
            "has moved since it was read. Read the screen again before choosing."
        )
