"""What every grounder answers with.

The contract lives here, not with a producer, so that no mode owns it.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

from ronin.agent.perception.elements import BoundingBox, GUIElement

MIN_PRESENCE = 0.5
"""How sure the model must be that the target is on the screen at all."""

MAX_SCREEN_FRACTION = 0.5
"""A box larger than this has found the screen rather than a target."""


@dataclass(frozen=True)
class Placement:
    """Where one described target is, and what the answer is worth."""

    box: BoundingBox
    present: float
    """How sure the model is that the target is on the screen. The gate."""

    box_confidence: Optional[float]
    """How sure the model is of the box. Reported, never a gate: it falls with
    the target's size whether or not the box is right."""

    @property
    def point(self) -> tuple[int, int]:
        """The point to touch."""
        return (
            (self.box.x_min + self.box.x_max) // 2,
            (self.box.y_min + self.box.y_max) // 2,
        )

    def as_element(self) -> GUIElement:
        """The same answer in the form perception hands to the rest of the agent."""
        return GUIElement(
            element="View",
            bounding_box=self.box,
            confidence=self.box_confidence or 0.0,
            type="grounded",
            text="",
        )


class Grounder(Protocol):
    """Anything that can place a described target on a screen."""

    def ground(self, screenshot: bytes, description: str) -> Optional[Placement]:
        """Where the described target is, or None when it cannot be placed."""
        ...
