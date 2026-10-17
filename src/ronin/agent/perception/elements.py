"""What perception gives the rest of the agent: where things are, what they are,
and what the pass that found them cost.

The contract lives here, not in `detector`, so that no producer owns it.
"""

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """A detected element's box in screenshot pixels, top-left origin."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __iter__(self):
        """Unpacks in the order that PIL's crop expects."""
        yield self.x_min
        yield self.y_min
        yield self.x_max
        yield self.y_max


class GUIElement(BaseModel):
    element: str
    bounding_box: BoundingBox
    confidence: float
    type: str
    text: str = ""


@dataclass
class ScanInfo:
    """What one pass over the screen cost and found."""

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
