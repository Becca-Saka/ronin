"""What perception gives the rest of the agent: where things are, and what they are.

The contract lives here, not in `image_descriptor`, so that no producer owns it.
"""

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
