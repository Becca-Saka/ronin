"""What perception hands the rest of the agent: where things are, and what they are.

The contract lives here rather than in `image_descriptor` so that it is not
owned by one of its producers.
"""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """A detected element's box in screenshot pixels, top-left origin."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __iter__(self):
        """Unpacks in PIL's crop order, which is where these boxes mostly end up."""
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
