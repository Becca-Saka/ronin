"""Grounding by description: the planner names a target, a model places it."""

import logging
from typing import Optional

from ronin.agent.grounding.placement import (
    MAX_SCREEN_FRACTION,
    MIN_PRESENCE,
    Placement,
)
from ronin.agent.perception.elements import BoundingBox

logger = logging.getLogger(__name__)


class DecisionGrounder:
    """A grounder backed by a decision model.

    One request asks two questions: where the target is, and whether it is
    there at all. The frame costs the same either way. The second question
    decides. The first one never does.
    """

    def __init__(self, client, min_presence: float = MIN_PRESENCE):
        self.client = client
        self.min_presence = min_presence

    def ground(self, screenshot: bytes, description: str) -> Optional[Placement]:
        from ronin.llm.jev_client import device_size, to_device

        # Plain mappings, so this seam needs nothing from the vendor package.
        response = self.client.ask(
            {
                "where": {"type": "box", "instructions": description},
                "present": {
                    "type": "noul",
                    "instructions": f"Is this visible in the image: {description}",
                },
            },
            screenshot=screenshot,
        )
        edges = response.answers["where"].box
        box_confidence = response.answers["where"].confidence
        present = response.answers["present"].noul

        fraction = (edges.right - edges.left) * (edges.bottom - edges.top)
        if present < self.min_presence or fraction > MAX_SCREEN_FRACTION:
            logger.info(
                "Could not place %r (present %.2f, covering %.0f%% of the screen)",
                description, present, fraction * 100,
            )
            return None

        x_min, y_min, x_max, y_max = to_device(
            device_size(screenshot), edges.left, edges.top, edges.right, edges.bottom
        )
        logger.info(
            "Grounded %r at (%d, %d, %d, %d), box confidence %s",
            description, x_min, y_min, x_max, y_max, box_confidence,
        )
        return Placement(
            box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            present=present,
            box_confidence=box_confidence,
        )
