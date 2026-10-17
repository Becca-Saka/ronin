"""Building the snapshot the rest of the loop decides from.

Perceive, record the transition, then number the targets once the keyboard
state is known.
"""

import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from ronin.agent.grounding import GROUNDING_MARKS
from ronin.agent.grounding import marks
from ronin.agent.perception.screen import ScreenPerception
from ronin.agent.perception.elements import ScanInfo
from ronin.agent.memory.trail import SessionTrail

logger = logging.getLogger(__name__)


@dataclass
class PerceptionSnapshot:
    """One reading of the screen, and everything decided from it."""

    gui_elements: list
    marks: list
    raw_img_bytes: bytes
    planner_img_bytes: bytes
    annotated_img: Image.Image
    current_phash: Optional[str]
    timestamp: str
    scan_info: Optional[ScanInfo] = None
    prev_phash: Optional[str] = None


def capture(
    perception: ScreenPerception,
    session_trail: Optional[SessionTrail],
    prev_phash: Optional[str],
) -> PerceptionSnapshot:
    gui_elements, raw_img, annotated_img = perception.perceive()
    current_phash = (
        str(perception.current_hash) if perception.current_hash else None
    )
    logger.info(
        "Perceived %d elements. pHash: %s", len(gui_elements), current_phash
    )

    if session_trail and current_phash:
        if prev_phash and current_phash != prev_phash:
            session_trail.flush_transition(current_phash)
        session_trail.record_screen(current_phash, "", gui_elements)

    buf = io.BytesIO()
    raw_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    return PerceptionSnapshot(
        gui_elements=gui_elements,
        marks=[],
        raw_img_bytes=raw_bytes,
        planner_img_bytes=raw_bytes,
        annotated_img=annotated_img,
        current_phash=current_phash,
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
        scan_info=perception.last_scan_info,
        prev_phash=prev_phash,
    )


def number_targets(snapshot: PerceptionSnapshot, ctx, grounding: str) -> None:
    """Numbers the elements and redraws the screenshot to match.

    Done here rather than at perception time, because only now is the keyboard
    state known. The keyboard's own keys must not carry numbers: they would
    outnumber the app's controls several times over.
    """
    if grounding != GROUNDING_MARKS:
        return
    snapshot.marks, snapshot.planner_img_bytes = marks.number(
        snapshot.gui_elements,
        snapshot.raw_img_bytes,
        keyboard_open=ctx.keyboard_open,
        screen_height=ctx.screen_height,
        refused=ctx.refused_marks.get(snapshot.current_phash) or [],
    )
