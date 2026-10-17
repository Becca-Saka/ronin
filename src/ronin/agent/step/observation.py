"""Reading the screen, and knowing when it is worth reading.

The snapshot is what the rest of the loop decides from, so one place builds it.
Perceive, record the transition, then number the targets once the keyboard
state is known. Settling and the drift check live here too. Both answer the
same question: is this screen ready to act on.
"""

import io
import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

import imagehash
from PIL import Image

from ronin.agent.device_planner import GROUNDING_MARKS
from ronin.agent.grounding import marks
from ronin.agent.perception import keyboard
from ronin.agent.perception.perception import ScreenPerception
from ronin.agent.perception.scan_dump import ScanInfo
from ronin.agent.trail import SessionTrail
from ronin.defaults import SETTLE_POLL_INTERVAL, SETTLE_STABLE_READINGS
from ronin.errors import ObservationUnavailable

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

    targets, keys = keyboard.split_keyboard(
        snapshot.gui_elements, ctx.keyboard_open, ctx.screen_height
    )
    raw_img = Image.open(io.BytesIO(snapshot.raw_img_bytes))
    width, height = raw_img.size
    built = marks.build(targets, screen_area=width * height)

    refused = ctx.refused_marks.get(snapshot.current_phash) or []
    snapshot.marks = [
        replace(mark, refuses_text=True)
        if any(marks.is_same_element(mark.element, box) for box in refused)
        else mark
        for mark in built
    ]

    marked = io.BytesIO()
    marks.render(
        raw_img, snapshot.marks, band=keyboard.keyboard_band(keys, width)
    ).save(marked, format="PNG")
    snapshot.planner_img_bytes = marked.getvalue()


def context_drift(
    device_control, last_external_nav: bool, settle_time: float
) -> Optional[str]:
    # Asked before anything else. On Android the prompt reads as lost focus and
    # recovery would press back, which answers it as "deny"; on iOS the app
    # stays in the foreground and nothing below would see the prompt at all.
    # The planner reads it off the screenshot like any other screen.
    if device_control.showing_system_prompt():
        logger.info("A system prompt is up; leaving it for the planner to answer.")
        return (
            "Info: the operating system is asking you something. Its prompt is on "
            "screen now. The prompt belongs to the system, not to the app. Answer "
            "it as the scenario requires. Do not leave the prompt. A dismissed "
            "permission prompt denies the permission."
        )

    if not device_control.is_still_in_sut():
        if not last_external_nav:
            logger.warning("Unexpected context drift. Attempting recovery...")
            device_control.recover_to_sut()
            time.sleep(settle_time)
            if not device_control.is_still_in_sut():
                logger.error("Recovery failed.")
                return (
                    "Warning: The device has drifted outside the target app (SUT) and recovery failed. "
                    "Please take an action to navigate back to the app, or terminate the scenario."
                )
            logger.info("Recovery successful.")
            return None
        return (
            "Info: The device is currently outside the target app (SUT) as expected. "
            "Make sure to return to the target app once the external tasks are completed."
        )
    return None


def settle(device_control, settle_time: float) -> Optional[str]:
    """Waits for the screen to stop changing, up to settle_time.

    A fixed sleep pays the worst case on every action. Most screens are static
    again well before it elapses. Returns the settled pHash, so the caller does
    not need a further screenshot.
    """
    deadline = max(1, int(settle_time / SETTLE_POLL_INTERVAL))
    phash = quick_phash(device_control)
    stable = 1
    for _ in range(deadline):
        time.sleep(SETTLE_POLL_INTERVAL)
        current = quick_phash(device_control)
        if current is not None and current == phash:
            stable += 1
            if stable >= SETTLE_STABLE_READINGS:
                return current
        else:
            stable = 1
        phash = current
    return phash


def quick_phash(device_control) -> Optional[str]:
    """Lightweight pHash capture after action — no YOLO/OCR, just a screenshot."""
    try:
        png_bytes = device_control.get_screenshot()
        img = Image.open(io.BytesIO(png_bytes))
        return str(imagehash.phash(img, hash_size=16))
    except (ObservationUnavailable, OSError):  # PIL raises OSError on bad bytes
        return None
