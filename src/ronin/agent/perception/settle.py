"""Waiting for a screen to stop changing, and reading it cheaply.

A perceptual hash answers both questions here, which costs a screenshot and no
detection at all.
"""

import io
import logging
import time
from typing import Optional

import imagehash
from PIL import Image

from ronin.defaults import SETTLE_POLL_INTERVAL, SETTLE_STABLE_READINGS
from ronin.errors import ObservationUnavailable

logger = logging.getLogger(__name__)


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
