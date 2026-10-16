"""What each step leaves behind on disk.

Both writers go through the media manager rather than straight to disk. These
files outlive the run, so the manager covers a declared secret on screen first.
The scan dump goes beside them, which keeps a step's screenshots and the
reading they came from together.
"""

import logging
from pathlib import Path

from ronin.agent.observation import PerceptionSnapshot
from ronin.agent.perception.media_manager import MediaManager
from ronin.agent.perception.scan_dump import (
    build_screen_scan_record,
    write_screen_scan_dump,
)
from ronin.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)


def save_step(
    media_manager: MediaManager,
    scrubber: SecretScrubber,
    session_dir: Path,
    step: int,
    snapshot: PerceptionSnapshot,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    base = f"step{step + 1}_{snapshot.timestamp}"
    annotated_path = session_dir / f"{base}.png"
    raw_path = session_dir / f"{base}_raw.png"
    scan_path = session_dir / f"{base}_scan.json"

    media_manager.save_redacted_pair(
        snapshot.annotated_img, snapshot.raw_img_bytes,
        snapshot.gui_elements, annotated_path, raw_path,
    )

    if snapshot.scan_info is None:
        return

    record = build_screen_scan_record(
        step=step + 1,
        timestamp=snapshot.timestamp,
        scan_info=snapshot.scan_info,
        gui_elements=snapshot.gui_elements,
        screenshot_paths={
            "raw": raw_path.name,
            "annotated": annotated_path.name,
            "scan": scan_path.name,
        },
        prev_phash=snapshot.prev_phash,
        scrubber=scrubber,
    )
    write_screen_scan_dump(scan_path, record)
    logger.debug("Perception scan dump saved to %s", scan_path)


def save_action(
    media_manager: MediaManager,
    session_dir: Path,
    step: int,
    snapshot: PerceptionSnapshot,
    action,
) -> None:
    click_path = session_dir / f"action_step{step + 1}_{snapshot.timestamp}.png"
    media_manager.save_redacted_action_annotation(
        snapshot.annotated_img, snapshot.gui_elements, action, click_path
    )
