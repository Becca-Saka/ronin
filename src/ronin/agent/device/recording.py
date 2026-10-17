"""Recording the screen for one session, on either platform.

Everything about the session video lives here: starting it, saving it to the
host, and removing what it left on the device.

Android has two recorders, chosen by USE_MEDIA_PROJECTION. screenrecord is a
shell binary that dies with the run; Media Projection names its file and can
report whether it started.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Optional, Protocol

from selenium.common.exceptions import WebDriverException

from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.files import MEDIA_PROJECTION_DIR, DeviceFiles
from ronin.agent.device.platform import Platform

logger = logging.getLogger(__name__)

START_MEDIA_PROJECTION = "mobile: startMediaProjectionRecording"
STOP_MEDIA_PROJECTION = "mobile: stopMediaProjectionRecording"
MEDIA_PROJECTION_RUNNING = "mobile: isMediaProjectionRecordingRunning"

RECORDING_START_ATTEMPTS = 2
"""Tries at starting the video before the run goes ahead without one."""

USE_MEDIA_PROJECTION = False
"""Whether Android records with Media Projection instead of screenrecord.

Media Projection names its file and reports whether it started, but Android
shows a screen-sharing indicator while it runs, which puts system UI on the
screen under test and outlives a killed run.
"""


class RecordingTarget(Protocol):
    """What the recorder needs from the session's device control.

    Named here so the recorder can ask for a driver and a serial without
    reaching for attributes it has not declared.
    """

    driver: Any
    serial: Optional[str]


class DeviceRecording:
    """The session video, for a DeviceSession to mix in.

    Expects the session to provide `platform`, `device_control`,
    `device_name`, `recordings_dir` and `session_id`.
    """

    platform: Platform
    device_control: Optional[RecordingTarget]
    device_name: str
    recordings_dir: Path
    session_id: str

    is_recording: bool = False
    _remote_name: Optional[str] = None
    _media_projection: bool = False

    @property
    def _driver(self) -> Any:
        """The Appium driver. Untyped, because the recording calls are vendor APIs."""
        return self.device_control.driver if self.device_control else None

    def _serial(self) -> Optional[str]:
        serial = self.device_control.serial if self.device_control else None
        return serial or AdbControl.resolve_emulator_serial(self.device_name)

    def start_recording(self) -> None:
        """Begins the video, or says that this run will not have one.

        The video is an artifact, not evidence a scenario rests on, so failing
        to start one must not end a run that could otherwise report results.
        """
        for attempt in range(1, RECORDING_START_ATTEMPTS + 1):
            try:
                if self._start_once():
                    self.is_recording = True
                    return
                logger.warning("The device reports a recording already running.")
                return
            except WebDriverException as exc:
                logger.warning(
                    "Screen recording did not start (attempt %d of %d): %s",
                    attempt,
                    RECORDING_START_ATTEMPTS,
                    exc,
                )
        logger.warning("Continuing without a video of this run.")

    def _start_once(self) -> bool:
        """Starts one recording, reporting whether it is now running."""
        if self.platform == Platform.ANDROID and USE_MEDIA_PROJECTION:
            return self._start_media_projection()
        self._driver.start_recording_screen()
        logger.info("Recording this run with screenrecord.")
        return True

    def _start_media_projection(self) -> bool:
        """Starts the Media Projection recorder, which can say whether it did."""
        self._remote_name = f"ronin_{self.session_id}.mp4"
        try:
            started = self._driver.execute_script(
                START_MEDIA_PROJECTION, {"filename": self._remote_name}
            )
        except WebDriverException as exc:
            logger.info("Media Projection is unavailable, using screenrecord: %s", exc)
            self._remote_name = None
            self._driver.start_recording_screen()
            return True

        if not started:
            return False
        self._media_projection = True
        running = bool(self._driver.execute_script(MEDIA_PROJECTION_RUNNING))
        if running:
            logger.info("Recording this run with Media Projection: %s", self._remote_name)
        return running

    def _free_path(self) -> Path:
        """Where to write this video without overwriting an earlier one.

        Session recovery reuses the session id, so the replacement session
        would land on the file the failed one just saved. The segment that
        recorded the failure is the one worth keeping.
        """
        path = self.recordings_dir / f"session_{self.session_id}.mp4"
        segment = 2
        while path.exists():
            path = self.recordings_dir / f"session_{self.session_id}_{segment}.mp4"
            segment += 1
        return path

    def save_recording(self) -> Optional[Path]:
        """Writes the video to the host, then clears it from the device."""
        if not self.is_recording:
            return None
        self.is_recording = False
        path = self._free_path()
        try:
            raw = (
                self._driver.execute_script(STOP_MEDIA_PROJECTION)
                if self._media_projection
                else self._driver.stop_recording_screen()
            )
        except WebDriverException as exc:
            logger.warning("Failed to stop the screen recording: %s", exc)
            return self._pull_recording(path)

        if not raw:
            logger.warning("The screen recording came back empty.")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(raw))
        logger.info("Session recording saved to %s", path)
        return path

    def _pull_recording(self, path: Path) -> Optional[Path]:
        """Copies the video off the device when the driver can no longer be asked.

        A session that stops responding is the one whose video is worth most,
        and it is the one the driver cannot hand over. Media Projection has
        already written the file under a name this session chose, so adb can
        fetch it without the driver.
        """
        serial = self._serial()
        if not self._remote_name or not serial:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        remote = f"{MEDIA_PROJECTION_DIR}/{self._remote_name}"
        if not DeviceFiles.pull_file(serial, remote, str(path)):
            return None
        logger.info("Recovered the session recording from the device: %s", path)
        return path

    def discard_device_recordings(self) -> None:
        """Removes recordings left on the device, after they have been saved.

        Media Projection keeps its file, and a session that dies before
        stopping leaves whatever the recorder was writing.
        """
        if self.platform != Platform.ANDROID:
            return
        serial = self._serial()
        if not serial:
            return
        if self._remote_name:
            DeviceFiles.remove_file(serial, f"{MEDIA_PROJECTION_DIR}/{self._remote_name}")
        DeviceFiles.remove_stale_recordings(serial)
