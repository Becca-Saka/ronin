"""Files a run leaves on a device, and the ones it has to take back off.

A recording is named by the session that made it, so it can be removed
exactly. A session that died leaves whatever the recorder was still writing,
which is why the sweep matches a shape rather than a name.
"""

import logging
import re

from ronin.utils.subprocess_runner import run_command, SubprocessRunError

logger = logging.getLogger(__name__)

SCREENRECORD_DIR = "/sdcard"
"""Where the screenrecord binary writes, and what the sweep looks through."""

MEDIA_PROJECTION_DIR = "/sdcard/Android/data/io.appium.settings/files"
"""Where the Appium Settings helper keeps a Media Projection recording.

Under scoped storage, so a sweep of /sdcard does not reach it.
"""

STALE_RECORDING = re.compile(r"[0-9a-f]{8}\.mp4")
"""What Appium names a screenrecord file: the first eight characters of a uuid."""

RONIN_RECORDING = re.compile(r"ronin_[0-9a-fA-F-]+\.mp4")
"""What Ronin names a Media Projection recording: its session id."""


class DeviceFiles:
    """Reading, copying and removing files on a connected device."""

    @staticmethod
    def remove_file(serial: str, remote_path: str) -> None:
        """Deletes one file from the device, saying nothing if it is already gone."""
        try:
            run_command(
                ["adb", "-s", serial, "shell", "rm", "-f", remote_path],
                timeout=20,
                log_command=False,
            )
        except SubprocessRunError as exc:
            logger.debug("Could not remove %s: %s", remote_path, exc)

    @staticmethod
    def pull_file(serial: str, remote_path: str, local_path: str) -> bool:
        """Copies a file off the device, reporting whether it arrived."""
        try:
            run_command(
                ["adb", "-s", serial, "pull", remote_path, local_path],
                timeout=120,
                check=True,
            )
        except SubprocessRunError as exc:
            logger.debug("Could not pull %s: %s", remote_path, exc)
            return False
        return True

    @staticmethod
    def stop_screenrecord(serial: str, remote_path: str) -> None:
        """Interrupts the recorder writing one file, and no other.

        screenrecord writes the index that makes an MP4 playable only when it
        stops. A file taken from a recorder that is still running has no
        index, and no player will open it.

        Matched on the file being written, because a device can be recording
        more than once. A bare name would reach another run's recorder, or a
        recording a person started, and stop that instead.
        """
        try:
            run_command(
                [
                    "adb", "-s", serial, "shell",
                    "pkill", "-INT", "-f", f"screenrecord.*{remote_path}",
                ],
                timeout=20,
                log_command=False,
            )
        except SubprocessRunError as exc:
            logger.debug("Could not interrupt screenrecord on %s: %s", serial, exc)

    @staticmethod
    def list_directory(serial: str, remote_dir: str) -> list[str]:
        """Names of the entries in a device directory, empty when adb cannot answer."""
        try:
            result = run_command(
                ["adb", "-s", serial, "shell", "ls", remote_dir],
                timeout=20,
                log_command=False,
            )
        except SubprocessRunError as exc:
            logger.debug("Could not list %s on %s: %s", remote_dir, serial, exc)
            return []
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    @staticmethod
    def remove_stale_recordings(serial: str) -> int:
        """Deletes recordings that earlier sessions left on the device.

        Two recorders leave two kinds of file. `screenrecord` writes to
        /sdcard under the first eight characters of a uuid, and Appium removes
        its own only on a clean stop. Media Projection writes the name Ronin
        gave it, into the Settings helper's own directory, and keeps it.

        Both patterns are exact, so the sweep cannot reach a fixture or
        anything a person put on the device.
        """
        removed = 0
        for name in DeviceFiles.list_directory(serial, SCREENRECORD_DIR):
            if STALE_RECORDING.fullmatch(name):
                DeviceFiles.remove_file(serial, f"{SCREENRECORD_DIR}/{name}")
                removed += 1
        for name in DeviceFiles.list_directory(serial, MEDIA_PROJECTION_DIR):
            if RONIN_RECORDING.fullmatch(name):
                DeviceFiles.remove_file(serial, f"{MEDIA_PROJECTION_DIR}/{name}")
                removed += 1
        if removed:
            logger.info("Removed %d recording(s) left on the device.", removed)
        return removed
