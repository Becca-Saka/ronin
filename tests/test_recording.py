"""The session video: starting it, saving it, and not losing a run over it."""

import base64
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from selenium.common.exceptions import WebDriverException

from ronin.agent.device.platform import Platform
from ronin.agent.device.recording import (
    MEDIA_PROJECTION_RUNNING,
    START_MEDIA_PROJECTION,
    STOP_MEDIA_PROJECTION,
    DeviceRecording,
)

PAYLOAD = b"\x00\x01video bytes"


def media_projection():
    """Opts a test into the recorder that is not the default."""
    return patch("ronin.agent.device.recording.USE_MEDIA_PROJECTION", True)


class FakeDriver:
    """An Appium driver that answers the recording calls."""

    def __init__(
        self,
        start=True,
        running=True,
        stop=None,
        fails=0,
        no_media_projection=False,
        screenrecord_fails=0,
    ):
        self.start, self.running, self.stop = start, running, stop
        self.fails, self.no_media_projection = fails, no_media_projection
        self.screenrecord_fails = screenrecord_fails
        self.calls: list = []
        self.screenrecord_started = False

    def execute_script(self, script, *args):
        self.calls.append((script, args[0] if args else None))
        if self.no_media_projection:
            raise WebDriverException("Unknown mobile command")
        if script == START_MEDIA_PROJECTION:
            if self.fails:
                self.fails -= 1
                raise WebDriverException("could not start")
            return self.start
        if script == MEDIA_PROJECTION_RUNNING:
            return self.running
        if script == STOP_MEDIA_PROJECTION:
            return self.stop
        raise AssertionError(f"unexpected script {script}")

    def start_recording_screen(self):
        if self.screenrecord_fails:
            self.screenrecord_fails -= 1
            raise WebDriverException("screen record file does not exist after 5000ms")
        self.screenrecord_started = True

    def stop_recording_screen(self):
        return self.stop


class Session(DeviceRecording):
    """The smallest host the mixin needs."""

    def __init__(self, driver, directory, platform=Platform.ANDROID):
        self.platform = platform
        self.device_control = SimpleNamespace(driver=driver, serial="emulator-5554")
        self.device_name = "Pixel_8"
        self.recordings_dir = Path(directory)
        self.session_id = "abc123"


class StartRecordingTests(unittest.TestCase):
    """A video that will not start must not end a run that could report results."""

    def _session(self, directory, **kwargs):
        return Session(FakeDriver(**kwargs), directory)

    def test_screenrecord_is_what_android_uses(self):
        """It shows nothing on screen and dies with the run.

        Media Projection makes Android display a screen-sharing indicator,
        and a dialog from it lands on the screen under test.
        """
        with TemporaryDirectory() as directory:
            session = self._session(directory)

            session.start_recording()

            self.assertTrue(session.is_recording)
            self.assertTrue(session.device_control.driver.screenrecord_started)
            self.assertEqual(session.device_control.driver.calls, [])
            self.assertIsNone(session._remote_name)

    def test_a_started_recording_is_named_after_the_session(self):
        """Knowing the name is what lets teardown remove exactly this file."""
        with TemporaryDirectory() as directory, media_projection():
            session = self._session(directory)

            session.start_recording()

            self.assertTrue(session.is_recording)
            self.assertEqual(
                session.device_control.driver.calls[0],
                (START_MEDIA_PROJECTION, {"filename": "ronin_abc123.mp4"}),
            )

    def test_the_device_is_asked_whether_it_is_really_recording(self):
        """Media Projection can report a start and not be running."""
        with TemporaryDirectory() as directory, media_projection():
            session = self._session(directory, running=False)

            session.start_recording()

            self.assertFalse(session.is_recording)

    def test_a_refused_start_does_not_end_the_run(self):
        """It returns False, rather than raising, when one is already running."""
        with TemporaryDirectory() as directory, media_projection():
            session = self._session(directory, start=False)

            session.start_recording()

            self.assertFalse(session.is_recording)

    def test_a_slow_screenrecord_start_is_tried_again(self):
        """Appium allows screenrecord a fixed 5s, which a loaded emulator loses.

        The second attempt meets a warm binary, so one failure is not the end.
        """
        with TemporaryDirectory() as directory:
            session = self._session(
                directory, no_media_projection=True, screenrecord_fails=1
            )

            session.start_recording()

            self.assertTrue(session.is_recording)

    def test_a_recorder_that_never_starts_leaves_the_run_going(self):
        """Neither mechanism works, and the scenarios still have to run."""
        with TemporaryDirectory() as directory:
            session = self._session(
                directory, no_media_projection=True, screenrecord_fails=99
            )

            session.start_recording()

            self.assertFalse(session.is_recording)

    def test_a_device_without_media_projection_uses_screenrecord(self):
        with TemporaryDirectory() as directory:
            session = self._session(directory, no_media_projection=True)

            session.start_recording()

            self.assertTrue(session.is_recording)
            self.assertTrue(session.device_control.driver.screenrecord_started)
            self.assertIsNone(session._remote_name)

    def test_ios_records_through_the_driver(self):
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(), directory, platform=Platform.IOS)

            session.start_recording()

            self.assertTrue(session.device_control.driver.screenrecord_started)
            self.assertEqual(session.device_control.driver.calls, [])


class SaveRecordingTests(unittest.TestCase):
    """Getting the video onto the host before anything deletes it."""

    def _save(self, directory, **kwargs):
        with media_projection():
            session = Session(FakeDriver(**kwargs), directory)
            session.start_recording()
        return session, session.save_recording()

    def test_the_video_is_written_where_the_session_keeps_them(self):
        with TemporaryDirectory() as directory:
            encoded = base64.b64encode(PAYLOAD).decode()
            session, path = self._save(directory, stop=encoded)

            self.assertEqual(path, Path(directory) / "session_abc123.mp4")
            self.assertEqual(path.read_bytes(), PAYLOAD)
            self.assertFalse(session.is_recording)

    def test_nothing_is_saved_when_no_recording_started(self):
        """Stopping one that never began reports an error on a real device."""
        with TemporaryDirectory() as directory:
            with media_projection():
                session = Session(FakeDriver(start=False), directory)
                session.start_recording()

            self.assertIsNone(session.save_recording())
            self.assertEqual(
                [call[0] for call in session.device_control.driver.calls],
                [START_MEDIA_PROJECTION],
            )

    def test_an_empty_video_is_reported_rather_than_written(self):
        """A zero-byte file looks like a recording and holds nothing."""
        with TemporaryDirectory() as directory:
            _, path = self._save(directory, stop="")

            self.assertIsNone(path)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_a_dead_driver_does_not_end_the_run(self):
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(stop=None), directory)
            session.start_recording()
            session.device_control.driver.no_media_projection = True

            with patch(
                "ronin.agent.device.recording.DeviceFiles.pull_file", return_value=False
            ):
                self.assertIsNone(session.save_recording())

    def test_a_video_the_driver_cannot_hand_over_is_pulled_instead(self):
        """A session that stopped responding is the one worth recording.

        Media Projection has already written the file under a name this
        session chose, so adb can fetch it without the driver.
        """
        with TemporaryDirectory() as directory:
            with media_projection():
                session = Session(FakeDriver(stop=None), directory)
                session.start_recording()
            session.device_control.driver.no_media_projection = True

            with patch(
                "ronin.agent.device.recording.DeviceFiles.pull_file", return_value=True
            ) as pull:
                path = session.save_recording()

            self.assertEqual(path, Path(directory) / "session_abc123.mp4")
            pull.assert_called_once_with(
                "emulator-5554",
                "/sdcard/Android/data/io.appium.settings/files/ronin_abc123.mp4",
                str(Path(directory) / "session_abc123.mp4"),
            )

    def test_a_screenrecord_video_cannot_be_pulled(self):
        """Appium picked that file name, and never told us what it was."""
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(no_media_projection=True), directory)
            session.start_recording()

            with patch(
                "ronin.agent.device.recording.DeviceFiles.pull_file"
            ) as pull:
                session.device_control.driver.stop = None
                session._media_projection = False
                self.assertIsNone(session.save_recording())

            pull.assert_not_called()


class DiscardRecordingTests(unittest.TestCase):
    """Media Projection keeps its file, so the session has to remove it."""

    def test_the_named_recording_is_removed_from_the_device(self):
        with TemporaryDirectory() as directory:
            with media_projection():
                session = Session(FakeDriver(), directory)
                session.start_recording()

            with (
                patch("ronin.agent.device.recording.DeviceFiles.remove_file") as remove,
                patch("ronin.agent.device.recording.DeviceFiles.remove_stale_recordings"),
            ):
                session.discard_device_recordings()

            remove.assert_called_once_with(
                "emulator-5554",
                "/sdcard/Android/data/io.appium.settings/files/ronin_abc123.mp4",
            )

    def test_older_recordings_are_swept_as_well(self):
        """A session that died never removed what it was writing."""
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(), directory)

            with (
                patch("ronin.agent.device.recording.DeviceFiles.remove_file"),
                patch(
                    "ronin.agent.device.recording.DeviceFiles.remove_stale_recordings"
                ) as sweep,
            ):
                session.discard_device_recordings()

            sweep.assert_called_once_with("emulator-5554")

    def test_ios_recordings_are_left_alone(self):
        """simctl holds them elsewhere, and adb cannot reach a simulator."""
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(), directory, platform=Platform.IOS)

            with patch(
                "ronin.agent.device.recording.DeviceFiles.remove_stale_recordings"
            ) as sweep:
                session.discard_device_recordings()

            sweep.assert_not_called()


class RecordingPathTests(unittest.TestCase):
    """Session recovery reuses the session id, so two videos want one name."""

    def test_a_second_recording_does_not_overwrite_the_first(self):
        """The segment that recorded the failure is the one worth keeping."""
        with TemporaryDirectory() as directory:
            first = Session(FakeDriver(stop=base64.b64encode(PAYLOAD).decode()), directory)
            first.start_recording()
            first_path = first.save_recording()

            second = Session(FakeDriver(stop=base64.b64encode(b"second").decode()), directory)
            second.start_recording()
            second_path = second.save_recording()

            self.assertEqual(first_path.name, "session_abc123.mp4")
            self.assertEqual(second_path.name, "session_abc123_2.mp4")
            self.assertEqual(first_path.read_bytes(), PAYLOAD)
            self.assertEqual(second_path.read_bytes(), b"second")
