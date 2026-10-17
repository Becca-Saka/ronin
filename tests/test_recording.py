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
    RECORDING_TIME_LIMIT,
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
        refuses_time_limit=False,
    ):
        self.start, self.running, self.stop = start, running, stop
        self.fails, self.no_media_projection = fails, no_media_projection
        self.screenrecord_fails = screenrecord_fails
        self.refuses_time_limit = refuses_time_limit
        self.calls: list = []
        self.screenrecord_started = False
        self.options: list = []

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

    def start_recording_screen(self, **options):
        self.options.append(options)
        if self.refuses_time_limit and "timeLimit" in options:
            raise WebDriverException("timeLimit is out of range")
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


class RecordingTest(unittest.TestCase):
    """Keeps adb out of the tests that are not about what is on the device.

    `self.listing` answers every directory read. Give it a `side_effect` of
    one list per call to describe a device whose contents change.
    """

    def setUp(self):
        listing = patch(
            "ronin.agent.device.recording.DeviceFiles.list_directory", return_value=[]
        )
        self.listing = listing.start()
        self.addCleanup(listing.stop)
        flush = patch("ronin.agent.device.recording.time.sleep")
        flush.start()
        self.addCleanup(flush.stop)


class StartRecordingTests(RecordingTest):
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

    def test_the_recorder_is_asked_for_longer_than_three_minutes(self):
        """Appium stops at 180s by default and does not say that it did."""
        with TemporaryDirectory() as directory:
            session = self._session(directory)

            session.start_recording()

            self.assertEqual(
                session.device_control.driver.options,
                [{"timeLimit": RECORDING_TIME_LIMIT}],
            )

    def test_a_driver_that_refuses_the_limit_still_records(self):
        """Three minutes of video beats none, so the last attempt asks for nothing."""
        with TemporaryDirectory() as directory:
            session = self._session(directory, refuses_time_limit=True)

            session.start_recording()

            self.assertTrue(session.is_recording)
            self.assertEqual(session.device_control.driver.options[-1], {})

    def test_the_file_appium_named_is_found_by_listing_around_the_start(self):
        """Appium picks the name and never reports it.

        Without it, a run whose driver dies leaves a complete video on the
        device that nothing can ask for.
        """
        with TemporaryDirectory() as directory:
            self.listing.side_effect = [["fixture.mp4"], ["fixture.mp4", "a1b2c3d4.mp4"]]
            session = self._session(directory)

            session.start_recording()

            self.assertEqual(session._remote_path, "/sdcard/a1b2c3d4.mp4")

    def test_two_new_videos_name_nothing(self):
        """Guessing saves somebody else's video, or deletes one that is not ours."""
        with TemporaryDirectory() as directory:
            self.listing.side_effect = [[], ["one.mp4", "two.mp4"]]
            session = self._session(directory)

            session.start_recording()

            self.assertIsNone(session._remote_path)

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
            self.assertEqual(
                session._remote_path,
                "/sdcard/Android/data/io.appium.settings/files/ronin_abc123.mp4",
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

    def test_ios_records_through_the_driver(self):
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(), directory, platform=Platform.IOS)

            session.start_recording()

            self.assertTrue(session.device_control.driver.screenrecord_started)
            self.assertEqual(session.device_control.driver.calls, [])
            self.assertIsNone(session._remote_path)


class SaveRecordingTests(RecordingTest):
    """Getting the video onto the host before anything deletes it."""

    def _save(self, directory, **kwargs):
        with media_projection():
            session = Session(FakeDriver(**kwargs), directory)
            session.start_recording()
        return session, session.save_recording()

    def test_the_video_is_written_where_the_session_keeps_them(self):
        with TemporaryDirectory() as directory:
            encoded = base64.b64encode(PAYLOAD).decode()
            with patch("ronin.agent.device.recording.DeviceFiles.remove_file"):
                session, path = self._save(directory, stop=encoded)

            self.assertEqual(path, Path(directory) / "session_abc123_001.mp4")
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

    def test_a_saved_video_is_cleared_from_the_device(self):
        """It has a copy on the host by now, and the device fills up."""
        with TemporaryDirectory() as directory:
            encoded = base64.b64encode(PAYLOAD).decode()
            with patch(
                "ronin.agent.device.recording.DeviceFiles.remove_file"
            ) as remove:
                session, _ = self._save(directory, stop=encoded)

            remove.assert_called_once_with(
                "emulator-5554",
                "/sdcard/Android/data/io.appium.settings/files/ronin_abc123.mp4",
            )
            self.assertIsNone(session._remote_path)

    def test_a_video_that_could_not_be_saved_stays_on_the_device(self):
        """The copy on the device is the only one there is.

        A run that crashes before it can save leaves a complete video behind,
        and a sweep that cannot tell it from an abandoned one removes it.
        """
        with TemporaryDirectory() as directory:
            with media_projection():
                session = Session(FakeDriver(stop=None), directory)
                session.start_recording()
            session.device_control.driver.no_media_projection = True

            with (
                patch(
                    "ronin.agent.device.recording.DeviceFiles.pull_file",
                    return_value=False,
                ),
                patch("ronin.agent.device.recording.DeviceFiles.remove_file") as remove,
            ):
                self.assertIsNone(session.save_recording())

            remove.assert_not_called()
            self.assertIsNotNone(session._remote_path)

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

            with (
                patch(
                    "ronin.agent.device.recording.DeviceFiles.pull_file",
                    return_value=True,
                ) as pull,
                patch("ronin.agent.device.recording.DeviceFiles.remove_file"),
            ):
                path = session.save_recording()

            self.assertEqual(path, Path(directory) / "session_abc123_001.mp4")
            pull.assert_called_once_with(
                "emulator-5554",
                "/sdcard/Android/data/io.appium.settings/files/ronin_abc123.mp4",
                str(Path(directory) / "session_abc123_001.mp4"),
            )

    def test_a_screenrecord_video_is_pulled_once_the_recorder_is_interrupted(self):
        """screenrecord writes the index that makes an MP4 playable on the way out."""
        with TemporaryDirectory() as directory:
            self.listing.side_effect = [[], ["a1b2c3d4.mp4"]]
            session = Session(FakeDriver(stop=None), directory)
            session.start_recording()

            def dead_stop():
                raise WebDriverException("A session is either terminated or not started")

            session.device_control.driver.stop_recording_screen = dead_stop

            with (
                patch(
                    "ronin.agent.device.recording.DeviceFiles.stop_screenrecord"
                ) as interrupt,
                patch(
                    "ronin.agent.device.recording.DeviceFiles.pull_file",
                    return_value=True,
                ) as pull,
                patch("ronin.agent.device.recording.DeviceFiles.remove_file"),
            ):
                path = session.save_recording()

            interrupt.assert_called_once_with("emulator-5554", "/sdcard/a1b2c3d4.mp4")
            pull.assert_called_once_with(
                "emulator-5554",
                "/sdcard/a1b2c3d4.mp4",
                str(Path(directory) / "session_abc123_001.mp4"),
            )
            self.assertEqual(path, Path(directory) / "session_abc123_001.mp4")

    def test_an_unnamed_screenrecord_video_cannot_be_pulled(self):
        """The listing named nothing, so there is no file to ask adb for."""
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(no_media_projection=True), directory)
            session.start_recording()

            with patch("ronin.agent.device.recording.DeviceFiles.pull_file") as pull:
                session.device_control.driver.stop = None
                self.assertIsNone(session.save_recording())

            pull.assert_not_called()


class CheckpointRecordingTests(RecordingTest):
    """A video exists only once it is stopped, so the run is cut as it goes."""

    def _recording(self, directory):
        session = Session(
            FakeDriver(stop=base64.b64encode(PAYLOAD).decode()), directory
        )
        session.start_recording()
        return session

    def test_each_scenario_closes_a_segment_of_its_own(self):
        """A crash costs the segment in flight, not the whole session."""
        with TemporaryDirectory() as directory:
            session = self._recording(directory)

            first = session.checkpoint_recording()
            second = session.checkpoint_recording()

            self.assertEqual(first.name, "session_abc123_001.mp4")
            self.assertEqual(second.name, "session_abc123_002.mp4")
            self.assertTrue(session.is_recording)

    def test_a_run_without_a_video_has_nothing_to_cut(self):
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(), directory)

            self.assertIsNone(session.checkpoint_recording())
            self.assertFalse(session.is_recording)


class StaleRecordingTests(RecordingTest):
    """What earlier runs abandoned is swept on the way in, not on the way out."""

    def test_leftovers_are_cleared_before_this_run_records(self):
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(), directory)

            with patch(
                "ronin.agent.device.recording.DeviceFiles.remove_stale_recordings"
            ) as sweep:
                session.discard_stale_recordings()

            sweep.assert_called_once_with("emulator-5554")

    def test_ios_recordings_are_left_alone(self):
        """simctl holds them elsewhere, and adb cannot reach a simulator."""
        with TemporaryDirectory() as directory:
            session = Session(FakeDriver(), directory, platform=Platform.IOS)

            with patch(
                "ronin.agent.device.recording.DeviceFiles.remove_stale_recordings"
            ) as sweep:
                session.discard_stale_recordings()

            sweep.assert_not_called()


class RecordingPathTests(RecordingTest):
    """Session recovery reuses the session id, so two videos want one name."""

    def test_a_second_session_does_not_overwrite_the_first(self):
        """The segment that recorded the failure is the one worth keeping."""
        with TemporaryDirectory() as directory:
            with patch("ronin.agent.device.recording.DeviceFiles.remove_file"):
                first = Session(
                    FakeDriver(stop=base64.b64encode(PAYLOAD).decode()), directory
                )
                first.start_recording()
                first_path = first.save_recording()

                second = Session(
                    FakeDriver(stop=base64.b64encode(b"second").decode()), directory
                )
                second.start_recording()
                second_path = second.save_recording()

            self.assertEqual(first_path.name, "session_abc123_001.mp4")
            self.assertEqual(second_path.name, "session_abc123_002.mp4")
            self.assertEqual(first_path.read_bytes(), PAYLOAD)
            self.assertEqual(second_path.read_bytes(), b"second")
