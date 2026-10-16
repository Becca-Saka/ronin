"""What a run removes from the device, and what it must leave alone.

A rotation of apps fills a device, and a full device fails runs for reasons
that look nothing like storage. Teardown removes what the run put there, which
means it has to know what it put there.
"""

import unittest
from unittest.mock import patch

from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.device_files import DeviceFiles
from ronin.agent.device.device_session import DeviceSession
from ronin.agent.device.platform import Platform


class StaleRecordingSweepTests(unittest.TestCase):
    """Removing recordings earlier sessions left behind."""

    def _sweep(self, sdcard, helper_dir):
        listings = {
            "/sdcard/": sdcard,
            "/sdcard/Android/data/io.appium.settings/files": helper_dir,
        }
        with (
            patch.object(DeviceFiles, "list_directory", side_effect=lambda _s, d: listings[d]),
            patch.object(DeviceFiles, "remove_file") as remove,
        ):
            count = DeviceFiles.remove_stale_recordings("emulator-5554")
        return count, [call.args[1] for call in remove.call_args_list]

    def test_an_appium_screenrecord_file_is_removed(self):
        """Appium names them after eight characters of a uuid."""
        count, removed = self._sweep(["dd1c421d.mp4"], [])

        self.assertEqual(count, 1)
        self.assertEqual(removed, ["/sdcard/dd1c421d.mp4"])

    def test_a_ronin_media_projection_file_is_removed(self):
        count, removed = self._sweep([], ["ronin_32424254-28e2-4508.mp4"])

        self.assertEqual(count, 1)
        self.assertEqual(
            removed,
            ["/sdcard/Android/data/io.appium.settings/files/ronin_32424254-28e2-4508.mp4"],
        )

    def test_test_fixtures_on_the_device_are_left_alone(self):
        """Journeys need media pushed to the device, and it has to survive."""
        count, removed = self._sweep(
            ["bbb_360.mp4", "bbb_external_subs.mp4", "Movies", "holiday.mp4"], []
        )

        self.assertEqual(count, 0)
        self.assertEqual(removed, [])

    def test_a_name_that_merely_contains_a_uuid_is_not_swept(self):
        """fullmatch, not search: my_dd1c421d.mp4 belongs to whoever made it."""
        count, _ = self._sweep(["my_dd1c421d.mp4", "dd1c421d.mp4.bak"], [])

        self.assertEqual(count, 0)


class OwnershipDetectionTests(unittest.TestCase):
    """Deciding whether this run is the one that installed the app.

    The answer has to be positive knowledge, because the consequence of
    guessing wrong is uninstalling somebody else's app.
    """

    def _session(self, serial, packages, package="org.fossify.calendar"):
        session = DeviceSession.__new__(DeviceSession)
        session.platform = Platform.ANDROID
        session.device_name = "Pixel_8"
        session.app_path = "calendar.apk"
        session._installed_by_run = None
        with (
            patch.object(AdbControl, "resolve_emulator_serial", return_value=serial),
            patch.object(AdbControl, "installed_packages", return_value=packages),
            patch.object(AdbControl, "get_apk_info", return_value=(package, None)),
        ):
            session._note_app_ownership()
        return session

    def test_an_app_absent_before_the_run_belongs_to_the_run(self):
        session = self._session("emulator-5554", {"com.android.settings"})

        self.assertTrue(session._installed_by_run)

    def test_an_app_already_there_does_not(self):
        session = self._session(
            "emulator-5554", {"com.android.settings", "org.fossify.calendar"}
        )

        self.assertFalse(session._installed_by_run)

    def test_a_device_that_has_not_booted_answers_nothing(self):
        """Appium boots the AVD after this runs, and it may hold the app."""
        session = self._session(None, {"anything"})

        self.assertFalse(session._installed_by_run)

    def test_an_adb_call_that_fails_answers_nothing(self):
        """installed_packages returns an empty set when it cannot ask."""
        session = self._session("emulator-5554", set())

        self.assertFalse(session._installed_by_run)

    def test_an_answer_the_caller_supplied_is_not_recalculated(self):
        """Recovery knows what the failed session learned.

        By then the app is installed, so looking again would call it
        preexisting and leave it on the device.
        """
        session = DeviceSession.__new__(DeviceSession)
        session.platform = Platform.ANDROID
        session._installed_by_run = True

        with patch.object(AdbControl, "resolve_emulator_serial") as resolve:
            session._note_app_ownership()

        self.assertTrue(session._installed_by_run)
        resolve.assert_not_called()


class UninstallPolicyTests(unittest.TestCase):
    """Teardown removes the app this run installed, and nothing else."""

    def _session(self, **overrides):
        session = DeviceSession.__new__(DeviceSession)
        session.platform = Platform.ANDROID
        session.device_name = "Pixel_8"
        session.keep_app = False
        session.reuse_app = False
        session._installed_by_run = True
        session.device_control = type(
            "Control", (), {"serial": "emulator-5554", "package_id": "org.fossify.calendar"}
        )()
        for name, value in overrides.items():
            setattr(session, name, value)
        return session

    def _clean(self, session, final=True):
        with (
            patch.object(DeviceSession, "discard_device_recordings"),
            patch.object(AdbControl, "uninstall_app") as uninstall,
        ):
            session._clean_device(final=final)
        return uninstall

    def test_session_recovery_keeps_the_app(self):
        """Recovery opens a new session against the same app.

        The scenarios still to run need it, so a close that is not the end of
        the run must leave it installed.
        """
        uninstall = self._clean(self._session(), final=False)

        uninstall.assert_not_called()

    def test_recovery_still_clears_recordings(self):
        """The dead session's video is saved by then, and its file remains."""
        session = self._session()
        with (
            patch.object(DeviceSession, "discard_device_recordings") as discard,
            patch.object(AdbControl, "uninstall_app"),
        ):
            session._clean_device(final=False)

        discard.assert_called_once()

    def test_an_app_this_run_installed_is_removed(self):
        uninstall = self._clean(self._session())

        uninstall.assert_called_once_with("emulator-5554", "org.fossify.calendar")

    def test_keep_app_leaves_it_installed(self):
        uninstall = self._clean(self._session(keep_app=True))

        uninstall.assert_not_called()

    def test_an_app_that_was_already_installed_is_never_removed(self):
        """It belongs to whoever put it there, not to this run."""
        uninstall = self._clean(self._session(_installed_by_run=False))

        uninstall.assert_not_called()

    def test_an_app_of_unknown_ownership_is_left_alone(self):
        """A cold emulator cannot be asked, and silence is not permission.

        Detection runs before Appium boots the AVD, so on a cold device
        nothing is known. Uninstalling then would remove an app the run
        never installed.
        """
        uninstall = self._clean(self._session(_installed_by_run=None))

        uninstall.assert_not_called()

    def test_reusing_a_build_keeps_it(self):
        """--reuse-app means that build on the device is the one under test."""
        uninstall = self._clean(self._session(reuse_app=True))

        uninstall.assert_not_called()

    def test_nothing_is_uninstalled_without_a_package(self):
        """A session that died before resolving one has no package to name."""
        control = type("Control", (), {"serial": "emulator-5554", "package_id": None})()
        uninstall = self._clean(self._session(device_control=control))

        uninstall.assert_not_called()

    def test_an_ios_session_is_left_to_simctl(self):
        """adb cannot reach a simulator."""
        session = self._session(platform=Platform.IOS)

        with (
            patch.object(DeviceSession, "discard_device_recordings") as discard,
            patch.object(AdbControl, "uninstall_app") as uninstall,
        ):
            session._clean_device(final=True)

        uninstall.assert_not_called()
        discard.assert_not_called()

    def test_recordings_are_cleared_before_the_app_goes(self):
        """Uninstalling first would take the app's directory with it."""
        order = []
        session = self._session()
        with (
            patch.object(
                DeviceSession,
                "discard_device_recordings",
                side_effect=lambda: order.append("recordings"),
            ),
            patch.object(
                AdbControl, "uninstall_app", side_effect=lambda *_: order.append("app")
            ),
        ):
            session._clean_device(final=True)

        self.assertEqual(order, ["recordings", "app"])
