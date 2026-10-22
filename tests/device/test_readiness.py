"""Checking the device before a run commits to it.

Generating a journey's scenarios costs a model call and a minute, and it runs
before anything touches the device. A device that cannot be driven was paid
for first and discovered afterwards, one failed scenario at a time.
"""

import unittest
from unittest.mock import patch

from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.platform import Platform
from ronin.agent.device.readiness import ensure_device_ready
from ronin.errors import DeviceUnavailable


class AndroidReadinessTests(unittest.TestCase):
    """What adb says about the device, before the run spends anything."""

    def _check(self, attached, serial, booted=True, device="Pixel_8", avds=("Pixel_8",)):
        with (
            patch.object(AdbControl, "attached_devices", return_value=attached),
            patch.object(AdbControl, "resolve_emulator_serial", return_value=serial),
            patch.object(AdbControl, "boot_completed", return_value=booted),
            patch.object(
                AdbControl, "list_avds", return_value=None if avds is None else list(avds)
            ),
        ):
            ensure_device_ready(Platform.ANDROID, device)

    def test_a_booted_device_is_ready(self):
        self._check({"emulator-5554": "device"}, "emulator-5554")

    def test_a_named_emulator_that_exists_may_be_started(self):
        """Nothing attached, but Ronin can boot the one the file names."""
        self._check({}, None)

    def test_another_device_being_attached_is_not_a_problem(self):
        """The named emulator is simply not running yet."""
        self._check({"emulator-5556": "device"}, None)

    def test_a_device_that_is_neither_up_nor_startable_stops_the_run(self):
        """The file names the device the run is for. If it is not attached and
        no emulator of that name exists, nothing can make the run meaningful."""
        with self.assertRaises(DeviceUnavailable) as caught:
            self._check({}, None, device="Pixel_8a", avds=("Pixel_8",))

        message = str(caught.exception)
        self.assertIn("Pixel_8a", message)
        self.assertIn("Pixel_8", message)

    def test_no_emulators_at_all_stops_the_run(self):
        with self.assertRaises(DeviceUnavailable) as caught:
            self._check({}, None, avds=())

        self.assertIn("none", str(caught.exception))

    def test_an_unanswerable_emulator_list_is_taken_on_trust(self):
        """Refusing because the question could not be asked would be worse."""
        self._check({}, None, device="Pixel_99", avds=None)

    def test_an_offline_device_stops_the_run(self):
        """resolve_emulator_serial only matches the 'device' state, so an
        offline emulator used to look exactly like no emulator at all."""
        with self.assertRaises(DeviceUnavailable) as caught:
            self._check({"emulator-5554": "offline"}, None)

        message = str(caught.exception)
        self.assertIn("emulator-5554", message)
        self.assertIn("offline", message)

    def test_an_unauthorized_device_stops_the_run(self):
        with self.assertRaises(DeviceUnavailable) as caught:
            self._check({"emulator-5554": "unauthorized"}, None)

        self.assertIn("unauthorized", str(caught.exception))

    def test_a_device_still_booting_stops_the_run(self):
        """Its screen is not up yet, so every scenario would fail on it."""
        with self.assertRaises(DeviceUnavailable) as caught:
            self._check({"emulator-5554": "device"}, "emulator-5554", booted=False)

        self.assertIn("booting", str(caught.exception))

    def test_ios_is_left_to_its_own_preflight(self):
        """adb cannot answer for a simulator."""
        with patch.object(AdbControl, "attached_devices") as attached:
            ensure_device_ready(Platform.IOS, "iPhone 17 Pro Max")

        attached.assert_not_called()


class DeviceStateTests(unittest.TestCase):
    """Reading what adb reports, including the states it refuses to work with."""

    def _devices(self, stdout):
        with patch(
            "ronin.agent.device.adb_control.run_command",
            return_value=type("R", (), {"stdout": stdout})(),
        ):
            return AdbControl.attached_devices()

    def test_every_device_is_read_with_its_state(self):
        listing = (
            "List of devices attached\n"
            "emulator-5554\tdevice\n"
            "emulator-5556\toffline\n"
        )

        self.assertEqual(
            self._devices(listing),
            {"emulator-5554": "device", "emulator-5556": "offline"},
        )

    def test_an_empty_listing_reads_as_nothing_attached(self):
        self.assertEqual(self._devices("List of devices attached\n\n"), {})
