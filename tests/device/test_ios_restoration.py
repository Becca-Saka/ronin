"""Handing an iOS device back the way a scenario found it.

Every iOS state is read through the driver, so a scenario reads only the
states its own declared capabilities could change. Three of them outlived
the scenario that set them until this: a simulated location, a battery
override, and a permission decision.
"""

import unittest
from unittest import mock

from fakes import IosDriver, ios_device

from ronin.agent.actuation.actions import (
    AccessibilityCapability,
    LocationCapability,
    PermissionCapability,
)
from ronin.agent.device import restoration
from ronin.errors import DeviceUnavailable

SCREEN_READER = AccessibilityCapability(enabled=True)


class IosRestorationTests(unittest.TestCase):
    """The state an iOS capability changes, put back the same way.

    The screen reader is the one state the driver both reads and writes. It
    changes what every scenario after it sees on arrival, and none of those
    declared it.
    """

    def setUp(self):
        self.reads = mock.patch.object(restoration, "receipts").start()
        self.adb = mock.patch.object(restoration, "AdbControl").start()
        self.addCleanup(mock.patch.stopall)

    def test_an_ios_device_is_not_read_through_adb(self):
        """adb answers for no simulator, so a value it returned would be another device's."""
        with restoration.preserved(
            ios_device(IosDriver(voice_over=False)), capabilities=[SCREEN_READER]
        ):
            pass

        self.reads.airplane_mode.assert_not_called()
        self.adb.set_airplane_mode.assert_not_called()

    def test_a_screen_reader_a_scenario_switched_on_is_switched_off(self):
        driver = IosDriver(voice_over=False)

        with restoration.preserved(ios_device(driver), capabilities=[SCREEN_READER]):
            driver.state["voice_over"] = True

        self.assertEqual(driver.sent, [("disableVoiceOver", {})])

    def test_a_screen_reader_a_scenario_switched_off_is_switched_on(self):
        driver = IosDriver(voice_over=True)

        with restoration.preserved(ios_device(driver), capabilities=[SCREEN_READER]):
            driver.state["voice_over"] = False

        self.assertEqual(driver.sent, [("enableVoiceOver", {})])

    def test_a_screen_reader_left_alone_is_not_touched(self):
        driver = IosDriver(voice_over=False)

        with restoration.preserved(ios_device(driver), capabilities=[SCREEN_READER]):
            pass

        self.assertEqual(driver.sent, [])

    def test_a_state_the_driver_would_not_answer_is_not_guessed_at(self):
        """VoiceOver automation arrived in iOS 27, and an older driver refuses the read."""
        driver = IosDriver(voice_over=None)

        with restoration.preserved(ios_device(driver), capabilities=[SCREEN_READER]) as found:
            driver.state["voice_over"] = True

        self.assertEqual(driver.sent, [])
        self.assertEqual(found.describe(), "nothing")

    def test_a_state_the_driver_would_not_answer_is_not_read_again(self):
        """The second read could change nothing, and costs a call and a warning."""
        driver = IosDriver(voice_over=None)

        with restoration.preserved(ios_device(driver), capabilities=[SCREEN_READER]):
            pass

        self.assertEqual(driver.reads, 1)

    def test_a_scenario_that_raised_still_hands_the_device_back(self):
        driver = IosDriver(voice_over=False)

        with self.assertRaises(DeviceUnavailable), restoration.preserved(
            ios_device(driver), capabilities=[SCREEN_READER]
        ):
            driver.state["voice_over"] = True
            raise DeviceUnavailable("the driver went away")

        self.assertEqual(driver.sent, [("disableVoiceOver", {})])

    def test_an_interrupted_scenario_still_hands_the_device_back(self):
        driver = IosDriver(voice_over=False)

        with self.assertRaises(KeyboardInterrupt), restoration.preserved(
            ios_device(driver), capabilities=[SCREEN_READER]
        ):
            driver.state["voice_over"] = True
            raise KeyboardInterrupt

        self.assertEqual(driver.sent, [("disableVoiceOver", {})])

    def test_a_restore_that_fails_does_not_stop_the_scenario_ending(self):
        driver = IosDriver(voice_over=False, refuse=("disableVoiceOver",))

        with restoration.preserved(ios_device(driver), capabilities=[SCREEN_READER]):
            driver.state["voice_over"] = True

        self.assertEqual(driver.asked("disableVoiceOver"), {})


class IosLeakTests(unittest.TestCase):
    """The iOS states that outlived the scenario that set them.

    Appium's own warning is explicit about the location: a simulated one
    stays until something resets it or the device restarts. It decided what
    every later scenario was testing, and none of those declared it.

    A battery override lasts as long and is not restored, because iOS battery
    is refused. A privacy decision is restored, and `PermissionRestorationTests`
    covers it.
    """

    def _preserved(self, driver, capabilities):
        return restoration.preserved(ios_device(driver), capabilities=capabilities)

    def test_a_location_a_scenario_simulated_is_taken_away_again(self):
        """A reset is the only way back to a device nobody had placed."""
        driver = IosDriver(location=None)

        with self._preserved(driver, [LocationCapability(latitude=1.0, longitude=2.0)]):
            driver.state["location"] = (51.5, -0.12)

        self.assertEqual(driver.asked("resetSimulatedLocation"), {})

    def test_a_location_the_scenario_moved_is_put_back_where_it_was(self):
        driver = IosDriver(location=(35.68, 139.69))

        with self._preserved(driver, [LocationCapability(latitude=1.0, longitude=2.0)]):
            driver.state["location"] = (51.5, -0.12)

        self.assertEqual(
            driver.asked("setSimulatedLocation"),
            {"latitude": 35.68, "longitude": 139.69},
        )

    def test_a_location_left_alone_is_not_written_back(self):
        driver = IosDriver(location=(35.68, 139.69))

        with self._preserved(driver, [LocationCapability(latitude=1.0, longitude=2.0)]):
            pass

        self.assertEqual(driver.sent, [])

    def test_a_device_that_was_never_placed_and_still_is_not_is_left_alone(self):
        """Nothing was simulated, and a reset would be a write nobody needs."""
        driver = IosDriver(location=None)

        with self._preserved(driver, [LocationCapability(latitude=1.0, longitude=2.0)]):
            pass

        self.assertEqual(driver.sent, [])






    def test_a_location_the_driver_would_not_answer_is_not_guessed_at(self):
        """An older driver refuses the read, which is not a device at nowhere."""
        driver = IosDriver(location=None, refuse=("getSimulatedLocation",))

        with self._preserved(driver, [LocationCapability(latitude=1.0, longitude=2.0)]) as found:
            pass

        self.assertIsNone(found.simulated_location)
        self.assertEqual(driver.sent, [])


    def test_a_scenario_that_declared_nothing_reads_nothing(self):
        """Four driver calls per scenario buy nothing when none can move."""
        driver = IosDriver(voice_over=False, location=(1.0, 2.0))

        with self._preserved(driver, []) as found:
            pass

        self.assertEqual(driver.reads, 0)
        self.assertEqual(found.describe(), "nothing")

    def test_only_the_declared_state_is_read(self):
        driver = IosDriver(voice_over=False, location=(1.0, 2.0))

        with self._preserved(driver, [LocationCapability(latitude=1.0, longitude=2.0)]) as found:
            # Read inside the scenario: the restore reads again on the way
            # out, to see whether the state moved.
            self.assertEqual(driver.reads, 1)

        self.assertIsNone(found.voice_over)
        self.assertEqual(found.simulated_location, "1.0,2.0")


class PermissionRestorationTests(unittest.TestCase):
    """A privacy decision outlives the install that received it.

    `simctl privacy` writes three states and reports none of them, so the
    read is the only evidence and `reset` is the only way back to a decision
    nobody has made yet. Written back as a boolean, an `unset` permission
    became an explicit denial for every scenario after it.
    """

    def setUp(self):
        self.simctl = mock.patch.object(restoration, "SimctlControl").start()
        self.addCleanup(mock.patch.stopall)

    def _preserved(self, driver):
        return restoration.preserved(
            ios_device(driver), capabilities=[PermissionCapability(permission='camera', grant=True)]
        )

    def _written(self):
        """The (action, service) pairs simctl was asked for."""
        return [
            (call.args[1], call.args[2])
            for call in self.simctl.set_privacy.call_args_list
        ]

    def test_a_decision_nobody_had_made_is_reset_rather_than_denied(self):
        driver = IosDriver(granted={"camera": "unset"})

        with self._preserved(driver):
            driver.state["granted"]["camera"] = "yes"

        self.assertEqual(self._written(), [("reset", "camera")])

    def test_a_permission_the_scenario_granted_is_taken_back(self):
        driver = IosDriver(granted={"camera": "no"})

        with self._preserved(driver):
            driver.state["granted"]["camera"] = "yes"

        self.assertEqual(self._written(), [("revoke", "camera")])

    def test_a_permission_the_scenario_revoked_is_granted_again(self):
        driver = IosDriver(granted={"camera": "yes"})

        with self._preserved(driver):
            driver.state["granted"]["camera"] = "no"

        self.assertEqual(self._written(), [("grant", "camera")])

    def test_a_permission_left_alone_is_not_rewritten(self):
        driver = IosDriver(granted={"camera": "yes", "photos": "unset"})

        with self._preserved(driver):
            pass

        self.assertEqual(self._written(), [])

    def test_a_limited_grant_is_not_written_back_as_anything(self):
        """Nothing sets a partial grant, so nothing here pretends to."""
        driver = IosDriver(granted={"photos": "limited"})

        with self._preserved(driver):
            driver.state["granted"]["photos"] = "yes"

        self.assertEqual(self._written(), [])

    def test_a_service_the_host_could_not_read_is_not_guessed_at(self):
        """Without applesimutils the read answers nothing, and nothing is written."""
        driver = IosDriver(granted={})

        with self._preserved(driver) as found:
            driver.state["granted"]["camera"] = "yes"

        self.assertEqual(found.permissions, {})
        self.assertEqual(self._written(), [])

    def test_only_the_service_the_steps_ask_for_is_read(self):
        """One step changes one service, and the others cost a call each.

        Read by kind, a camera scenario also asked about location, which iOS
        refuses, so every camera run carried a warning about a service it
        never touched.
        """
        driver = IosDriver(granted={"camera": "yes", "photos": "yes"})

        with self._preserved(driver) as found:
            # Read inside the scenario: the restore reads again on the way
            # out, to see whether the decision moved.
            self.assertEqual(driver.reads, 1)

        self.assertEqual(set(found.permissions), {"camera"})

    def test_two_steps_are_both_read(self):
        driver = IosDriver(granted={"camera": "yes", "photos": "unset"})
        capabilities = [
            PermissionCapability(permission="camera", grant=True),
            PermissionCapability(permission="photos", grant=False),
        ]

        with restoration.preserved(
            ios_device(driver), capabilities=capabilities
        ) as found:
            pass

        self.assertEqual(set(found.permissions), {"camera", "photos"})

    def test_a_location_step_reads_no_privacy_service(self):
        """iOS refuses a location decision, so there is nothing to put back."""
        driver = IosDriver(granted={"camera": "yes"})
        capabilities = [PermissionCapability(permission="location", grant=True)]

        with restoration.preserved(
            ios_device(driver), capabilities=capabilities
        ) as found:
            pass

        self.assertIsNone(found.permissions)
        self.assertEqual(driver.reads, 0)
