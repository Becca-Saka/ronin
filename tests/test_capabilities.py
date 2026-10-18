"""Setting a device state, and refusing to pretend it was set.

Two things are under test. A capability that cannot be applied raises rather
than logging a warning and returning, because a scenario about being offline
must not report a pass while the device stayed online. And a capability that
was applied is read back, because a command exiting 0 says it ran and not that
the state changed.

The driver is asked first and the command line second, so each case says which
one it expects, and the fallback is exercised by a driver that lacks the method.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from ronin.agent.actuation.actions import (
    AccessibilityCapability,
    AppStateCapability,
    BatteryCapability,
    BiometricCapability,
    CapabilityKind,
    ClipboardCapability,
    DeeplinkCapability,
    InterruptCapability,
    LocaleCapability,
    LocationCapability,
    MediaInjectCapability,
    NetworkCapability,
    PermissionCapability,
    TimezoneCapability,
)
from ronin.agent.device import capability_controller, capability_receipts
from ronin.agent.device.capability_controller import CapabilityController
from ronin.agent.device.platform import Platform
from ronin.errors import ActionRejected

SERIAL = "emulator-5554"
PACKAGE = "com.example.app"

# Every read-back the command-line fallback can ask for. Silenced by default,
# so a dispatch case is about the command that was sent and nothing else.
READERS = (
    "airplane_mode",
    "timezone",
    "app_locale",
    "battery_level",
    "battery_state",
    "permission_granted",
    "accessibility_enabled",
    "app_running",
)

# One instance of every capability, so a capability added to the union without
# a case in the dispatch fails here rather than at run time on a device.
EVERY_CAPABILITY = {
    CapabilityKind.LOCATION: LocationCapability(latitude=1.0, longitude=2.0),
    CapabilityKind.NETWORK: NetworkCapability(mode="offline"),
    CapabilityKind.BATTERY: BatteryCapability(level=50),
    CapabilityKind.TIMEZONE: TimezoneCapability(timezone="Europe/London"),
    CapabilityKind.LOCALE: LocaleCapability(language="fr", country="FR"),
    CapabilityKind.PERMISSION: PermissionCapability(permission="camera", grant=True),
    CapabilityKind.INTERRUPT: InterruptCapability(type="low_memory"),
    CapabilityKind.BIOMETRIC: BiometricCapability(action="match"),
    CapabilityKind.CLIPBOARD: ClipboardCapability(text="hello"),
    CapabilityKind.DEEPLINK: DeeplinkCapability(url="app://cart"),
    CapabilityKind.APP_STATE: AppStateCapability(action="force_stop"),
    CapabilityKind.MEDIA_INJECT: MediaInjectCapability(local_path=__file__),
    CapabilityKind.ACCESSIBILITY: AccessibilityCapability(enabled=True),
}


class FakeDriver:
    """A driver that records the `mobile:` methods it was asked for.

    `answers` is what a read returns. `without` names methods this driver does
    not have, which is how a real driver behaves on a platform or version that
    never implemented one, and is what sends a capability to the command line.
    """

    def __init__(self, answers=None, without=(), package=PACKAGE):
        self.capabilities = {"appPackage": package, "udid": SERIAL}
        self.calls = []
        self.answers = answers or {}
        self.without = set(without)
        self.set_network_connection = mock.Mock()
        self.finger_print = mock.Mock()

    def execute_script(self, script, params=None):
        method = script.removeprefix("mobile: ")
        self.calls.append((method, params or {}))
        if method in self.without:
            raise RuntimeError(f"Unknown mobile command '{method}'")
        return self.answers.get(method)

    def asked(self, method):
        """The params one method was called with, or None if it never was."""
        for name, params in self.calls:
            if name == method:
                return params
        return None


def controller(platform=Platform.ANDROID, package=PACKAGE, driver=None):
    driver = driver or FakeDriver(package=package)
    device = SimpleNamespace(
        driver=driver,
        platform=platform,
        device_name=SERIAL,
        serial=SERIAL,
        package_id=package,
        adb_control=SimpleNamespace(resolve_emulator_serial=lambda name: SERIAL),
    )
    return CapabilityController(device)


def silent_reads():
    """A device that answers no command-line state question."""
    patched = mock.patch.object(capability_receipts, "receipts")
    reads = patched.start()
    for name in READERS:
        getattr(reads, name).return_value = None
    return patched, reads


class DriverFirstTests(unittest.TestCase):
    """The session already holds a driver, so it is asked before adb."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)
        self.driver = FakeDriver()
        self.control = controller(driver=self.driver)

    def _run(self, capability):
        with mock.patch.object(capability_controller, "AdbControl") as adb, \
             mock.patch.object(capability_controller, "SimctlControl") as simctl:
            self.control.execute(capability)
        return adb, simctl

    def test_going_offline_turns_off_both_radios(self):
        """The driver takes the radios apart, where adb has only airplane mode."""
        adb, _ = self._run(NetworkCapability(mode="offline"))

        self.assertEqual(
            self.driver.asked("setConnectivity"),
            {"wifi": False, "data": False, "airplaneMode": True},
        )
        adb.set_airplane_mode.assert_not_called()

    def test_wifi_only_leaves_data_off_without_a_bitmask(self):
        self._run(NetworkCapability(mode="wifi_only"))

        self.assertEqual(
            self.driver.asked("setConnectivity"),
            {"wifi": True, "data": False, "airplaneMode": False},
        )
        self.driver.set_network_connection.assert_not_called()

    def test_a_permission_goes_through_the_driver(self):
        adb, _ = self._run(PermissionCapability(permission="camera", grant=True))

        self.assertEqual(
            self.driver.asked("changePermissions"),
            {
                "permissions": ["android.permission.CAMERA"],
                "appPackage": PACKAGE,
                "action": "grant",
            },
        )
        adb.set_app_permission.assert_not_called()

    def test_a_deep_link_is_one_call_on_either_platform(self):
        self._run(DeeplinkCapability(url="app://cart"))

        self.assertEqual(self.driver.asked("deepLink"), {"url": "app://cart"})

    def test_stopping_the_app_goes_through_the_driver(self):
        adb, _ = self._run(AppStateCapability(action="force_stop"))

        self.assertIsNotNone(self.driver.asked("terminateApp"))
        adb.force_stop_app.assert_not_called()

    def test_low_memory_needs_no_pid_through_the_driver(self):
        """The adb path needs a pid, so it refuses a stopped app. This does not."""
        adb, _ = self._run(InterruptCapability(type="low_memory"))

        self.assertEqual(
            self.driver.asked("sendTrimMemory"),
            {"pkg": PACKAGE, "level": "RUNNING_CRITICAL"},
        )
        adb.get_app_pid.assert_not_called()

    def test_every_capability_has_a_case_in_the_dispatch(self):
        """A model in the union with no case would fall to the refusal."""
        self.assertEqual(set(EVERY_CAPABILITY), set(CapabilityKind))

        for kind, capability in EVERY_CAPABILITY.items():
            if kind is CapabilityKind.TIMEZONE:
                continue  # Refused on both platforms; covered in RefusalTests.
            with self.subTest(kind=kind.value):
                self._run(capability)


class FallbackTests(unittest.TestCase):
    """A driver without the method is not a reason to skip the capability."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _run(self, capability, without, platform=Platform.ANDROID):
        driver = FakeDriver(without=without)
        control = controller(platform=platform, driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb, \
             mock.patch.object(capability_controller, "SimctlControl") as simctl:
            control.execute(capability)
        return adb, simctl, driver

    def test_connectivity_falls_back_to_airplane_mode(self):
        adb, _, driver = self._run(
            NetworkCapability(mode="offline"), without=["setConnectivity"]
        )

        self.assertIsNotNone(driver.asked("setConnectivity"))
        adb.set_airplane_mode.assert_called_once_with(SERIAL, True)

    def test_a_mode_adb_cannot_set_falls_back_to_the_appium_bitmask(self):
        _, _, driver = self._run(
            NetworkCapability(mode="wifi_only"), without=["setConnectivity"]
        )

        driver.set_network_connection.assert_called_once_with(2)

    def test_a_permission_falls_back_to_pm(self):
        adb, _, _ = self._run(
            PermissionCapability(permission="camera", grant=False),
            without=["changePermissions"],
        )

        adb.set_app_permission.assert_called_once_with(
            SERIAL, PACKAGE, "android.permission.CAMERA", False
        )

    def test_low_memory_falls_back_to_the_pid_path(self):
        adb, _, _ = self._run(
            InterruptCapability(type="low_memory"), without=["sendTrimMemory"]
        )

        adb.get_app_pid.assert_called_once_with(SERIAL, PACKAGE)

    def test_a_stopped_app_with_no_pid_is_refused_on_the_fallback(self):
        driver = FakeDriver(without=["sendTrimMemory"])
        control = controller(driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            adb.get_app_pid.return_value = ""
            with self.assertRaises(ActionRejected) as raised:
                control.execute(InterruptCapability(type="low_memory"))

        self.assertIn("pid", str(raised.exception))

    def test_a_simulator_location_falls_back_to_simctl(self):
        _, simctl, _ = self._run(
            LocationCapability(latitude=1.0, longitude=2.0),
            without=["setSimulatedLocation"],
            platform=Platform.IOS,
        )

        simctl.set_location.assert_called_once_with(SERIAL, 1.0, 2.0)


class RefusalTests(unittest.TestCase):
    """A capability this platform cannot apply raises, and names both."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _refuse(self, capability, platform=Platform.IOS, package=PACKAGE):
        control = controller(platform, package)
        with mock.patch.object(capability_controller, "AdbControl"), \
             mock.patch.object(capability_controller, "SimctlControl"), \
             self.assertRaises(ActionRejected) as raised:
            control.execute(capability)
        return str(raised.exception)

    def test_the_refusal_names_the_capability_and_the_platform(self):
        message = self._refuse(NetworkCapability(mode="offline"))

        self.assertIn("network", message)
        self.assertIn("ios", message)

    def test_a_timezone_no_path_can_set_is_refused_on_android(self):
        """The setting that looks like it works is not the one the device reads."""
        message = self._refuse(
            TimezoneCapability(timezone="UTC"), platform=Platform.ANDROID
        )

        self.assertIn("timezone", message)

    def test_a_timezone_is_refused_on_ios_too(self):
        self.assertIn("timezone", self._refuse(TimezoneCapability(timezone="UTC")))

    def test_a_call_the_simulator_cannot_place_is_refused(self):
        self.assertIn(
            "interrupt", self._refuse(InterruptCapability(type="incoming_call"))
        )

    def test_a_capability_needing_the_app_is_refused_without_one(self):
        message = self._refuse(
            PermissionCapability(permission="camera", grant=True),
            platform=Platform.ANDROID,
            package=None,
        )

        self.assertIn("package", message)

    def test_a_file_that_is_not_there_is_refused(self):
        message = self._refuse(
            MediaInjectCapability(local_path="/no/such/file.png"),
            platform=Platform.ANDROID,
        )

        self.assertIn("/no/such/file.png", message)

    def test_a_permission_the_platform_does_not_name_is_refused(self):
        """simctl has no notification service, and 'all' would grant the rest."""
        message = self._refuse(
            PermissionCapability(permission="notifications", grant=True)
        )

        self.assertIn("notifications", message)

    def test_a_failing_command_is_reported_as_a_refusal(self):
        """Nothing downstream may read the device as having changed."""
        control = controller(Platform.ANDROID)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            adb.set_battery_status.side_effect = RuntimeError("adb: device offline")
            with self.assertRaises(ActionRejected) as raised:
                control.execute(BatteryCapability(level=50))

        self.assertIn("device offline", str(raised.exception))


class VocabularyTests(unittest.TestCase):
    """What the planner is offered comes from the models themselves."""

    def test_the_prompt_lists_every_capability(self):
        from ronin.agent.actuation.actions import capability_vocabulary

        listing = capability_vocabulary()

        for kind in CapabilityKind:
            self.assertIn(kind.value, listing)

    def test_a_capability_s_choices_reach_the_prompt(self):
        """A mode the model cannot see is a mode it will not ask for."""
        from ronin.agent.actuation.actions import capability_vocabulary

        listing = capability_vocabulary()

        for mode in ("offline", "online", "wifi_only", "data_only", "airplane"):
            self.assertIn(mode, listing)

    def test_an_action_carries_a_typed_capability_not_a_dict(self):
        from ronin.agent.actuation.actions import CapabilityAction

        action = CapabilityAction.model_validate(
            {
                "type": "capability",
                "capability": {"kind": "network", "mode": "offline"},
                "reason": "the scenario is about losing connectivity",
            }
        )

        self.assertIsInstance(action.capability, NetworkCapability)
        self.assertEqual(action.capability.mode, "offline")

    def test_a_mode_no_capability_offers_is_refused(self):
        from pydantic import ValidationError

        from ronin.agent.actuation.actions import CapabilityAction

        with self.assertRaises(ValidationError):
            CapabilityAction.model_validate(
                {
                    "type": "capability",
                    "capability": {"kind": "network", "mode": "2g"},
                    "reason": "throttle",
                }
            )


class DeviceTruthTests(unittest.TestCase):
    """Where a tool's real contract differs from the obvious reading of it.

    A write can land somewhere the device does not read, a command can exit 0
    having done nothing, and an argument can be one the tool does not accept.
    A fixture agrees with whichever reading it was written from, so each of
    these pins the contract instead.
    """

    def test_the_timezone_is_read_from_the_property_the_system_keeps(self):
        """`settings global timezone` and the property disagree.

        Writing the setting leaves the property alone, so reading the setting
        back confirms the write and says nothing about the device.
        """
        from ronin.agent.device import receipts as module

        with mock.patch(
            "ronin.agent.device.adb_control.AdbControl.ask",
            return_value="Europe/London",
        ) as ask:
            self.assertEqual(module.timezone(SERIAL), "Europe/London")

        self.assertIn("persist.sys.timezone", ask.call_args[0])

    def test_a_permission_the_app_never_asked_for_is_not_merely_unreadable(self):
        """`pm grant` exits 0 for one of these and grants nothing."""
        from ronin.agent.device import receipts as module

        dump = "    requested permissions:\n      android.permission.INTERNET\n"
        with mock.patch(
            "ronin.agent.device.adb_control.AdbControl.ask", return_value=dump
        ):
            answer = module.permission_granted(
                SERIAL, PACKAGE, "android.permission.CAMERA"
            )

        self.assertEqual(answer, module.NOT_REQUESTED)

    def test_a_discharging_battery_uses_a_word_simctl_accepts(self):
        from ronin.agent.device.simctl_control import BATTERY_STATES

        control = controller(Platform.IOS)
        with mock.patch.object(capability_controller, "SimctlControl") as simctl:
            control._battery(SERIAL, BatteryCapability(level=42, charging=False))

        state = simctl.override_battery.call_args[0][2]
        self.assertEqual(state, "discharging")
        self.assertIn(state, BATTERY_STATES)

    def test_a_battery_state_simctl_rejects_never_reaches_it(self):
        """simctl answers an invalid state with a usage message and exit 117."""
        from ronin.agent.device.simctl_control import SimctlControl
        from ronin.errors import DeviceUnavailable

        with mock.patch(
            "ronin.agent.device.simctl_control.run_command"
        ) as run, self.assertRaises(DeviceUnavailable):
            SimctlControl.override_battery(SERIAL, 42, "discharged")

        run.assert_not_called()

    def test_granting_notifications_does_not_grant_everything(self):
        """simctl's 'all' service would grant every other permission too."""
        services = capability_controller.IOS_PRIVACY_SERVICES

        self.assertNotIn("all", services.values())
        self.assertNotIn("notifications", services)


class BothWaysFailedTests(unittest.TestCase):
    """When the driver and the command line both fail, which reason survives.

    The driver names the constraint and the command line reports the symptom.
    Refused a call on a physical phone, the driver says the method is for
    emulators only, and `adb emu` exits non-zero with nothing to say. Losing
    the first message leaves the one that explains nothing.
    """

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def test_the_driver_s_reason_leads_and_both_are_kept(self):
        from ronin.errors import DeviceUnavailable

        driver = FakeDriver(without=["gsmCall"])
        control = controller(driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            adb.simulate_incoming_call.side_effect = DeviceUnavailable(
                "Failed to simulate incoming call: adb -s X emu gsm call 555"
            )
            with self.assertRaises(ActionRejected) as raised:
                control.execute(InterruptCapability(type="incoming_call"))

        message = str(raised.exception)
        self.assertIn("gsmCall", message)
        self.assertLess(message.index("driver refused"), message.index("then failed"))

    def test_a_command_line_that_works_is_not_an_error(self):
        driver = FakeDriver(without=["gsmCall"])
        control = controller(driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            control.execute(InterruptCapability(type="incoming_call"))

        adb.simulate_incoming_call.assert_called_once()
