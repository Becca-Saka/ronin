"""Setting the device states a scenario asks for.

Every capability arrives as its own model, and every one is dispatched by an
explicit case. A capability this platform cannot apply raises, because the
alternative is a scenario that reads as having run offline while the device
stayed online.

The driver comes first and adb or simctl second. The session already holds a
driver, the driver speaks to both platforms in one vocabulary, and it reads
back what it sets. A command line is the fallback for what it has no method
for, and for a device the driver answers about but cannot change.
"""

import logging
import os
from typing import Callable, Optional

from ronin.agent.actuation.actions import (
    AccessibilityCapability,
    AppStateCapability,
    BatteryCapability,
    BiometricCapability,
    Capability,
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
from ronin.agent.device import capability_receipts
from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.capability_maps import (
    ANDROID_PERMISSIONS,
    APPIUM_NETWORK_TYPES,
    CONNECTIVITY,
    IOS_PRIVACY_SERVICES,
    TALKBACK,
)
from ronin.agent.device.driver_commands import DriverCommands
from ronin.agent.device.platform import Platform
from ronin.agent.device.receipts import Receipt
from ronin.agent.device.simctl_control import BATTERY_STATES, SimctlControl
from ronin.errors import ActionRejected, DeviceUnavailable

logger = logging.getLogger(__name__)

class CapabilityController:
    """Dispatches system-level state commands to the active device."""

    def __init__(self, device_control):
        """Initializes CapabilityController with active device controls.

        Args:
            device_control: Active DeviceControl instance.
        """
        self.device_control = device_control
        self.driver = device_control.driver
        self.platform = device_control.platform
        self.adb_control = getattr(device_control, "adb_control", None)
        self.commands = DriverCommands(device_control.driver)

    def get_udid_or_serial(self) -> str:
        """Resolves the current device serial number (Android) or UDID (iOS).

        Returns:
            String representing the device serial/UDID.
        """
        if self.platform == Platform.IOS:
            udid = self.driver.capabilities.get("udid")
            if not udid:
                udid = getattr(self.device_control, "device_name", None)
            return udid or ""
        else:
            if hasattr(self.device_control, "device_name"):
                name = self.device_control.device_name
                adb_runner = (
                    self.adb_control if self.adb_control is not None else AdbControl
                )
                resolved = adb_runner.resolve_emulator_serial(name)
                if resolved:
                    return resolved
                return name
            return "emulator-5554"

    def execute(self, capability: Capability) -> Optional[Receipt]:
        """Applies one capability, reads the state back, or refuses it.

        Args:
            capability: The device state to set.

        Returns:
            What the device reports about the state, or None where nothing can
            read it back.

        Raises:
            ActionRejected: The capability could not be applied here, or the
                device reports a state other than the one asked for. Nothing
                may read the device as being in the requested state.
        """
        logger.info("Executing capability control: %s", capability.kind.value)
        target = self.get_udid_or_serial()
        try:
            self._apply(target, capability)
        except ActionRejected:
            raise
        except Exception as exc:
            logger.error(
                "Failed to execute capability %s: %s", capability.kind.value, exc
            )
            raise ActionRejected(
                f"Capability {capability.kind.value} failed: {exc}"
            ) from exc

        receipt = capability_receipts.read_back(
            self.platform, target, capability, self.commands, self._known_package()
        )
        if receipt is None:
            return None
        if receipt.unreadable:
            logger.warning("Capability applied but not confirmed. %s", receipt)
        elif not receipt.confirmed:
            raise ActionRejected(f"The device did not take the state. {receipt}")
        else:
            logger.info("Capability confirmed. %s", receipt)
        return receipt

    def _apply(self, target: str, capability: Capability) -> None:
        """Routes one capability to the method that applies it.

        Matched on the model rather than looked up by name. The wildcard below
        refuses a capability with no case rather than ignoring it, and
        `test_every_capability_has_a_case_in_the_dispatch` is what catches one
        before a device does.
        """
        match capability:
            case LocationCapability():
                self._location(target, capability)
            case NetworkCapability():
                self._network(target, capability)
            case BatteryCapability():
                self._battery(target, capability)
            case TimezoneCapability():
                self._timezone(capability)
            case LocaleCapability():
                self._locale(target, capability)
            case PermissionCapability():
                self._permission(target, capability)
            case InterruptCapability():
                self._interrupt(target, capability)
            case BiometricCapability():
                self._biometric(capability)
            case ClipboardCapability():
                self._clipboard(capability)
            case DeeplinkCapability():
                self._deeplink(target, capability)
            case AppStateCapability():
                self._app_state(target, capability)
            case MediaInjectCapability():
                self._media_inject(target, capability)
            case AccessibilityCapability():
                self._accessibility(target, capability)
            case _:
                raise ActionRejected(
                    f"No handler for capability {capability.kind.value}"
                )

    def _refuse(self, capability: Capability, detail: str) -> ActionRejected:
        """The refusal to raise, naming the capability and this platform."""
        return ActionRejected(
            f"{capability.kind.value} cannot be applied on "
            f"{self.platform.value}: {detail}"
        )

    def _package(self, capability: Capability) -> str:
        """The app under test, which several capabilities act on.

        Raises:
            ActionRejected: No package is known, so the capability would be
                applied to nothing.
        """
        package = self.driver.capabilities.get("appPackage") or getattr(
            self.device_control, "package_id", None
        )
        if not package:
            raise self._refuse(capability, "the app package id is not known")
        return package

    def _or_else(self, doing: str, driver: Callable, command: Callable) -> None:
        """The driver's way, and the command line when the driver has no way.

        A driver missing a method and a device refusing one raise alike, so
        both land here. Falling back on either is right: the command line is
        the older path and works where the driver is not built for the device.

        When both fail, the driver's reason leads. It names the constraint, where the command line reports the symptom.
        Refused a call on a physical phone, the driver says "gsmCall is
        only available for emulators". `adb emu` only exits non-zero,
        with nothing to say.

        Raises:
            DeviceUnavailable: Neither way worked, carrying both reasons.
        """
        refusal = None
        try:
            driver()
            return
        except DeviceUnavailable as exc:
            refusal = exc
            logger.warning(
                "The driver could not %s (%s). Using the command line.", doing, exc
            )
        try:
            command()
        except DeviceUnavailable as failure:
            raise DeviceUnavailable(
                f"Could not {doing}. The driver refused: {refusal}. "
                f"The command line then failed: {failure}"
            ) from failure

    # --- Platform Handler Implementations ---

    def _location(self, target: str, capability: LocationCapability) -> None:
        if self.platform == Platform.IOS:
            self._or_else(
                "set the location",
                lambda: self.commands.set_ios_location(
                    capability.latitude, capability.longitude
                ),
                lambda: SimctlControl.set_location(
                    target, capability.latitude, capability.longitude
                ),
            )
        else:
            self._or_else(
                "set the location",
                lambda: self.commands.set_android_location(
                    capability.latitude, capability.longitude
                ),
                lambda: AdbControl.mock_location(
                    target, capability.latitude, capability.longitude
                ),
            )

    def _network(self, target: str, capability: NetworkCapability) -> None:
        if self.platform != Platform.ANDROID:
            raise self._refuse(
                capability,
                "the simulator shares the host network connection, and the "
                "condition inducers that would change it need a real device",
            )

        radios = CONNECTIVITY[capability.mode]
        self._or_else(
            f"set connectivity to {capability.mode}",
            lambda: self.commands.set_connectivity(**radios),
            lambda: self._network_by_command(target, capability.mode),
        )

    def _network_by_command(self, target: str, mode: str) -> None:
        """Airplane mode through adb, and the rest through Appium's bitmask."""
        if mode in ("offline", "airplane", "online"):
            AdbControl.set_airplane_mode(target, mode != "online")
            return
        self.driver.set_network_connection(APPIUM_NETWORK_TYPES[mode])

    def _battery(self, target: str, capability: BatteryCapability) -> None:
        if self.platform == Platform.IOS:
            state = BATTERY_STATES[0] if capability.charging else BATTERY_STATES[2]
            SimctlControl.override_battery(target, capability.level, state)
        else:
            AdbControl.set_battery_status(
                target, capability.level, capability.charging
            )

    def _timezone(self, capability: TimezoneCapability) -> None:
        """Refused on both platforms, because neither offers a path that works.

        On Android the system zone is `persist.sys.timezone`, which the shell
        user cannot set. A write to `settings global timezone` changes
        nothing the device reads. Reading that setting back confirms
        the write while the clock stays where it was. simctl sets no zone at all.
        """
        raise self._refuse(
            capability,
            "no supported path sets the system zone, and the one that looks "
            "like it does is not read by the device",
        )

    def _locale(self, target: str, capability: LocaleCapability) -> None:
        if self.platform == Platform.IOS:
            self.commands.configure_ios_localization(
                capability.language, capability.country
            )
            return
        package = self._package(capability)
        locale = (
            f"{capability.language}-{capability.country}"
            if capability.country
            else capability.language
        )
        AdbControl.set_app_locale(target, package, locale)

    def _permission(self, target: str, capability: PermissionCapability) -> None:
        package = self._package(capability)

        if self.platform == Platform.IOS:
            service = IOS_PRIVACY_SERVICES.get(capability.permission)
            if not service:
                raise self._refuse(
                    capability,
                    f"'{capability.permission}' names no simctl privacy service",
                )
            action = "grant" if capability.grant else "revoke"
            self._or_else(
                f"{action} {service}",
                lambda: self.commands.set_ios_permission(
                    package, service, capability.grant
                ),
                lambda: SimctlControl.set_privacy(target, action, service, package),
            )
            return

        permission = ANDROID_PERMISSIONS.get(capability.permission)
        if not permission:
            raise self._refuse(
                capability, f"'{capability.permission}' names no Android permission"
            )
        self._or_else(
            f"change {permission}",
            lambda: self.commands.change_android_permissions(
                package, [permission], capability.grant
            ),
            lambda: AdbControl.set_app_permission(
                target, package, permission, capability.grant
            ),
        )

    def _interrupt(self, target: str, capability: InterruptCapability) -> None:
        if capability.type == "low_memory":
            if self.platform == Platform.IOS:
                self._or_else(
                    "post a memory warning",
                    self.commands.send_ios_memory_warning,
                    lambda: SimctlControl.notify_low_memory(target),
                )
                return
            package = self._package(capability)
            self._or_else(
                "send a trim-memory signal",
                lambda: self.commands.send_android_trim_memory(package),
                lambda: self._trim_memory_by_pid(target, package, capability),
            )
            return

        if self.platform != Platform.ANDROID:
            raise self._refuse(
                capability, "simctl cannot place a call into the simulator"
            )
        self._or_else(
            "place a call",
            lambda: self.commands.gsm_call(capability.phone),
            lambda: AdbControl.simulate_incoming_call(target, capability.phone),
        )

    def _trim_memory_by_pid(
        self, target: str, package: str, capability: Capability
    ) -> None:
        """The adb path, which needs the pid the driver did not."""
        pid = AdbControl.get_app_pid(target, package)
        if not pid:
            raise self._refuse(
                capability, f"{package} is not running, so it has no pid"
            )
        AdbControl.simulate_low_memory(target, pid)

    def _biometric(self, capability: BiometricCapability) -> None:
        if self.platform == Platform.IOS:
            enroll = capability.action in ("enroll", "match")
            self.commands.change("enrollBiometric", {"isEnabled": enroll})
            if capability.action in ("match", "unmatch"):
                self.commands.change(
                    "sendBiometricMatch",
                    {
                        "type": capability.type,
                        "match": capability.action == "match",
                    },
                )
        else:
            self._or_else(
                "touch the fingerprint sensor",
                lambda: self.commands.fingerprint(capability.finger_id),
                lambda: self.driver.finger_print(capability.finger_id),
            )

    def _clipboard(self, capability: ClipboardCapability) -> None:
        logger.info("Setting device clipboard")
        self.commands.set_clipboard(capability.text)

    def _deeplink(self, target: str, capability: DeeplinkCapability) -> None:
        if self.platform == Platform.IOS:
            self._or_else(
                "open a URL",
                lambda: self.commands.open_deeplink(capability.url),
                lambda: SimctlControl.open_url(target, capability.url),
            )
            return
        package = self._package(capability)
        self._or_else(
            "open a deep link",
            lambda: self.commands.open_deeplink(capability.url),
            lambda: AdbControl.open_deeplink(target, capability.url, package),
        )

    def _app_state(self, target: str, capability: AppStateCapability) -> None:
        package = self._package(capability)

        if capability.action == "force_stop":
            self._or_else(
                "terminate the app",
                lambda: self.commands.terminate_app(package),
                lambda: self._terminate_by_command(target, package),
            )
            return

        if self.platform != Platform.ANDROID:
            self._or_else(
                "clear the app container",
                lambda: self.commands.clear_app(package),
                lambda: (_ for _ in ()).throw(
                    self._refuse(capability, "simctl cannot clear an installed app")
                ),
            )
            return
        AdbControl.clear_app_data(target, package)

    def _terminate_by_command(self, target: str, package: str) -> None:
        if self.platform == Platform.IOS:
            SimctlControl.terminate(target, package)
        else:
            AdbControl.force_stop_app(target, package)

    def _media_inject(self, target: str, capability: MediaInjectCapability) -> None:
        if not os.path.exists(capability.local_path):
            raise self._refuse(capability, f"no file at '{capability.local_path}'")

        if self.platform == Platform.IOS:
            SimctlControl.add_media(target, capability.local_path)
        else:
            destination = f"/sdcard/Pictures/{os.path.basename(capability.local_path)}"
            AdbControl.push_file(target, capability.local_path, destination)
            AdbControl.scan_media_file(target, destination)

    def _accessibility(self, target: str, capability: AccessibilityCapability) -> None:
        if self.platform == Platform.IOS:
            self.commands.set_ios_voice_over(capability.enabled)
            return
        AdbControl.set_accessibility_service(target, TALKBACK, capability.enabled)

    def _known_package(self) -> Optional[str]:
        """The app under test, or None. Unlike `_package`, this never raises.

        A receipt is read after the capability was applied, so a missing
        package there is not a reason to fail the capability a second time.
        """
        return self.driver.capabilities.get("appPackage") or getattr(
            self.device_control, "package_id", None
        )
