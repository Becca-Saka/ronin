import logging
import os
import subprocess
import re
from ronin.agent.device.platform import Platform
from ronin.agent.actuation.actions import CapabilityType
from ronin.utils.subprocess_runner import run_command
from ronin.agent.device.adb_control import AdbControl

logger = logging.getLogger(__name__)


class CapabilityController:
    """Dispatches system-level state commands (mock location, permission grants, battery overrides) to active devices."""

    def __init__(self, device_control):
        """Initializes CapabilityController with active device controls.

        Args:
            device_control: Active DeviceControl instance.
        """
        self.device_control = device_control
        self.driver = device_control.driver
        self.platform = device_control.platform
        self.adb_control = getattr(device_control, "adb_control", None)

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

    def execute(self, capability: CapabilityType, parameters: dict):
        """Routes capability actions to the platform-specific implementation methods.

        Args:
            capability: Enum value representing the CapabilityType.
            parameters: Key-value parameters defining the target values (e.g. coordinates or mode).
        """
        logger.info(
            "Executing capability control: %s with params %s",
            capability.value,
            parameters,
        )
        udid_or_serial = self.get_udid_or_serial()

        method_name = f"_handle_{capability.value}"
        if hasattr(self, method_name):
            try:
                getattr(self, method_name)(udid_or_serial, parameters)
            except Exception as e:
                logger.error("Failed to execute capability %s: %s", capability.value, e)
                raise RuntimeError(f"Capability {capability.value} failed: {e}")
        else:
            logger.warning("Unsupported capability type: %s", capability.value)

    # --- Platform Handler Implementations ---

    def _handle_location(self, udid_or_serial: str, params: dict):
        lat = params.get("latitude")
        lng = params.get("longitude")
        if lat is None or lng is None:
            raise ValueError("Location requires 'latitude' and 'longitude'")

        if self.platform == Platform.IOS:
            cmd = ["xcrun", "simctl", "location", udid_or_serial, "set", f"{lat},{lng}"]
            logger.info("iOS location mock: %s", " ".join(cmd))
            run_command(cmd, check=True)
        else:
            logger.info("Android location mock: %s, %s", lat, lng)
            AdbControl.mock_location(udid_or_serial, lat, lng)

    def _handle_network(self, udid_or_serial: str, params: dict):
        mode = str(params.get("mode", "online")).lower()
        if self.platform == Platform.ANDROID:
            success = False
            if mode in ("offline", "airplane"):
                try:
                    logger.info("Enabling airplane mode via ADB connectivity service")
                    AdbControl.set_airplane_mode(udid_or_serial, True)
                    success = True
                except Exception as e:
                    logger.warning(
                        "ADB connectivity control failed: %s. Falling back to Appium set_network_connection.",
                        e,
                    )
            elif mode == "online":
                try:
                    logger.info("Disabling airplane mode via ADB connectivity service")
                    AdbControl.set_airplane_mode(udid_or_serial, False)
                    success = True
                except Exception as e:
                    logger.warning(
                        "ADB connectivity control failed: %s. Falling back to Appium set_network_connection.",
                        e,
                    )

            if not success:
                # Appium Network Connection IDs: 1 (Airplane), 2 (Wifi), 4 (Data), 6 (All)
                type_id = 6
                if mode in ("offline", "airplane"):
                    type_id = 1
                elif mode == "wifi_only":
                    type_id = 2
                elif mode == "data_only":
                    type_id = 4
                logger.info(
                    "Setting Android network connection to type: %d (%s) via Appium",
                    type_id,
                    mode,
                )
                self.driver.set_network_connection(type_id)
        else:
            logger.warning("Network throttling not supported on iOS Simulator.")

    def _handle_battery(self, udid_or_serial: str, params: dict):
        level = int(params.get("level", 100))
        charging = bool(params.get("charging", True))

        if self.platform == Platform.IOS:
            state = "charged" if charging else "discharged"
            cmd = [
                "xcrun",
                "simctl",
                "status_bar",
                udid_or_serial,
                "override",
                "--batteryLevel",
                str(level),
                "--batteryState",
                state,
            ]
            logger.info("iOS status bar override: %s", " ".join(cmd))
            run_command(cmd, check=True)
        else:
            logger.info(
                "Setting Android battery level to %d, charging=%s", level, charging
            )
            AdbControl.set_battery_status(udid_or_serial, level, charging)

    def _handle_timezone(self, udid_or_serial: str, params: dict):
        tz = params.get("timezone")
        if not tz:
            raise ValueError("Timezone requires 'timezone' parameter")

        if self.platform == Platform.ANDROID:
            logger.info("Setting Android timezone: %s", tz)
            AdbControl.set_timezone(udid_or_serial, tz)
            # Must restart app for timezone to take effect
            package = self.driver.capabilities.get("appPackage") or getattr(
                self.device_control, "package_id", None
            )
            if package:
                logger.info("Restarting app %s to apply timezone changes", package)
                AdbControl.force_stop_app(udid_or_serial, package)
                AdbControl.start_app_via_monkey(udid_or_serial, package)
        else:
            logger.warning("Timezone change not supported on iOS Simulator via simctl.")

    def _handle_locale(self, udid_or_serial: str, params: dict):
        lang = params.get("language")
        country = params.get("country", "")
        if not lang:
            raise ValueError("Locale requires 'language' parameter")

        if self.platform == Platform.ANDROID:
            locale_str = f"{lang}-{country}" if country else lang
            package = self.driver.capabilities.get("appPackage") or getattr(
                self.device_control, "package_id", None
            )
            if package:
                logger.info("Setting Android locale: %s", locale_str)
                AdbControl.set_app_locale(udid_or_serial, package, locale_str)
            else:
                logger.warning("Cannot set Android locale without target package ID")
        else:
            logger.warning(
                "Locale change must be passed at boot/app launch on iOS Simulator."
            )

    def _handle_permission(self, udid_or_serial: str, params: dict):
        perm = params.get("permission")
        grant = bool(params.get("grant", True))
        package = self.driver.capabilities.get("appPackage") or getattr(
            self.device_control, "package_id", None
        )

        if not perm:
            raise ValueError("Permission requires 'permission' parameter")
        if not package:
            raise ValueError("Permission requires target app package ID/bundle ID")

        action = "grant" if grant else "revoke"
        if self.platform == Platform.IOS:
            service_map = {
                "location": "location",
                "camera": "camera",
                "notifications": "all",
                "photos": "photos",
                "contacts": "contacts",
            }
            service = service_map.get(perm.lower(), perm.lower())
            cmd = [
                "xcrun",
                "simctl",
                "privacy",
                udid_or_serial,
                action,
                service,
                package,
            ]
            logger.info("iOS permission command: %s", " ".join(cmd))
            run_command(cmd, check=True)
        else:
            perm_map = {
                "camera": "android.permission.CAMERA",
                "location": "android.permission.ACCESS_FINE_LOCATION",
                "contacts": "android.permission.READ_CONTACTS",
                "storage": "android.permission.READ_EXTERNAL_STORAGE",
            }
            android_perm = perm_map.get(perm.lower(), perm)
            logger.info(
                "Android permission command: %s for package %s", android_perm, package
            )
            AdbControl.set_app_permission(udid_or_serial, package, android_perm, grant)

    def _handle_interrupt(self, udid_or_serial: str, params: dict):
        int_type = str(params.get("type", "low_memory")).lower()
        if int_type == "low_memory":
            if self.platform == Platform.IOS:
                cmd = [
                    "xcrun",
                    "simctl",
                    "notify_post",
                    udid_or_serial,
                    "com.apple.CoreSimulator.SimDevice.MemoryNotification",
                ]
                logger.info("iOS Memory Warning: %s", " ".join(cmd))
                run_command(cmd, check=True)
            else:
                package = self.driver.capabilities.get("appPackage") or getattr(
                    self.device_control, "package_id", None
                )
                if package:
                    pid = AdbControl.get_app_pid(udid_or_serial, package)
                    if pid:
                        logger.info("Android low memory simulation for PID: %s", pid)
                        AdbControl.simulate_low_memory(udid_or_serial, pid)
        elif int_type == "incoming_call":
            phone = params.get("phone", "555-5555")
            if self.platform == Platform.ANDROID:
                logger.info("Android incoming call: %s", phone)
                AdbControl.simulate_incoming_call(udid_or_serial, phone)
            else:
                logger.warning(
                    "Incoming call simulation not supported on iOS Simulator via simctl."
                )

    def _handle_clipboard(self, udid_or_serial: str, params: dict):
        text = params.get("text", "")
        logger.info("Setting device clipboard: %s", text)
        self.driver.set_clipboard_text(text)

    def _handle_biometric(self, udid_or_serial: str, params: dict):
        action = str(params.get("action", "match")).lower()
        if self.platform == Platform.IOS:
            enroll = action in ("enroll", "match")
            self.device_control.execute_script(
                "mobile: enrollBiometric", {"isEnabled": enroll}
            )
            if action in ("match", "unmatch"):
                biometric_type = params.get("type", "faceId")
                match_val = action == "match"
                self.device_control.execute_script(
                    "mobile: sendBiometricMatch",
                    {"type": biometric_type, "match": match_val},
                )
        else:
            finger_id = int(params.get("finger_id", 1))
            self.driver.finger_print(finger_id)

    def _handle_deeplink(self, udid_or_serial: str, params: dict):
        url = params.get("url")
        if not url:
            raise ValueError("Deep Link requires 'url' parameter")

        if self.platform == Platform.IOS:
            cmd = ["xcrun", "simctl", "openurl", udid_or_serial, url]
            logger.info("iOS open URL: %s", " ".join(cmd))
            run_command(cmd, check=True)
        else:
            package = self.driver.capabilities.get("appPackage") or getattr(
                self.device_control, "package_id", None
            )
            logger.info("Android open deep link: %s for package %s", url, package)
            AdbControl.open_deeplink(udid_or_serial, url, package)

    def _handle_app_state(self, udid_or_serial: str, params: dict):
        action = str(params.get("action", "force_stop")).lower()
        package = self.driver.capabilities.get("appPackage") or getattr(
            self.device_control, "package_id", None
        )
        if not package:
            raise ValueError(
                "App State control requires target app package ID/bundle ID"
            )

        if action == "force_stop":
            if self.platform == Platform.IOS:
                cmd = ["xcrun", "simctl", "terminate", udid_or_serial, package]
                logger.info("iOS terminate app: %s", " ".join(cmd))
                run_command(cmd, check=True)
            else:
                logger.info("Android force stop app: %s", package)
                AdbControl.force_stop_app(udid_or_serial, package)
        elif action == "clear_data":
            if self.platform == Platform.ANDROID:
                logger.info("Android clear app data: %s", package)
                AdbControl.clear_app_data(udid_or_serial, package)
            else:
                logger.warning(
                    "Clearing iOS App data not supported directly via simctl."
                )

    def _handle_media_inject(self, udid_or_serial: str, params: dict):
        local_path = params.get("local_path")
        if not local_path or not os.path.exists(local_path):
            raise ValueError(
                f"Media Inject requires a valid, existing 'local_path'. Provided: {local_path}"
            )

        if self.platform == Platform.IOS:
            cmd = ["xcrun", "simctl", "addmedia", udid_or_serial, local_path]
            logger.info("iOS simctl add media: %s", " ".join(cmd))
            run_command(cmd, check=True)
        else:
            dest_path = f"/sdcard/Pictures/{os.path.basename(local_path)}"
            logger.info("Android media push: %s to %s", local_path, dest_path)
            AdbControl.push_file(udid_or_serial, local_path, dest_path)
            AdbControl.scan_media_file(udid_or_serial, dest_path)

    def _handle_accessibility(self, udid_or_serial: str, params: dict):
        enabled = bool(params.get("enabled", True))
        if self.platform == Platform.ANDROID:
            service = "com.google.android.talkback/com.google.android.marvin.talkback.TalkBackService"
            logger.info(
                "Setting Android accessibility service %s: enabled=%s", service, enabled
            )
            AdbControl.set_accessibility_service(udid_or_serial, service, enabled)
        else:
            logger.warning(
                "Accessibility toggle capability not supported on iOS Simulator via simctl."
            )
