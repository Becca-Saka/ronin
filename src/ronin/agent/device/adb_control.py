import logging
import os
import re
from typing import Optional

from ronin.errors import DeviceUnavailable
from ronin.utils.subprocess_runner import run_command, SubprocessRunError

logger = logging.getLogger(__name__)


class AdbControl:
    """Provides shell interfaces and wrapper utilities for working with Android devices via ADB."""

    ADB = "adb"
    BASE_SETTINGS = ["adb", "shell", "settings"]
    BASE_SYSTEM = ["adb", "shell", "settings", "put", "system"]

    @staticmethod
    def run_adb_command(command, doing: str = ""):
        """Runs one adb command, or reports what the device could not do.

        Args:
            command: A list of command-line arguments to execute.
            doing: What the command was for. The failure message includes it,
                because the argv does not say why Ronin ran the command.

        Returns:
            The CompletedProcess result from the run command.

        Raises:
            DeviceUnavailable: If the command failed, or Ronin could not run it.
        """
        try:
            return run_command(command, check=True)
        except SubprocessRunError as exc:
            attempt = f"Failed to {doing}" if doing else "Failed to run adb command"
            raise DeviceUnavailable(f"{attempt}: {' '.join(command)}") from exc

    @staticmethod
    def find_aapt():
        """Finds the path to the aapt binary in the local Android SDK installation directory.

        Returns:
            The string path to the resolved aapt executable.
        """
        try:
            ANDROID_HOME = os.getenv("ANDROID_HOME")
            if ANDROID_HOME is None:
                raise DeviceUnavailable("ANDROID_HOME is not set")
            command = ["find", f"{ANDROID_HOME}/build-tools", "-name", "aapt"]
            result = run_command(command, check=True, capture_output=True, text=True)
            if not result.stdout.strip():
                raise DeviceUnavailable(f"No aapt under {ANDROID_HOME}/build-tools")
            return result.stdout.split("\n")[0]
        except SubprocessRunError as exc:
            raise DeviceUnavailable(f"Failed to find aapt: {exc.stderr}") from exc

    @staticmethod
    def get_apk_info(apk_path):
        """Inspects badging info inside a built APK file to fetch its Package ID and launchable activity.

        Args:
            apk_path: The filesystem path to the target APK.

        Returns:
            A tuple of (package_id, launchable_activity_name) parsed from the badging details.
        """
        try:
            aapt = AdbControl.find_aapt()
            logger.info("Using aapt at: %s", aapt)
            result = run_command(
                [aapt, "dump", "badging", apk_path], check=True, capture_output=True, text=True
            )
            stdout = result.stdout
            package_match = re.search(r"package: name='([^']+)'", stdout)
            activity_match = re.search(r"launchable-activity: name='([^']+)'", stdout)
            package_id = package_match.group(1) if package_match else None
            activity = activity_match.group(1) if activity_match else None
            return package_id, activity
        except SubprocessRunError as exc:
            raise DeviceUnavailable(f"Failed to read {apk_path}: {exc.stderr}") from exc

    @staticmethod
    def toggle_tap_highlights(serial: str, enabled: bool):
        """Enables or disables visual touch indicators/highlights on the target screen.

        Args:
            serial: The target device serial number.
            enabled: True to display touch highlights, False to turn them off.
        """
        val = "1" if enabled else "0"
        command = ["adb", "-s", serial, "shell", "settings", "put", "system", "show_touches", val]
        AdbControl.run_adb_command(command, doing="toggle tap highlights")

    @staticmethod
    def toggle_pointer_location(serial: str, enabled: bool):
        """Enables or disables pointer coordinates tracing overlay at the top of the screen.

        Args:
            serial: The target device serial number.
            enabled: True to display trace details, False to turn them off.
        """
        val = "1" if enabled else "0"
        command = ["adb", "-s", serial, "shell", "settings", "put", "system", "pointer_location", val]
        AdbControl.run_adb_command(command, doing="toggle pointer location")
        logger.info("Pointer location %s", 'enabled' if enabled else 'disabled')

    @staticmethod
    def enable_feedback(serial: str, pointer_location: bool, tap_highlights: bool):
        """Configures device gesture feedback preferences in Developer Settings.

        Args:
            serial: The target device serial number.
            pointer_location: True to display pointer coordinates, False otherwise.
            tap_highlights: True to show circle touches, False otherwise.
        """
        AdbControl.toggle_pointer_location(serial, pointer_location)
        AdbControl.toggle_tap_highlights(serial, tap_highlights)

    @staticmethod
    def disable_feedback(serial: str):
        """Disables all system pointer tracking and touch highlights on the device."""
        AdbControl.toggle_pointer_location(serial, False)
        AdbControl.toggle_tap_highlights(serial, False)

    @staticmethod
    def avd_name(serial: str) -> Optional[str]:
        """The AVD a running emulator was started from.

        'adb emu avd name' needs the emulator console auth token and prints
        nothing without it, so read the boot property instead.
        """
        for command in (
            ["adb", "-s", serial, "shell", "getprop", "ro.boot.qemu.avd_name"],
            ["adb", "-s", serial, "emu", "avd", "name"],
        ):
            try:
                result = run_command(command, timeout=10, log_command=False)
            except SubprocessRunError as exc:
                logger.debug("Could not read the AVD name via %s: %s", command[-1], exc)
                continue
            name = (result.stdout or "").strip().splitlines()
            if result.returncode == 0 and name and name[0].strip():
                return name[0].strip()
        return None

    @staticmethod
    def installed_packages(serial: str) -> set[str]:
        """Package names currently installed on a device, empty when adb cannot answer."""
        try:
            result = run_command(
                ["adb", "-s", serial, "shell", "pm", "list", "packages"],
                timeout=20,
                log_command=False,
            )
        except SubprocessRunError as exc:
            logger.debug("Could not list packages on %s: %s", serial, exc)
            return set()
        return {
            line.strip().removeprefix("package:")
            for line in (result.stdout or "").splitlines()
            if line.strip()
        }

    @staticmethod
    def resolve_emulator_serial(device_name: str) -> str | None:
        """Resolves an AVD profile name or serial to an active emulator serial number."""
        try:
            result = run_command(["adb", "devices"], capture_output=True, text=True)
            if result.returncode != 0:
                return None

            devices = re.findall(r"^([^\s]+)\s+device", result.stdout, re.MULTILINE)
            for serial in devices:
                if serial == device_name:
                    return serial

                # If it's an emulator serial, query its AVD profile name
                if serial.startswith("emulator-") and AdbControl.avd_name(serial) == device_name:
                    return serial
        except SubprocessRunError as e:
            logger.warning("Failed to resolve emulator serial for %s: %s", device_name, e)
        return None

    @staticmethod
    def kill_emulator(device_name: str, serial: str = None):
        """Terminates a running emulator matching the specified device name (AVD profile name or serial)."""
        target_serial = serial or AdbControl.resolve_emulator_serial(device_name)
        if target_serial:
            logger.info("Found matching emulator serial '%s' for '%s'. Terminating.", target_serial, device_name)
            AdbControl.run_adb_command(["adb", "-s", target_serial, "emu", "kill"], doing="kill emulator")
        else:
            logger.warning("No active emulator serial found matching '%s'. Skipping termination.", device_name)

    @staticmethod
    def mock_location(serial: str, lat: float, lng: float):
        """Mocks GPS location coordinates on the emulator."""
        cmd = ["adb", "-s", serial, "emu", "geo", "fix", str(lng), str(lat)]
        AdbControl.run_adb_command(cmd, doing="mock location")

    @staticmethod
    def set_airplane_mode(serial: str, enabled: bool):
        """Enables or disables airplane mode via connectivity service."""
        action = "enable" if enabled else "disable"
        cmd = ["adb", "-s", serial, "shell", "cmd", "connectivity", "airplane-mode", action]
        AdbControl.run_adb_command(cmd, doing="set airplane mode")

    @staticmethod
    def set_battery_status(serial: str, level: int, charging: bool):
        """Overrides battery level and AC power charging status."""
        ac_val = "1" if charging else "0"
        AdbControl.run_adb_command(["adb", "-s", serial, "shell", "dumpsys", "battery", "set", "level", str(level)], doing="set battery status")
        AdbControl.run_adb_command(["adb", "-s", serial, "shell", "dumpsys", "battery", "set", "ac", ac_val], doing="set battery status")

    @staticmethod
    def set_timezone(serial: str, timezone: str):
        """Sets the system timezone via settings global service."""
        cmd = ["adb", "-s", serial, "shell", "settings", "put", "global", "timezone", timezone]
        AdbControl.run_adb_command(cmd, doing="set timezone")

    @staticmethod
    def set_app_locale(serial: str, package_id: str, locale_str: str):
        """Configures language locale settings for the target application package."""
        cmd = ["adb", "-s", serial, "shell", "cmd", "locale", "set-app-locales", package_id, "--locales", locale_str]
        AdbControl.run_adb_command(cmd, doing="set app locale")

    @staticmethod
    def set_app_permission(serial: str, package_id: str, permission: str, grant: bool):
        """Grants or revokes a specific system permission for the target package."""
        action_cmd = "grant" if grant else "revoke"
        cmd = ["adb", "-s", serial, "shell", "pm", action_cmd, package_id, permission]
        AdbControl.run_adb_command(cmd, doing="set app permission")

    @staticmethod
    def get_app_pid(serial: str, package_id: str) -> str:
        """Retrieves the active process ID (PID) of the target package."""
        try:
            res = run_command(["adb", "-s", serial, "shell", "pidof", package_id], capture_output=True, text=True)
            return res.stdout.strip()
        except SubprocessRunError as e:
            logger.warning("Failed to get PID for %s: %s", package_id, e)
            return ""

    @staticmethod
    def simulate_low_memory(serial: str, pid: str):
        """Triggers a low memory simulation (RUNNING_CRITICAL) on the target PID."""
        cmd = ["adb", "-s", serial, "shell", "am", "send-trim-memory", pid, "RUNNING_CRITICAL"]
        AdbControl.run_adb_command(cmd, doing="simulate low memory")

    @staticmethod
    def simulate_incoming_call(serial: str, phone: str):
        """Simulates an incoming GSM call on the emulator."""
        cmd = ["adb", "-s", serial, "emu", "gsm", "call", str(phone)]
        AdbControl.run_adb_command(cmd, doing="simulate incoming call")

    @staticmethod
    def open_deeplink(serial: str, url: str, package_id: str = None):
        """Launches a deep link URL intent via Activity Manager."""
        cmd = ["adb", "-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url]
        if package_id:
            cmd.append(package_id)
        AdbControl.run_adb_command(cmd, doing="open deep link")

    @staticmethod
    def force_stop_app(serial: str, package_id: str):
        """Force stops the target application package."""
        cmd = ["adb", "-s", serial, "shell", "am", "force-stop", package_id]
        AdbControl.run_adb_command(cmd, doing="force stop app")

    @staticmethod
    def clear_app_data(serial: str, package_id: str):
        """Clears all storage and user data for the target application package."""
        cmd = ["adb", "-s", serial, "shell", "pm", "clear", package_id]
        AdbControl.run_adb_command(cmd, doing="clear app data")

    @staticmethod
    def push_file(serial: str, local_path: str, remote_path: str):
        """Pushes a local file to the emulator filesystem."""
        cmd = ["adb", "-s", serial, "push", local_path, remote_path]
        AdbControl.run_adb_command(cmd, doing="push file")

    @staticmethod
    def scan_media_file(serial: str, remote_path: str):
        """Broadcasts a media scan request for the specified path."""
        cmd = [
            "adb", "-s", serial, "shell", "am", "broadcast",
            "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
            "-d", f"file://{remote_path}"
        ]
        AdbControl.run_adb_command(cmd, doing="scan media file")

    @staticmethod
    def start_app_via_monkey(serial: str, package_id: str):
        """Starts the target app using the monkey tool."""
        cmd = ["adb", "-s", serial, "shell", "monkey", "-p", package_id, "1"]
        AdbControl.run_adb_command(cmd, doing="start app via monkey")

    @staticmethod
    def set_accessibility_service(serial: str, service: str, enabled: bool):
        """Enables or disables a system accessibility service globally on the device via settings."""
        if enabled:
            logger.info("Registering accessibility service: %s", service)
            AdbControl.run_adb_command(["adb", "-s", serial, "shell", "settings", "put", "secure", "enabled_accessibility_services", service], doing="set accessibility service")
            AdbControl.run_adb_command(["adb", "-s", serial, "shell", "settings", "put", "secure", "accessibility_enabled", "1"], doing="set accessibility service")
        else:
            logger.info("Deregistering accessibility services")
            AdbControl.run_adb_command(["adb", "-s", serial, "shell", "settings", "put", "secure", "enabled_accessibility_services", '""'], doing="set accessibility service")
            AdbControl.run_adb_command(["adb", "-s", serial, "shell", "settings", "put", "secure", "accessibility_enabled", "0"], doing="set accessibility service")
