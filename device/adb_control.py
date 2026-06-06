import logging
import os
import re

from utils.subprocess_runner import run_command, SubprocessRunError

logger = logging.getLogger(__name__)


class AdbControl:
    """Provides shell interfaces and wrapper utilities for working with Android devices via ADB."""

    ADB = "adb"
    BASE_SETTINGS = ["adb", "shell", "settings"]
    BASE_SYSTEM = ["adb", "shell", "settings", "put", "system"]

    @staticmethod
    def run_adb_command(command):
        """Runs a raw command in a shell subprocess.

        Args:
            command: A list of command-line arguments to execute.

        Returns:
            The CompletedProcess result from the run command.
        """
        try:
            return run_command(command, check=True)
        except SubprocessRunError as e:
            raise RuntimeError(f"Failed to run command: {command}") from e

    @staticmethod
    def find_aapt():
        """Finds the path to the aapt binary in the local Android SDK installation directory.

        Returns:
            The string path to the resolved aapt executable.
        """
        try:
            ANDROID_HOME = os.getenv("ANDROID_HOME")
            if ANDROID_HOME is None:
                raise ValueError("ANDROID_HOME is not set")
            command = ["find", f"{ANDROID_HOME}/build-tools", "-name", "aapt"]
            result = run_command(command, check=True, capture_output=True, text=True)
            if not result.stdout.strip():
                raise ValueError("aapt not found")
            aapt_path = result.stdout.split("\n")[0]
            if aapt_path is None:
                raise ValueError("aapt not found")
            return aapt_path
        except SubprocessRunError as e:
            raise RuntimeError(f"Failed to find aapt: {e.stderr}") from e

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
        except SubprocessRunError as e:
            raise RuntimeError(f"Failed to get APK info: {e.stderr}") from e

    @staticmethod
    def toggle_tap_highlights(serial: str, enabled: bool):
        """Enables or disables visual touch indicators/highlights on the target screen.

        Args:
            serial: The target device serial number.
            enabled: True to display touch highlights, False to turn them off.
        """
        try:
            val = "1" if enabled else "0"
            command = ["adb", "-s", serial, "shell", "settings", "put", "system", "show_touches", val]
            AdbControl.run_adb_command(command)
        except Exception as e:
            raise RuntimeError(f"Failed to toggle tap highlights: {e}")

    @staticmethod
    def toggle_pointer_location(serial: str, enabled: bool):
        """Enables or disables pointer coordinates tracing overlay at the top of the screen.

        Args:
            serial: The target device serial number.
            enabled: True to display trace details, False to turn them off.
        """
        try:
            val = "1" if enabled else "0"
            command = ["adb", "-s", serial, "shell", "settings", "put", "system", "pointer_location", val]
            AdbControl.run_adb_command(command)
            logger.info("Pointer location %s", 'enabled' if enabled else 'disabled')
        except Exception as e:
            raise RuntimeError(f"Failed to toggle pointer location: {e}")

    @staticmethod
    def enable_feedback(serial: str, pointer_location: bool, tap_highlights: bool):
        """Configures device gesture feedback preferences in Developer Settings.

        Args:
            serial: The target device serial number.
            pointer_location: True to display pointer coordinates, False otherwise.
            tap_highlights: True to show circle touches, False otherwise.
        """
        try:
            AdbControl.toggle_pointer_location(serial, pointer_location)
            AdbControl.toggle_tap_highlights(serial, tap_highlights)
        except Exception as e:
            raise RuntimeError(f"Failed to enable feedback: {e}")

    @staticmethod
    def disable_feedback(serial: str):
        """Disables all system pointer tracking and touch highlights on the device."""
        try:
            AdbControl.toggle_pointer_location(serial, False)
            AdbControl.toggle_tap_highlights(serial, False)
        except Exception as e:
            raise RuntimeError(f"Failed to disable feedback: {e}")

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
                if serial.startswith("emulator-"):
                    avd_result = run_command(
                        ["adb", "-s", serial, "emu", "avd", "name"],
                        capture_output=True,
                        text=True,
                        timeout=3.0
                    )
                    if avd_result.returncode == 0:
                        avd_name = avd_result.stdout.strip()
                        if avd_name == device_name:
                            return serial
        except Exception as e:
            logger.warning("Failed to resolve emulator serial for %s: %s", device_name, e)
        return None

    @staticmethod
    def kill_emulator(device_name: str, serial: str = None):
        """Terminates a running emulator matching the specified device name (AVD profile name or serial)."""
        try:
            target_serial = serial or AdbControl.resolve_emulator_serial(device_name)
            if target_serial:
                logger.info("Found matching emulator serial '%s' for '%s'. Terminating.", target_serial, device_name)
                AdbControl.run_adb_command(["adb", "-s", target_serial, "emu", "kill"])
            else:
                logger.warning("No active emulator serial found matching '%s'. Skipping termination.", device_name)
        except Exception as e:
            raise RuntimeError(f"Failed to kill emulator: {e}")

    @staticmethod
    def mock_location(serial: str, lat: float, lng: float):
        """Mocks GPS location coordinates on the emulator."""
        try:
            cmd = ["adb", "-s", serial, "emu", "geo", "fix", str(lng), str(lat)]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to mock location: {e}")

    @staticmethod
    def set_airplane_mode(serial: str, enabled: bool):
        """Enables or disables airplane mode via connectivity service."""
        try:
            action = "enable" if enabled else "disable"
            cmd = ["adb", "-s", serial, "shell", "cmd", "connectivity", "airplane-mode", action]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to set airplane mode: {e}")

    @staticmethod
    def set_battery_status(serial: str, level: int, charging: bool):
        """Overrides battery level and AC power charging status."""
        try:
            ac_val = "1" if charging else "0"
            run_command(["adb", "-s", serial, "shell", "dumpsys", "battery", "set", "level", str(level)], check=True)
            run_command(["adb", "-s", serial, "shell", "dumpsys", "battery", "set", "ac", ac_val], check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to set battery status: {e}")

    @staticmethod
    def set_timezone(serial: str, timezone: str):
        """Sets the system timezone via settings global service."""
        try:
            cmd = ["adb", "-s", serial, "shell", "settings", "put", "global", "timezone", timezone]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to set timezone: {e}")

    @staticmethod
    def set_app_locale(serial: str, package_id: str, locale_str: str):
        """Configures language locale settings for the target application package."""
        try:
            cmd = ["adb", "-s", serial, "shell", "cmd", "locale", "set-app-locales", package_id, "--locales", locale_str]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to set app locale: {e}")

    @staticmethod
    def set_app_permission(serial: str, package_id: str, permission: str, grant: bool):
        """Grants or revokes a specific system permission for the target package."""
        try:
            action_cmd = "grant" if grant else "revoke"
            cmd = ["adb", "-s", serial, "shell", "pm", action_cmd, package_id, permission]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to set app permission: {e}")

    @staticmethod
    def get_app_pid(serial: str, package_id: str) -> str:
        """Retrieves the active process ID (PID) of the target package."""
        try:
            res = run_command(["adb", "-s", serial, "shell", "pidof", package_id], capture_output=True, text=True)
            return res.stdout.strip()
        except Exception as e:
            logger.warning("Failed to get PID for %s: %s", package_id, e)
            return ""

    @staticmethod
    def simulate_low_memory(serial: str, pid: str):
        """Triggers a low memory simulation (RUNNING_CRITICAL) on the target PID."""
        try:
            cmd = ["adb", "-s", serial, "shell", "am", "send-trim-memory", pid, "RUNNING_CRITICAL"]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to simulate low memory: {e}")

    @staticmethod
    def simulate_incoming_call(serial: str, phone: str):
        """Simulates an incoming GSM call on the emulator."""
        try:
            cmd = ["adb", "-s", serial, "emu", "gsm", "call", str(phone)]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to simulate incoming call: {e}")

    @staticmethod
    def open_deeplink(serial: str, url: str, package_id: str = None):
        """Launches a deep link URL intent via Activity Manager."""
        try:
            cmd = ["adb", "-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url]
            if package_id:
                cmd.append(package_id)
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to open deep link: {e}")

    @staticmethod
    def force_stop_app(serial: str, package_id: str):
        """Force stops the target application package."""
        try:
            cmd = ["adb", "-s", serial, "shell", "am", "force-stop", package_id]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to force stop app: {e}")

    @staticmethod
    def clear_app_data(serial: str, package_id: str):
        """Clears all storage and user data for the target application package."""
        try:
            cmd = ["adb", "-s", serial, "shell", "pm", "clear", package_id]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to clear app data: {e}")

    @staticmethod
    def push_file(serial: str, local_path: str, remote_path: str):
        """Pushes a local file to the emulator filesystem."""
        try:
            cmd = ["adb", "-s", serial, "push", local_path, remote_path]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to push file: {e}")

    @staticmethod
    def scan_media_file(serial: str, remote_path: str):
        """Broadcasts a media scan request for the specified path."""
        try:
            cmd = [
                "adb", "-s", serial, "shell", "am", "broadcast",
                "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
                "-d", f"file://{remote_path}"
            ]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to scan media file: {e}")

    @staticmethod
    def start_app_via_monkey(serial: str, package_id: str):
        """Starts the target app using the monkey tool."""
        try:
            cmd = ["adb", "-s", serial, "shell", "monkey", "-p", package_id, "1"]
            run_command(cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Failed to start app via monkey: {e}")
