import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)


class AdbControl:
    """Provides shell interfaces and wrapper utilities for working with Android devices via ADB."""

    def __init__(self):
        """Initializes the AdbControl utility with base settings commands."""
        self.adb = "adb"
        self.base_settings = ["adb", "shell", "settings"]
        self.base_system = self.base_settings + ["put", "system"]

    def run_adb_command(self, command):
        """Runs a raw command in a shell subprocess.

        Args:
            command: A list of command-line arguments to execute.

        Returns:
            The CompletedProcess result from the run command.
        """
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to run command: {command}")
        return result

    def find_aapt(self):
        """Finds the path to the aapt binary in the local Android SDK installation directory.

        Returns:
            The string path to the resolved aapt executable.
        """
        ANDROID_HOME = os.getenv("ANDROID_HOME")
        if ANDROID_HOME is None:
            raise ValueError("ANDROID_HOME is not set")
        command = ["find", f"{ANDROID_HOME}/build-tools", "-name", "aapt"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to find aapt: {result.stderr}")
        if not result.stdout.strip():
            raise ValueError("aapt not found")
        aapt_path = result.stdout.split("\n")[0]
        if aapt_path is None:
            raise ValueError("aapt not found")
        return aapt_path

    def get_apk_info(self, apk_path):
        """Inspects badging info inside a built APK file to fetch its Package ID and launchable activity.

        Args:
            apk_path: The filesystem path to the target APK.

        Returns:
            A tuple of (package_id, launchable_activity_name) parsed from the badging details.
        """
        aapt = self.find_aapt()
        logger.info("Using aapt at: %s", aapt)
        result = subprocess.run(
            [aapt, "dump", "badging", apk_path], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get APK info: {result.stderr}")
        stdout = result.stdout
        package_match = re.search(r"package: name='([^']+)'", stdout)
        activity_match = re.search(r"launchable-activity: name='([^']+)'", stdout)
        package_id = package_match.group(1) if package_match else None
        activity = activity_match.group(1) if activity_match else None
        return package_id, activity

    def toggle_tap_highlights(self, enabled: bool):
        """Enables or disables visual touch indicators/highlights on the target screen.

        Args:
            enabled: True to display touch highlights, False to turn them off.
        """
        try:
            command = (
                self.base_system + ["show_touches", "1"]
                if enabled
                else self.base_system + ["show_touches", "0"]
            )
            self.run_adb_command(command)
        except Exception as e:
            raise RuntimeError(f"Failed to toggle tap highlights: {e}")

    def toggle_pointer_location(self, enabled: bool):
        """Enables or disables pointer coordinates tracing overlay at the top of the screen.

        Args:
            enabled: True to display trace details, False to turn them off.
        """
        try:
            command = (
                self.base_system + ["pointer_location", "1"]
                if enabled
                else self.base_system + ["pointer_location", "0"]
            )
            self.run_adb_command(command)
            logger.info("Pointer location %s", 'enabled' if enabled else 'disabled')
        except Exception as e:
            raise RuntimeError(f"Failed to toggle pointer location: {e}")

    def enable_feedback(self, pointer_location: bool, tap_highlights: bool):
        """Configures device gesture feedback preferences in Developer Settings.

        Args:
            pointer_location: True to display pointer coordinates, False otherwise.
            tap_highlights: True to show circle touches, False otherwise.
        """
        try:
            self.toggle_pointer_location(pointer_location)
            self.toggle_tap_highlights(tap_highlights)
        except Exception as e:
            raise RuntimeError(f"Failed to enable feedback: {e}")

    def disable_feedback(self):
        """Disables all system pointer tracking and touch highlights on the device."""
        try:
            self.toggle_pointer_location(False)
            self.toggle_tap_highlights(False)
        except Exception as e:
            raise RuntimeError(f"Failed to disable feedback: {e}")

    def resolve_emulator_serial(self, device_name: str) -> str | None:
        """Resolves an AVD profile name or serial to an active emulator serial number."""
        try:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
            if result.returncode != 0:
                return None

            devices = re.findall(r"^([^\s]+)\s+device", result.stdout, re.MULTILINE)
            for serial in devices:
                if serial == device_name:
                    return serial

                # If it's an emulator serial, query its AVD profile name
                if serial.startswith("emulator-"):
                    avd_result = subprocess.run(
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

    def kill_emulator(self, device_name: str, serial: str = None):
        """Terminates a running emulator matching the specified device name (AVD profile name or serial)."""
        try:
            target_serial = serial or self.resolve_emulator_serial(device_name)
            if target_serial:
                logger.info("Found matching emulator serial '%s' for '%s'. Terminating.", target_serial, device_name)
                self.run_adb_command(["adb", "-s", target_serial, "emu", "kill"])
            else:
                logger.warning("No active emulator serial found matching '%s'. Skipping termination.", device_name)
        except Exception as e:
            raise RuntimeError(f"Failed to kill emulator: {e}")
