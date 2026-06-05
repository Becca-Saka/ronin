import os
import re
import subprocess


class AdbControl:
    def __init__(self):
        self.adb = "adb"
        self.base_settings = ["adb", "shell", "settings"]
        self.base_system = self.base_settings + ["put", "system"]

    def run_adb_command(self, command):
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to run command: {command}")
        return result

    def find_aapt(self):
        "Find the aapt command in the Android SDK"

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
        aapt = self.find_aapt()
        print(f"aapt: {aapt}")
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
        try:
            command = (
                self.base_system + ["pointer_location", "1"]
                if enabled
                else self.base_system + ["pointer_location", "0"]
            )
            self.run_adb_command(command)
            print(f"Pointer location {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            raise RuntimeError(f"Failed to toggle pointer location: {e}")

    def enable_feedback(self, pointer_location: bool, tap_highlights: bool):
        try:
            self.toggle_pointer_location(pointer_location)
            self.toggle_tap_highlights(tap_highlights)
        except Exception as e:
            raise RuntimeError(f"Failed to enable feedback: {e}")

    def disable_feedback(self):
        try:
            self.toggle_pointer_location(False)
            self.toggle_tap_highlights(False)
        except Exception as e:
            raise RuntimeError(f"Failed to disable feedback: {e}")

    def kill_emulator(self, device_name: str):
        try:
            self.run_adb_command(["adb", "-s", device_name, "emu", "kill"])
        except Exception as e:
            raise RuntimeError(f"Failed to kill emulator: {e}")
