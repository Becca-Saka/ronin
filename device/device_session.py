import os
import time
from pathlib import Path
from uuid import uuid4

from appium.webdriver.appium_service import AppiumService

from device.adb_control import AdbControl
from device.device_control import DeviceControl
from utils.platform_checks import run_ios_preflight, validate_app_path_for_platform


class DeviceSession:
    def __init__(
        self,
        device_name: str,
        app_path: str,
        record_session: bool = False,
        session_dir: Path = Path("sessions"),
        platform: str = "android",
    ):
        self.session_dir = session_dir
        self.device_name = device_name
        self.app_path = app_path
        self.platform = platform
        self.device_control: DeviceControl | None = None
        # Run all preflight checks before touching any hardware
        if self.platform == "ios":
            run_ios_preflight(app_path)
        else:
            validate_app_path_for_platform(app_path, self.platform)
            self.adb_control = AdbControl()
        self.record_session = record_session
        self.is_recording = False
        self.session_id = str(uuid4())
        host = os.environ.get("APPIUM_HOST", "127.0.0.1")
        port = int(os.environ.get("APPIUM_PORT", "4723"))

        self._appium_service = AppiumService()
        self._appium_args = {
            "host": host,
            "port": port,
            "args": ["--log", "/tmp/ai_e2e_appium.log"],
        }

    def _start_appium(self):
        if self._appium_service.is_running:
            print("Appium already running")
            return
        print("Starting Appium...")
        self._appium_service.start(**self._appium_args)
        if not self._appium_service.is_running:
            raise RuntimeError("Appium failed to start. See /tmp/ai_e2e_appium.log")
        print("Appium started")

    def start(self, is_headless: bool = False) -> DeviceControl:
        self._start_appium()
        self.device_control = DeviceControl(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            adb_control=self.adb_control if self.platform == "android" else None,
        )
        if self.platform == "android":
            self.adb_control.enable_feedback(
                pointer_location=False, tap_highlights=True
            )
        time.sleep(6)
        if self.record_session:
            self.device_control.start_recording()
            self.is_recording = True
        return self.device_control

    def close(self):
        if self.is_recording:
            session_path = self.session_dir / f"session_{self.session_id}.mp4"
            self.device_control.stop_recording(session_path)
            self.is_recording = False
            print(f"Session recording saved to {session_path}")
        if self.device_control:
            self.device_control.quit()
        if self._appium_service.is_running:
            self._appium_service.stop()
