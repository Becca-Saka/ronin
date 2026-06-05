import os
import time

from appium.webdriver.appium_service import AppiumService

from device.adb_control import AdbControl
from device.device_control import DeviceControl


class DeviceSession:
    def __init__(self, device_name: str, app_path: str):
        self.adb_control = AdbControl()
        self.device_name = device_name
        self.app_path = app_path
        self.device_control: DeviceControl | None = None

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

    def start(self) -> DeviceControl:
        self._start_appium()
        self.device_control = DeviceControl(
            device_name=self.device_name,
            app_path=self.app_path,
            adb_control=self.adb_control,
        )
        self.adb_control.enable_feedback(pointer_location=False, tap_highlights=True)
        time.sleep(6)
        return self.device_control

    def close(self):
        if self.device_control:
            self.device_control.quit()
        if self._appium_service.is_running:
            self._appium_service.stop()
