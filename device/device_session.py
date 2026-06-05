"""
Manages Appium server lifecycle and device connection session.
Supports starting Appium programmatically, checking platforms, and screen recording.
"""

import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from appium.webdriver.appium_service import AppiumService

from device.adb_control import AdbControl
from device.device_control import DeviceControl
from device.platform import Platform
from utils.platform_checks import run_ios_preflight, validate_app_path_for_platform

logger = logging.getLogger(__name__)


class DeviceSession:
    """Manages the startup/teardown of the Appium server and mobile device driver session."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        record_session: bool = False,
        session_dir: Path = Path("sessions"),
        platform: Platform = Platform.ANDROID,
    ):
        """Initializes a DeviceSession.

        Args:
            device_name: Name of the emulator/simulator/device to run against.
            app_path: Path to the target mobile app bundle (.apk, .app, or .ipa).
            record_session: If True, records the screen during execution.
            session_dir: Directory where screen recordings will be saved.
            platform: Platform name ('android' or 'ios').
        """

        self.session_dir = session_dir
        self.device_name = device_name
        self.app_path = app_path
        self.platform = platform
        self.device_control: DeviceControl | None = None
        # Run all preflight checks before touching any hardware
        if self.platform == Platform.IOS:
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
            "args": ["--log", "/tmp/ronin_appium.log"],
        }

    def _start_appium(self):
        if self._appium_service.is_running:
            logger.info("Appium already running")
            return
        logger.info("Starting Appium...")
        self._appium_service.start(**self._appium_args)
        if not self._appium_service.is_running:
            raise RuntimeError("Appium failed to start. See /tmp/ronin_appium.log")
        logger.info("Appium started")

    def start(self, is_headless: bool = False) -> DeviceControl:
        self._start_appium()
        self.device_control = DeviceControl(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            adb_control=self.adb_control if self.platform == Platform.ANDROID else None,
            is_headless=is_headless,
        )
        if self.platform == Platform.ANDROID:
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
            try:
                self.device_control.stop_recording(session_path)
                logger.info("Session recording saved to %s", session_path)
            except Exception as e:
                logger.warning("Failed to save session recording: %s", e)
            self.is_recording = False
        if self.device_control:
            try:
                self.device_control.quit()
            except Exception as e:
                logger.warning("Failed to quit device control: %s", e)
        if self._appium_service.is_running:
            try:
                self._appium_service.stop()
                logger.info("Appium service stopped")
            except Exception as e:
                logger.warning("Failed to stop Appium service: %s", e)
