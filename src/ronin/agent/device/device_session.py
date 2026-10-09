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

from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.device_control import DeviceControl
from ronin.agent.device.platform import Platform
from ronin.utils.platform_checks import run_android_preflight, run_ios_preflight

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
        session_id: str = None,
    ):
        """Initializes a DeviceSession.

        Args:
            device_name: Name of the emulator/simulator/device to run against.
            app_path: Path to the target mobile app bundle (.apk, .app, or .ipa).
            record_session: If True, records the screen during execution.
            session_dir: Directory where screen recordings will be saved.
            platform: Platform name ('android' or 'ios').
            session_id: Optional existing session ID to reuse.
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
            run_android_preflight(app_path)
            self.adb_control = None
        self.record_session = record_session
        self.is_recording = False
        self.session_id = session_id or str(uuid4())
        
        # Ensure structured session directories exist
        self.session_root = self.session_dir / self.session_id
        self.logs_dir = self.session_root / "logs"
        self.screenshots_dir = self.session_root / "screenshots"
        self.recordings_dir = self.session_root / "recordings"
        self.telemetry_dir = self.session_root / "telemetry"

        self._ensure_directories()

        self.appium_log_path = self.logs_dir / f"appium_session_{self.session_id}.log"
        self.root_log_file = self.logs_dir / f"ronin_session_{self.session_id}.log"
        
        # Create a file log handler to save all python logging output to the assets directory
        self._file_handler = logging.FileHandler(self.root_log_file, encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(self._file_handler)
        
        host = os.environ.get("APPIUM_HOST", "127.0.0.1")
        port = int(os.environ.get("APPIUM_PORT", "4723"))
 
        self._appium_service = AppiumService()
        self._appium_args = {
            "host": host,
            "port": port,
            "args": ["--log", str(self.appium_log_path)],
        }
 
    def _ensure_directories(self) -> None:
        for directory in (
            self.logs_dir,
            self.screenshots_dir,
            self.recordings_dir,
            self.telemetry_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _start_appium(self):
        if self._appium_service.is_running:
            logger.info("Appium already running")
            return
        logger.info("Starting Appium...")
        self._appium_service.start(**self._appium_args)
        if not self._appium_service.is_running:
            raise RuntimeError(f"Appium failed to start. See {self.appium_log_path}")
        logger.info("Appium started")
 
    def start(self, is_headless: bool = False) -> DeviceControl:
        self._ensure_directories()
        self._start_appium()
        self.device_control = DeviceControl(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            adb_control=None,
            is_headless=is_headless,
        )
        if self.platform == Platform.ANDROID:
            serial = self.device_control.serial or self.device_name
            AdbControl.enable_feedback(
                serial=serial, pointer_location=False, tap_highlights=True
            )
        time.sleep(6)
        if self.record_session:
            self.device_control.start_recording()
            self.is_recording = True
        return self.device_control

    def close(self):
        if self.is_recording:
            session_path = self.recordings_dir / f"session_{self.session_id}.mp4"
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
        if hasattr(self, "_file_handler"):
            try:
                logging.getLogger().removeHandler(self._file_handler)
                self._file_handler.close()
            except Exception as e:
                logger.warning("Failed to close root file handler: %s", e)
