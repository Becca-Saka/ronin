"""
High-level control interface for interacting with mobile devices (Android/iOS) via Appium.
Encapsulates driver initialization, screenshots, and gesture/input execution.
"""

import base64
import logging
import time
from pathlib import Path
from typing import Optional

from appium import webdriver
from selenium.common.exceptions import WebDriverException
from appium.options.android import UiAutomator2Options
from appium.options.ios import XCUITestOptions
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput

from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.focus import Focus
from ronin.agent.device.diagnostics import diagnose_avd_launch
from ronin.agent.device.platform import Platform
from ronin.defaults import (
    AVD_LAUNCH_TIMEOUT_MS,
)
from ronin.errors import DeviceUnavailable, ObservationUnavailable
from ronin.utils.subprocess_runner import run_command

logger = logging.getLogger(__name__)

ANDROID_KEYCODE_BACK = 4
IOS_APP_STATE_FOREGROUND = 4
RECOVERY_SLEEP_AFTER_BACK = 1.5
RECOVERY_SLEEP_AFTER_ACTIVATE = 2.0


class DeviceControl:
    """Wrapper for executing gesture actions and capturing state from mobile devices."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        adb_control: AdbControl = None,
        is_headless: bool = False,
        platform: Platform = Platform.ANDROID,
        keep_device: bool = False,
        reuse_app: bool = False,
        server_url: str = "http://localhost:4723",
    ):
        self.platform = platform
        self.device_name = device_name
        self.app_path = app_path
        self.bundle_id = None
        self.keep_device = keep_device
        self.reuse_app = reuse_app
        self.server_url = server_url
        if platform == Platform.IOS:
            options = XCUITestOptions()
            options.platform_name = "iOS"
            options.automation_name = "XCUITest"
            options.device_name = device_name
            options.app = app_path
            options.is_headless = is_headless
            options.set_capability("appium:enforceAppInstall", not reuse_app)
            self.adb_control = None
        else:
            self._device_was_running = bool(
                AdbControl.resolve_emulator_serial(device_name)
            )
            options = UiAutomator2Options()
            options.app = app_path
            options.platform_name = "Android"
            options.device_name = device_name
            options.is_headless = is_headless
            options.avd = device_name
            options.avd_launch_timeout = AVD_LAUNCH_TIMEOUT_MS
            options.avd_ready_timeout = AVD_LAUNCH_TIMEOUT_MS
            options.set_capability("appium:enforceAppInstall", not reuse_app)
            if keep_device:
                self._skip_redundant_setup(options)
            self.adb_control = adb_control
            adb_runner = adb_control if adb_control is not None else AdbControl
            package_id, activity = adb_runner.get_apk_info(app_path)
            self.package_id = package_id
            options.app_package = package_id
            options.app_activity = activity

        self.driver = self._start_session(options)
        self.device_info = self._init_device_info()

    def _skip_redundant_setup(self, options) -> None:
        """Reuses the Appium helper apps already on a warm device.

        Session creation spends most of its time reinstalling and restarting
        the UiAutomator2 server and settings app. Skipping that is only safe
        once they are actually present, so check before claiming they are.
        """
        serial = AdbControl.resolve_emulator_serial(self.device_name) or self.device_name
        packages = AdbControl.installed_packages(serial)
        required = {"io.appium.uiautomator2.server", "io.appium.uiautomator2.server.test"}
        if not required.issubset(packages):
            logger.info("Appium server not yet on %s; doing a full device setup.", serial)
            return
        logger.info("Reusing the Appium server already installed on %s.", serial)
        options.set_capability("appium:skipServerInstallation", True)
        if "io.appium.settings" in packages:
            options.set_capability("appium:skipDeviceInitialization", True)

    def _start_session(self, options):
        """Creates the Appium session, translating a failed AVD launch into a reason.

        Appium reports any launch problem as 'Error getting AVD with retry', hiding
        whatever the emulator actually complained about.
        """
        try:
            return webdriver.Remote(self.server_url, options=options)
        except WebDriverException as exc:
            message = str(exc)
            if "getting AVD" not in message and "Condition unmet" not in message:
                raise
            raise DeviceUnavailable(
                f"The emulator '{self.device_name}' did not start within "
                f"{AVD_LAUNCH_TIMEOUT_MS // 1000}s.\n{diagnose_avd_launch(self.device_name)}"
            ) from exc

    def _resolve_serial(self):
        # Try to resolve emulator/device serial from capabilities
        serial = self.driver.capabilities.get(
            "deviceUDID"
        ) or self.driver.capabilities.get("udid")
        # If not in capabilities, fall back to adb resolution on Android
        original_device_name = self.device_name
        self.device_name = self.driver.capabilities.get("deviceName")
        adb_runner = self.adb_control if self.adb_control is not None else AdbControl
        if not serial and self.platform == Platform.ANDROID:
            serial = adb_runner.resolve_emulator_serial(original_device_name)
        return serial

    def _init_device_info(self):

        size = self.driver.get_window_size()
        raw = self.driver.capabilities.get("pixelRatio", 1)
        pixel_ratio = float(str(raw)[:5])
        logger.info(
            "Screen: %sx%s dp | Pixel ratio: %s",
            size["width"],
            size["height"],
            pixel_ratio,
        )
        self.serial = self._resolve_serial()

        self.platform_name = self.driver.capabilities.get("platformName")
        self.os_version = self.driver.capabilities.get("platformVersion")
        self.model = self.driver.capabilities.get("deviceModel")
        self.screen_width = size["width"]
        self.screen_height = size["height"]
        self.pixel_ratio = pixel_ratio
        logger.info(
            "Device connected. Name: %s, Platform: %s, OS Version: %s, Model: %s, Serial: %s",
            self.device_name,
            self.platform_name,
            self.os_version,
            self.model,
            self.serial,
        )

    @property
    def focus(self) -> Focus:
        """Focus checks bound to this session's driver and screen size."""
        area = getattr(self, "screen_width", 0) * getattr(self, "screen_height", 0)
        return Focus(self.driver, area)

    def get_screenshot(self):
        """The screen as PNG bytes, or ObservationUnavailable if unreadable.

        The driver can raise almost anything for a lost session.
        """
        try:
            return self.driver.get_screenshot_as_png()
        except Exception as exc:
            raise ObservationUnavailable(
                f"Could not read the screen of '{self.device_name}': {exc}"
            ) from exc

    def __tap_screen(self, x, y):
        touch = PointerInput("touch", "finger")
        builder = ActionBuilder(self.driver, mouse=touch)
        builder.pointer_action.move_to_location(x, y)
        builder.pointer_action.pointer_down()
        builder.pointer_action.pause(0.1)
        builder.pointer_action.release()
        builder.perform()

    def tap(self, x, y, with_feedback: bool = True):
        if with_feedback:
            self.__tap_screen(x, y)
        else:
            self.driver.tap([(x, y)])





    def type(self, text, x, y):
        if self.platform == Platform.IOS:
            self.tap(x, y)
            time.sleep(1.0)
            self.execute_script("mobile: type", {"text": text})
        else:
            # The planner focuses a field in the preceding step. Avoid tapping it
            # again after the keyboard has reflowed the layout when it is already
            # the active element at the current coordinates.
            before = self.focus.rect()
            if not self.focus.contains(x, y):
                self.tap(x, y)
                self.focus.wait_for(x, y)
            self.focus.assert_at(x, y, "TYPE", before=before)
            self.driver.switch_to.active_element.send_keys(text)

    def clear(self, x, y):
        try:
            before = self.focus.rect()
            self.tap(x, y)
            self.focus.wait_for(x, y)
            if self.platform != Platform.IOS:
                self.focus.assert_at(x, y, "CLEAR", before=before)
            if self.platform == Platform.IOS:
                self.driver.switch_to.active_element.clear()
            else:
                try:
                    # Select All (Ctrl+A) and then Delete (Backspace) to ensure the field is completely cleared
                    self.driver.press_keycode(29, metastate=28672)
                    time.sleep(0.2)
                    self.driver.press_keycode(67)
                    time.sleep(0.2)
                except Exception as e:
                    logger.warning("Failed Ctrl+A clear on Android: %s. Trying fallback.", e)
                    self.driver.switch_to.active_element.clear()
        except Exception as e:
            logger.error("Failed to clear text field at (%s, %s): %s", x, y, e)
            raise

    def double_tap(self, x, y, with_feedback: bool = True):
        if self.platform == Platform.IOS:
            self.execute_script("mobile: doubleTap", {"x": x, "y": y})
        else:
            self.execute_script("mobile: doubleClickGesture", {"x": x, "y": y})
        time.sleep(0.5)

    def long_press(self, x, y, duration):
        if self.platform == Platform.IOS:
            self.execute_script(
                "mobile: touchAndHold", {"x": x, "y": y, "duration": duration}
            )
        else:
            self.execute_script(
                "mobile: longClickGesture",
                {"x": x, "y": y, "duration": int(duration * 1000)},
            )
        time.sleep(duration + 0.5)

    def drag_and_drop(self, source_x, source_y, destination_x, destination_y):
        if self.platform == Platform.IOS:
            self.execute_script(
                "mobile: dragFromToForDuration",
                {
                    "fromX": source_x,
                    "fromY": source_y,
                    "toX": destination_x,
                    "toY": destination_y,
                    "duration": 1.0,
                },
            )
        else:
            self.execute_script(
                "mobile: dragGesture",
                {
                    "startX": source_x,
                    "startY": source_y,
                    "endX": destination_x,
                    "endY": destination_y,
                },
            )
        time.sleep(0.5)

    def key_event(self, key):
        if self.platform == Platform.ANDROID:
            self.driver.press_keycode(key)
        else:
            self.execute_script("mobile: pressButton", {"name": key})
        time.sleep(0.5)

    def swipe(self, start_x, start_y, end_x, end_y, duration_ms: int = 400):
        """Performs a swipe gesture using W3C pointer actions.

        Args:
            start_x: Starting x coordinate.
            start_y: Starting y coordinate.
            end_x: Ending x coordinate.
            end_y: Ending y coordinate.
            duration_ms: Duration of the swipe gesture in milliseconds.
        """
        touch = PointerInput("touch", "finger")
        builder = ActionBuilder(self.driver, mouse=touch)
        builder.pointer_action.move_to_location(start_x, start_y)
        builder.pointer_action.pointer_down()
        builder.pointer_action.pause(duration_ms / 1000)
        builder.pointer_action.move_to_location(end_x, end_y)
        builder.pointer_action.release()
        builder.perform()
        time.sleep(0.5)

    def remove_app(self, package_id):
        self.driver.remove_app(package_id)

    def remove_current_app(self):
        self.remove_app(self.package_id)

    def quit(self):
        try:
            self.driver.quit()
        except Exception as e:
            logger.warning("Failed to quit driver cleanly: %s", e)
        if self.platform == Platform.ANDROID:
            serial = self.serial or self.device_name
            try:
                AdbControl.disable_feedback(serial)
            except Exception as e:
                logger.warning("Failed to disable feedback: %s", e)
            if self.keep_device or getattr(self, "_device_was_running", False):
                logger.info("Leaving '%s' running for the next run.", self.device_name)
                return
            try:
                AdbControl.kill_emulator(self.device_name, serial=self.serial)
            except Exception as e:
                logger.warning("Failed to kill emulator: %s", e)

    def execute_script(self, script, args):
        self.driver.execute_script(script, args if isinstance(args, list) else [args])

    def start_recording(self):
        self.driver.start_recording_screen()

    def stop_recording(self, output_path: str | Path = "session.mp4") -> Path:
        raw = self.driver.stop_recording_screen()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(raw))
        logger.info("Recording saved to %s", path)
        return path

    def _resolve_bundle_id(self) -> str | None:
        """Lazily resolve and cache the iOS bundle ID from driver capabilities."""
        if not self.bundle_id:
            self.bundle_id = (
                self.driver.capabilities.get("bundleId")
                or self.driver.capabilities.get("appium:bundleId")
            )
        return self.bundle_id

    def is_still_in_sut(self) -> bool:
        """Checks if the device is currently focused on the target app under test (SUT)."""
        try:
            if self.platform == Platform.ANDROID:
                return self.driver.current_package == self.package_id
            bundle_id = self._resolve_bundle_id()
            if bundle_id:
                return self.driver.query_app_state(bundle_id) == IOS_APP_STATE_FOREGROUND
        except Exception as e:
            logger.warning("Failed to check SUT focus: %s", e)
        return True

    def is_keyboard_open(self) -> Optional[bool]:
        """Detect whether the soft keyboard is currently visible on screen.

        Tries Appium first (cross-platform), then falls back to a native check:
        - Android: adb dumpsys input_method
        - iOS: idb accessibility tree query
        Returns None if all checks fail.
        """
        try:
            return bool(self.driver.is_keyboard_shown())
        except Exception:
            logger.debug("Appium is_keyboard_shown() unavailable, using native fallback")

        if self.platform == Platform.ANDROID:
            try:
                serial = self.serial or self.device_name
                result = run_command(
                    ["adb", "-s", serial, "shell", "dumpsys", "input_method"],
                    capture_output=True, text=True, timeout=3.0,
                )
                return "mInputShown=true" in (result.stdout or "")
            except Exception as e:
                logger.debug("ADB keyboard check failed: %s", e)

        elif self.platform == Platform.IOS:
            try:
                udid = self.serial or self.device_name
                result = run_command(
                    ["idb", "ui", "describe-all", "--udid", udid],
                    capture_output=True, text=True, timeout=3.0,
                )
                return "XCUIElementTypeKeyboard" in (result.stdout or "")
            except Exception as e:
                logger.debug("idb keyboard check failed: %s", e)

        return None

    def recover_to_sut(self) -> None:
        """Attempts to recover the focus back to the target app under test (SUT) after drift."""
        logger.info(
            "Context drift detected. Attempting to recover focus back to the target app..."
        )
        if self.platform == Platform.ANDROID:
            try:
                self.key_event(ANDROID_KEYCODE_BACK)
                time.sleep(RECOVERY_SLEEP_AFTER_BACK)
            except Exception as e:
                logger.warning("Failed to press back key during recovery: %s", e)

        if self.is_still_in_sut():
            return

        app_id = (
            self.package_id
            if self.platform == Platform.ANDROID
            else self._resolve_bundle_id()
        )

        if not app_id:
            logger.error("Cannot recover: no app ID available.")
            return

        try:
            logger.info("Activating SUT app: %s", app_id)
            self.driver.activate_app(app_id)
            time.sleep(RECOVERY_SLEEP_AFTER_ACTIVATE)
        except Exception as e:
            logger.error("Failed to activate SUT app: %s", e)
