"""
High-level control interface for interacting with mobile devices (Android/iOS) via Appium.
Encapsulates driver initialization, screenshots, and gesture/input execution.
"""

import base64
import logging
import time
from pathlib import Path

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.options.ios import XCUITestOptions
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput

from device.adb_control import AdbControl
from device.platform import Platform

logger = logging.getLogger(__name__)


class DeviceControl:
    """Wrapper for executing gesture actions and capturing state from mobile devices."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        adb_control: AdbControl,
        is_headless: bool = False,
        platform: Platform = Platform.ANDROID,
    ):
        self.platform = platform
        if platform == Platform.IOS:
            options = XCUITestOptions()
            options.platform_name = "iOS"
            options.automation_name = "XCUITest"
            options.device_name = device_name
            options.app = app_path
            options.is_headless = is_headless
            self.adb_control = None
        else:
            options = UiAutomator2Options()
            options.app = app_path
            options.platform_name = "Android"
            options.device_name = device_name
            options.is_headless = is_headless
            options.avd = device_name
            options.avd_launch_timeout = 60000
            options.avd_ready_timeout = 60000
            self.adb_control = adb_control
            package_id, activity = self.adb_control.get_apk_info(app_path)
            self.package_id = package_id
            options.app_package = package_id
            options.app_activity = activity

        self.driver = webdriver.Remote("http://localhost:4723", options=options)
        self.device_info = self._init_device_info()

    def _init_device_info(self):

        size = self.driver.get_window_size()
        raw = self.driver.capabilities.get("pixelRatio", 1)
        pixel_ratio = float(str(raw)[:5])
        logger.info(
            "Screen: %sx%s dp | Pixel ratio: %s", size['width'], size['height'], pixel_ratio
        )
        self.device_name = self.driver.capabilities.get("deviceName")
        self.platform_name = self.driver.capabilities.get("platformName")
        self.os_version = self.driver.capabilities.get("platformVersion")
        self.model = self.driver.capabilities.get("deviceModel")
        self.screen_width = size["width"]
        self.screen_height = size["height"]
        self.pixel_ratio = pixel_ratio
        logger.info(
            "Device connected. Name: %s, Platform: %s, OS Version: %s, Model: %s",
            self.device_name, self.platform_name, self.os_version, self.model
        )

    def get_screenshot(self):
        return self.driver.get_screenshot_as_png()

    def __tap_screen(self, x, y):
        try:
            touch = PointerInput("touch", "finger")
            builder = ActionBuilder(self.driver, mouse=touch)
            builder.pointer_action.move_to_location(x, y)
            builder.pointer_action.pointer_down()
            builder.pointer_action.pause(0.1)
            builder.pointer_action.release()
            builder.perform()
        except Exception as e:
            logger.error("Failed to tap at (%s, %s): %s", x, y, e)
            return False
        return True

    def tap(self, x, y, with_feedback: bool = True):
        if with_feedback:
            self.__tap_screen(x, y)
        else:
            self.driver.tap([(x, y)])

    def type(self, text, x, y):
        self.tap(x, y)
        time.sleep(1.0)
        if self.platform == Platform.IOS:
            self.execute_script("mobile: type", {"text": text})
        else:
            self.driver.switch_to.active_element.send_keys(text)
        time.sleep(0.5)

    def double_tap(self, x, y, with_feedback: bool = True):
        self.tap(x, y, with_feedback=with_feedback)
        time.sleep(0.2)
        self.tap(x, y, with_feedback=with_feedback)
        time.sleep(0.2)

    def long_press(self, x, y, duration):
        self.execute_script(
            "mobile: longPress", {"x": x, "y": y, "duration": duration * 1000}
        )
        time.sleep(duration)

    def drag_and_drop(self, source_x, source_y, destination_x, destination_y):
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
        self.execute_script("mobile: pressKey", {"keycode": key})
        time.sleep(0.5)

    def swipe(self, start_x, start_y, end_x, end_y):
        self.execute_script(
            "mobile: swipeGesture",
            {
                "left": start_x,
                "top": start_y,
                "width": abs(end_x - start_x),
                "height": abs(end_y - start_y),
            },
        )
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
        if self.platform == Platform.ANDROID and self.adb_control:
            try:
                self.adb_control.disable_feedback()
            except Exception as e:
                logger.warning("Failed to disable feedback: %s", e)
            try:
                self.adb_control.kill_emulator(self.device_name)
            except Exception as e:
                logger.warning("Failed to kill emulator: %s", e)

    def execute_script(self, script, args):
        self.driver.execute_script(script, args)

    def start_recording(self):
        self.driver.start_recording_screen()

    def stop_recording(self, output_path: str | Path = "session.mp4") -> Path:
        raw = self.driver.stop_recording_screen()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(raw))
        logger.info("Recording saved to %s", path)
        return path
