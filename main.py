import os
import sys

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

APPIUM_URL = "http://127.0.0.1:4723"
APP_PACKAGE = os.environ.get("APP_PACKAGE", "com.example.testing_app")
APP_ACTIVITY = ".MainActivity"
TAP_COUNT = 3


def _counter_element(driver: webdriver.Remote):
    return driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().descriptionContains("counter_value")',
    )


def _counter_text(driver: webdriver.Remote) -> str:
    desc = (
        _counter_element(driver).get_attribute("contentDescription") or ""
    ).strip()
    if "," in desc:
        return desc.split(",", 1)[0].strip()
    return desc


def run_test() -> None:
    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.device_name = "Android"
    options.automation_name = "UiAutomator2"
    options.app_package = APP_PACKAGE
    options.app_activity = APP_ACTIVITY
    options.no_reset = True

    driver = webdriver.Remote(APPIUM_URL, options=options)
    wait = WebDriverWait(driver, 15)

    try:
        wait.until(
            EC.presence_of_element_located(
                (
                    AppiumBy.ANDROID_UIAUTOMATOR,
                    'new UiSelector().description("increment_button")',
                )
            )
        )
        assert _counter_text(driver) == "0", f"expected counter 0, got {_counter_text(driver)!r}"

        increment = driver.find_element(
            AppiumBy.ANDROID_UIAUTOMATOR,
            'new UiSelector().description("increment_button")',
        )
        for _ in range(TAP_COUNT):
            increment.click()

        wait.until(lambda d: _counter_text(d) == str(TAP_COUNT))
        assert _counter_text(driver) == str(TAP_COUNT)

        print(f"PASS: counter is {TAP_COUNT} after {TAP_COUNT} taps")
    finally:
        driver.quit()


if __name__ == "__main__":
    try:
        run_test()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
