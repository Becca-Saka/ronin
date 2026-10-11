"""Default values shared by every entrypoint.

Single source of truth for the flag defaults, so the CLI, the testing CLI and
`ronin doctor` cannot drift apart on what "default" means.
"""

DEFAULT_PLATFORM = "android"
DEFAULT_DEVICE = "Pixel_8a"
DEFAULT_MODEL = "openrouter/google/gemini-2.5-flash"
DEFAULT_OUTPUT_DIR = "assets"
DEFAULT_MAX_STEPS = 20
DEFAULT_SETTLE_TIME = 2.0
DEFAULT_APP_PATH = "example/testing_app/build/app/outputs/flutter-apk/app-debug.apk"

# A cold emulator boot regularly exceeds Appium's 60s default.
AVD_LAUNCH_TIMEOUT_MS = 180_000

# Focus is polled after a tap rather than slept on.
FOCUS_POLL_INTERVAL = 0.1
FOCUS_POLL_TIMEOUT = 1.5
