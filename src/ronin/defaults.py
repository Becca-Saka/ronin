"""Default values shared by every entrypoint.

Single source of truth for the flag defaults, so the CLI, the testing CLI and
`ronin doctor` cannot drift apart on what "default" means.
"""

DEFAULT_PLATFORM = "android"
DEFAULT_DEVICE = "Pixel_8a"
DEFAULT_MODEL = "openrouter/google/gemini-2.5-flash"
DEFAULT_OUTPUT_DIR = "assets"
# The ceiling for any one scenario; each scenario is given what its own
# instructions suggest it needs, below this.
DEFAULT_MAX_STEPS = 40
STEP_BUDGET_FLOOR = 15
STEP_BUDGET_PER_INSTRUCTION = 2
DEFAULT_SETTLE_TIME = 2.0
DEFAULT_APP_PATH = None

# A cold emulator boot regularly exceeds Appium's 60s default.
AVD_LAUNCH_TIMEOUT_MS = 180_000

# Settling waits for the screen to stop changing rather than sleeping blind.
SETTLE_POLL_INTERVAL = 0.35
SETTLE_STABLE_READINGS = 2
FOCUS_POLL_INTERVAL = 0.1
FOCUS_POLL_TIMEOUT = 1.5
