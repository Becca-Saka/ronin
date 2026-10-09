# Ronin

AI-driven Mobile (Android and iOS) end-to-end testing agent capable of automatically controlling any mobile application.

This agent-based automation reads the device screen visually, plans a sequence of actions based on a user-defined goal, and executes them on the emulator/simulator in real time.

---

## Prerequisites

Ensure you have the following installed on your system:

### 1. System & SDK Dependencies
* **Node.js** (v18+) - Required to run Appium.
* **Android SDK** (for Android testing) - Install via Android Studio.
  * Set `ANDROID_HOME` in your environment (e.g., `export ANDROID_HOME=~/Library/Android/sdk`).
  * Ensure emulator and platform-tools are in your `PATH`:
    ```bash
    export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"
    ```
* **Xcode & macOS** (for iOS testing) - Required to build and run iOS simulators/devices.
  * Install Xcode from the App Store and ensure Xcode command-line tools are installed:
    ```bash
    xcode-select --install
    ```

### 2. Python Environment & Devices
* **Python 3.12+** - Managed via [uv](https://docs.astral.sh/uv/).
* **Android Virtual Device (AVD)** - For Android, a running or configured emulator (default: `Pixel_8a`).
  * To check your available AVDs:
    ```bash
    emulator -list-avds
    ```
* **iOS Simulator** - For iOS, a configured simulator (e.g., `iPhone 15` or your preferred simulator model).
  * To check available iOS simulator devices:
    ```bash
    xcrun simctl list devices
    ```

---

## One-Time Setup

Follow these steps to set up the project:

### 1. Install Appium & Mobile Drivers
Install Appium globally using npm, and install the drivers for both Android and iOS:
```bash
npm install -g appium
appium driver install uiautomator2
appium driver install xcuitest
```
Verify that Appium is installed successfully:
```bash
appium --version
appium driver list --installed
```

### 2. Install Python Dependencies
Sync all python packages specified in [pyproject.toml](pyproject.toml) using `uv`:
```bash
uv sync --extra vision
```
This installs the `ronin` command along with Appium Client, LiteLLM, Pillow, Pydantic, and NumPy, plus the `vision` extra (EasyOCR and Ultralytics/YOLO).

The `vision` extra is separate because it pulls in PyTorch — 127MB on Apple Silicon, 555MB on Linux x86_64. Plain `uv sync` gives you a much faster install that can do everything except screen perception; Ronin will tell you to add the extra if you try to run without it.

Check that the command is available:
```bash
uv run ronin --version
uv run ronin --help
```

### 3. Configure Environment Variables
Create a `.env` file in the root of the project to supply your API credentials:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Verify Your Environment
`ronin doctor` checks every prerequisite above and prints the command to fix anything missing:
```bash
uv run ronin doctor
```
```
Ronin
  ✓ python: 3.12.9
  ✓ vision deps: ultralytics, easyocr, huggingface_hub
  ✓ api key: OPENROUTER_API_KEY set

Appium
  ✓ node: v24.13.0
  ✓ appium: 3.5.0
  ✓ driver: uiautomator2: installed
  ✗ driver: xcuitest: not installed
      ronin setup   (or: appium driver install xcuitest)
```
It exits non-zero when something blocking is missing, so it works as a CI gate. Checks for a platform you don't target are reported as warnings rather than failures:
```bash
uv run ronin doctor --platform android   # skip the iOS checks
uv run ronin doctor --json               # machine-readable output
```

---

## Building the Sample App (Example)

*Note: Building this example requires the **Flutter SDK** to be installed.*

Before running the E2E script, build the Flutter app for your target platform:

### For Android:
```bash
cd example/testing_app
flutter build apk --debug
cd ../..
```
This produces the debug APK at `example/testing_app/build/app/outputs/flutter-apk/app-debug.apk`.

### For iOS Simulator:
```bash
cd example/testing_app
flutter build ios --simulator
cd ../..
```
This produces the simulator app bundle at `example/testing_app/build/ios/iphonesimulator/Runner.app`.

---

## Mode 1: Journey-Based QA Testing (Recommended)

Ronin's primary mode is **User Journey Testing** — you describe a user flow in plain English inside a YAML file, and Ronin automatically generates a full suite of test scenarios, executes them on the device, and produces structured reports.

### 1. Write a Journey File

Create a YAML file describing your user journey (see [`example/checkout_journey.yaml`](example/checkout_journey.yaml)):

```yaml
id: notification_journey_01
name: E2E Notification Detail and Switch Toggle Journey

journey: >
  A user opens the app and lands on the sign-in screen.
  They type 'test@email.com' in the Email field and 'secret-password123' in the Password field.
  They click the Login button. Once logged in, they click on the first notification item.
  On the detail screen, they type 'Awesome notification!' in the reply field,
  toggle the 'Pin notification' switch to true, and click the Submit button.

security_rules:
  - "The logged-in user's password must never be exposed in plain text on the Home or Dashboard screens."
  - "The reply text input must not leak password characters or unmasked sensitive strings."
```

### 2. Run the Test Journey

```bash
uv run ronin test example/checkout_journey.yaml
```

To use a more capable primary model alongside a faster/cheaper model for execution steps:
```bash
uv run ronin test example/checkout_journey.yaml \
  --model openrouter/google/gemini-2.5-pro \
  --fast-model openrouter/google/gemini-2.5-flash
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `openrouter/google/gemini-2.5-flash` | Primary LiteLLM model for scenario generation and auditing |
| `--fast-model` | *(same as model)* | Cheaper/faster model for per-step ReAct planning |
| `--device` | `Pixel_8a` | Emulator or simulator name |
| `--platform` | `android` | Target platform (`android` or `ios`) |
| `--app-path` | *(flutter debug apk)* | Path to app binary (`.apk` or `.app`) |
| `--max-steps` | `20` | Maximum ReAct steps per scenario |
| `--settle-time` | `2.0` | Seconds to wait after each action |
| `--output-dir` | `assets/` | Directory to save reports and screenshots |
| `--use-multimodal-vision` | `False` | Allow LLM to override YOLO element coordinates |
| `--headless` | `False` | Run emulator headlessly |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### What Happens Under the Hood

1. **Journey Parsing**: The YAML file is loaded and parsed into a `UserJourney` object.
2. **Scenario Generation**: An LLM decomposes the journey description into a suite of diverse test scenarios covering happy paths, edge cases, boundary tests, destructive monkey flows, security compliance, accessibility, and performance.
3. **Device Setup**: Appium starts, connects to the emulator/simulator, and installs the app.
4. **ReAct Execution Loop**: For each scenario, Ronin runs a closed-loop cycle:
   * **Perceive** — Screenshots the device screen; runs YOLOv8 to detect UI elements and EasyOCR to read text.
   * **Audit** — The LLM generates context-specific screen assertions (e.g. "is the password field masked?") and verifies them alongside security rules.
   * **Plan** — The LLM decides the next action to perform (tap, type, swipe, etc.).
   * **Act** — The orchestrator executes the action on the device via Appium.
   * **Continue** — Failures are recorded but the journey continues to the next screen. Only a hard app crash terminates early.
5. **App Reset**: The app is restarted between each scenario for a clean state.

### Outputs

All outputs are saved to the `assets/` directory (configurable via `--output-dir`):

| File | Description |
|---|---|
| `report_<id>_<timestamp>.md` | Human-readable Markdown report with per-screen passed/failed verifications and detected issues |
| `junit_<id>_<timestamp>.xml` | Standard JUnit XML for CI systems (Jenkins, GitHub Actions, GitLab CI) |
| `llm_usage_<timestamp>.json` | LLM token usage and estimated cost breakdown per model |
| `scen_<id>_step<n>_*.png` | Annotated screenshots for each execution step |

---

## Mode 2: Single-Goal Agent

For quick one-off automation tasks, pass the goal directly on the command line:

```bash
uv run ronin run "Sign in with email 'test@email.com' and navigate to the settings screen." \
  --app-path example/testing_app/build/app/outputs/flutter-apk/app-debug.apk
```

`ronin run` accepts the same device and model flags as `ronin test` (`--device`, `--platform`, `--model`, `--fast-model`, `--max-steps`, `--settle-time`, `--headless`, `--output-dir`), plus a required `--app-path`. It exits `0` when the goal is reached and `1` otherwise.

You can override the log level on any command:
```bash
uv run ronin --log-level DEBUG run "..." --app-path ...
```

> **Note**: This mode does not generate test scenarios, produce reports, or perform QA auditing. Use `ronin test` for structured testing.

> `uv run main.py` still works but is deprecated — it forwards to `ronin`, mapping the old `test-journey` subcommand to `test`.

