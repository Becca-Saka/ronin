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
Sync all python packages specified in [pyproject.toml](file:///Users/becca/StudioProjects/ai_e2e/pyproject.toml) using `uv`:
```bash
uv sync
```
*Note: This will install Appium Client, LiteLLM, Pillow, Pydantic, NumPy, EasyOCR, and Ultralytics (YOLO).*

### 3. Configure Environment Variables
Create a `.env` file in the root of the project to supply your API credentials:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
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

## Running the E2E Agent

You can execute the automation script to start the session. Update the configuration inside [main.py](file:///Users/becca/StudioProjects/ai_e2e/main.py) to specify your target platform.

```bash
uv run python main.py
```

You can optionally override the log level via the CLI (choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`):
```bash
uv run python main.py --log-level DEBUG
```

### Configuring main.py
In [main.py](file:///Users/becca/StudioProjects/ai_e2e/main.py), configure the `RoninAgent` instantiation to match your target platform and device configuration:

#### Android Configuration:
```python
agent = RoninAgent(
    device_name="Pixel_8a",
    app_path="example/testing_app/build/app/outputs/flutter-apk/app-debug.apk",
    platform=Platform.ANDROID,
)
```

#### iOS Configuration:
```python
agent = RoninAgent(
    device_name="iPhone 15",  # Match your iOS simulator name
    app_path="example/testing_app/build/ios/iphonesimulator/Runner.app",
    platform=Platform.IOS,
)
```

### What Happens Under the Hood
1. **Appium Startup**: The script launches the Appium server programmatically on port `4723`.
2. **Device Connection**: It connects to the specified emulator/simulator and installs/launches the sample app.
3. **Screen Perception**:
   * Takes a screenshot of the current screen.
   * Feeds the image to **YOLOv8** (automatically downloading the `deki-yolo` UI-element detection model from Hugging Face on the first run).
   * Runs **EasyOCR** (which downloads its English language model on the first run) on text/button elements to extract visible labels.
4. **AI Planning**:
   * Annotates the screen image with bounding boxes.
   * Sends the annotated image and element details to **Gemini 2.5 Flash** (via OpenRouter/LiteLLM).
   * The LLM creates a high-level task plan and maps it to coordinates-based UI actions (e.g., tap at `(x, y)`, type text).
5. **Execution**: The orchestrator triggers Appium inputs to perform the planned actions on the device.
6. **Recording**: A screen recording of the session is saved as an `.mp4` file in the `assets/` directory.

