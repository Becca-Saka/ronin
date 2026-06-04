# ai_e2e

Android end-to-end testing with Appium, UiAutomator2, and a Flutter sample app.

## Prerequisites

- Node.js (Appium)
- Android SDK (`ANDROID_HOME`, e.g. `~/Library/Android/sdk`)
- Flutter SDK
- [uv](https://docs.astral.sh/uv/) for Python
- An AVD (default: `Pixel_8a`)

Put the SDK emulator on `PATH` before deprecated `sdk/tools`:

```bash
export ANDROID_HOME=~/Library/Android/sdk
export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"
```

## One-time setup

```bash
npm install -g appium
appium driver install uiautomator2

uv sync
```

## Run the full flow

Starts the emulator (if needed), builds and installs the Flutter APK, starts Appium (if needed), runs the Python test, then uninstalls the app. If this script started the emulator, it stops it when finished (including on test failure):

```bash
chmod +x run_e2e.sh
./run_e2e.sh
```

Override the AVD name:

```bash
AVD=Pixel_8a ./run_e2e.sh
```

## Manual steps

**Emulator**

```bash
emulator -list-avds
emulator -avd Pixel_8a
```

**Build and install the sample app**

```bash
cd example/testing_app
flutter build apk --debug
adb install -r build/app/outputs/flutter-apk/app-debug.apk
adb shell pm list packages | grep testing_app
```

**Appium and test**

```bash
appium
uv run python main.py
```

## Testable UI

The sample app (`example/testing_app`) exposes semantics labels for Appium:

| Label | Role |
|-------|------|
| `increment_button` | FAB tap target |
| `counter_value` | Counter display |
| `e2e_app_title` | App bar title |

## Python only

With emulator, Appium, and the app already running:

```bash
uv run python main.py
```
