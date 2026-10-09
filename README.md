# Ronin

Ronin is an AI-powered end-to-end testing agent for Android and iOS apps. Give it a user journey or a single goal and it will inspect the screen, plan actions, control an emulator or simulator through Appium, and verify the result.

> Ronin is currently alpha software.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12+
- Node.js 18+ (required by Appium)
- Android Studio and the Android SDK for Android testing
- Xcode and its command-line tools for iOS testing (macOS only)

The Flutter SDK is only needed to build the included sample app.

## Quick start

Clone the repository and install Ronin with its vision dependencies:

```bash
git clone https://github.com/Becca-Saka/ronin.git
cd ronin
uv sync --extra vision
```

Run the guided setup, then verify your environment:

```bash
uv run ronin setup
uv run ronin doctor
uv run ronin devices
```

`ronin setup` installs Appium, the driver for your configured platform, and the required model files. Use `ronin setup --dry-run` to preview its changes or `ronin setup -y` for non-interactive setup.

Add the API key required by your chosen [LiteLLM-supported model](https://docs.litellm.ai/docs/providers) to a `.env` file or your shell environment. The default model uses OpenRouter:

```env
OPENROUTER_API_KEY=your_key_here
```

## Test a user journey

A journey file describes the flow to test and any security rules Ronin should enforce:

```yaml
id: sign_in
name: Sign in and open settings
journey: >
  Sign in with test@example.com and open the settings screen.
security_rules:
  - The user's password must never be shown in plain text.
```

Run it with:

```bash
uv run ronin test path/to/journey.yaml --app-path path/to/app.apk
```

See [`example/checkout_journey.yaml`](example/checkout_journey.yaml) for a complete example. Journey files may be YAML or JSON and may contain one journey or a suite of journeys.

Ronin writes Markdown and JUnit reports, LLM usage data, and annotated screenshots to `assets/` by default.

## Run a single goal

For one-off automation without generated test scenarios or QA reports:

```bash
uv run ronin run "Sign in and open the settings screen" \
  --app-path path/to/app.apk
```

Android APKs and iOS `.app` or `.ipa` bundles are supported.

## Build the sample app

The example journey targets the Flutter app in `example/testing_app`.

Android:

```bash
cd example/testing_app
flutter build apk --debug
```

iOS Simulator:

```bash
cd example/testing_app
flutter build ios --simulator
```

Return to the repository root before running Ronin. The generated app paths are:

- Android: `example/testing_app/build/app/outputs/flutter-apk/app-debug.apk`
- iOS: `example/testing_app/build/ios/iphonesimulator/Runner.app`

## Configuration

Inspect or change the saved configuration with:

```bash
uv run ronin config
uv run ronin config set device Pixel_8a
uv run ronin config set model openrouter/google/gemini-2.5-pro --project
```

Configuration is resolved in this order: CLI flags, `RONIN_*` environment variables, the nearest `ronin.yaml`, `~/.ronin/config.yaml`, then built-in defaults. Use `--project` to save a setting in `./ronin.yaml` instead of the global config.

## Commands

| Command | Purpose |
|---|---|
| `ronin setup` | Install and configure runtime dependencies |
| `ronin doctor` | Check prerequisites and report fixes |
| `ronin devices` | List, create, and select devices |
| `ronin config` | View or change settings |
| `ronin test <file>` | Generate and run tests from a journey file |
| `ronin run <goal>` | Drive an app toward a single goal |

Run `uv run ronin <command> --help` for all options. Add `--log-level DEBUG` before the command for detailed logs.
