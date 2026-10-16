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
env:
  email: test@example.com
  password:
    value: ${RONIN_TEST_PASSWORD}
    secret: true
journey: >
  Sign in with {{email}} and {{password}}, then open the settings screen.
security_rules:
  - The user's password must never be shown in plain text.
```

`env` declares the values a journey uses and `{{name}}` inserts them into its
text. `${VAR}` reads from the environment, so a real credential never has to be
committed. Write a plain value for ordinary data, or give it a `value` and
`secret: true` to mark it sensitive — a top-level `secrets:` list of names works
too. A `${VAR}` that is not set fails the journey rather than running as a
blank value.

Values declared as secret are masked by name in the Markdown and JUnit reports,
the progress file, the logs and the perception scan dumps — all of which show
`{{password}}` — and the screen regions holding them are blacked out in saved
screenshots. Screen recordings are **not** redacted; turn recording off for a
flow whose secrets must not be captured. The agent itself always receives the real values:
it cannot type a password it was not given, and it cannot audit a rule like "the
card number must never be shown in plain text" if the card number is hidden from
it. Both blocks are optional.

Run it with:

```bash
uv run ronin test path/to/journey.yaml --app-path path/to/app.apk
```

See [`example/checkout_journey.yaml`](example/checkout_journey.yaml) for a complete example. Journey files may be YAML or JSON and may contain one journey or a suite of journeys.

Ronin writes Markdown and JUnit reports, LLM usage data, and annotated screenshots to `assets/` by default.

Generated suites are capped at 10 scenarios. Ronin only generates assertions it
can currently evidence from screenshots, detected UI text, action outcomes, and
elapsed execution. Network traffic, HTTP headers, semantic accessibility trees,
screen readers, focus order, and measured contrast ratios require separate
specialist tooling and are reported as unsupported if supplied manually.

Scenario results distinguish app failures from harness and evidence failures:
`FAILED` means an observed app assertion failed; `ERROR` and `ABORTED` identify
infrastructure or interruption; `UNVERIFIED` means required proof was missing;
and `UNSUPPORTED` means the runtime lacks the requested evidence source. Only
`PASSED` counts as verified success. Progress, Markdown, JUnit, and usage files
are written atomically, with partial Markdown/JUnit checkpoints maintained
during a run. Values a journey declares as secret are masked by name in
persisted text and covered in saved screenshots, except screen recordings.

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

## Development

Install the tooling, then run the checks that CI runs:

```bash
uv sync --group dev
uv run pytest
uv run ruff check
uv run python tools/quality_gate.py
uv run python tools/type_gate.py
uv run coverage run -m pytest && uv run coverage report
```

`tools/quality_gate.py` enforces the file-size limit and needs no dependencies.
Complexity is ruff's `C901`. `tools/type_gate.py` runs Pyright and compares the
per-file error counts against its baseline.

Three places record the existing debt: `tools/quality-baseline.toml`,
`tools/type-baseline.toml`, and `[tool.ruff.lint.per-file-ignores]`. All three
are ratchets. Lower an entry when a file improves. Do not add new ones.
