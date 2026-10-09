"""Individual environment checks behind `ronin doctor`.

Each check is read-only and returns a CheckResult rather than raising, so a
broken environment produces a full report instead of stopping at the first
problem. Anything that can be fixed by running a command carries that
command as its hint.
"""

import io
import os
import platform as platform_module
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ronin import __version__
from ronin.defaults import DEFAULT_MODEL
from ronin.utils.subprocess_runner import run_command
from ronin.utils.vision_deps import VISION_PACKAGES, missing_vision_packages

MIN_NODE_MAJOR = 18
MIN_PYTHON = (3, 12)

ANDROID_SDK_ENV_VARS = ("ANDROID_HOME", "ANDROID_SDK_ROOT")
MACOS_DEFAULT_SDK = Path.home() / "Library" / "Android" / "sdk"

PATH_EXPORT_HINT = 'export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"'


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Outcome of a single environment check."""

    name: str
    status: Status
    detail: str = ""
    hint: Optional[str] = None


@dataclass
class CheckSection:
    """A named group of related checks."""

    title: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARN]


def is_macos() -> bool:
    return platform_module.system() == "Darwin"


def _capture(cmd: list[str], timeout: float = 20.0) -> Optional[subprocess.CompletedProcess]:
    """Runs a command quietly, returning None when the binary is missing or hangs."""
    try:
        return run_command(cmd, timeout=timeout, log_command=False)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None


def _streams(result: subprocess.CompletedProcess) -> str:
    """Appium and friends split output across stdout and stderr; search both."""
    return f"{result.stdout or ''}\n{result.stderr or ''}"


def check_python() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] >= MIN_PYTHON:
        return CheckResult("python", Status.OK, version)
    return CheckResult(
        "python",
        Status.FAIL,
        f"{version} is too old",
        f"Ronin needs Python {'.'.join(map(str, MIN_PYTHON))} or newer",
    )


def check_ronin() -> CheckResult:
    location = Path(__file__).resolve().parents[1]
    return CheckResult("ronin", Status.OK, f"{__version__} ({location})")


def check_vision_deps() -> CheckResult:
    missing = missing_vision_packages()
    if not missing:
        return CheckResult("vision deps", Status.OK, ", ".join(VISION_PACKAGES))
    return CheckResult(
        "vision deps",
        Status.FAIL,
        f"missing: {', '.join(missing)}",
        "ronin setup   (or: uv sync --extra vision)",
    )


def check_credentials(model: str = DEFAULT_MODEL) -> CheckResult:
    """Asks LiteLLM what the default model needs, and whether the environment has it.

    Deferring to LiteLLM keeps this honest when DEFAULT_MODEL changes provider,
    and covers providers needing more than a single key.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - litellm is a base dependency
        return CheckResult("credentials", Status.WARN, f"could not load litellm: {exc}")

    # validate_environment prints a provider banner for models it cannot route.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        missing = litellm.validate_environment(model).get("missing_keys") or []

    if not missing:
        return CheckResult("credentials", Status.OK, f"{model} is configured")

    hint = f"Add {'='.join([missing[0], '...'])} to a .env file in this directory"
    if missing[0] == "OPENROUTER_API_KEY":
        hint += " (https://openrouter.ai/keys)"
    return CheckResult(
        "credentials",
        Status.FAIL,
        f"{model} needs {', '.join(missing)}",
        hint,
    )


def check_node() -> CheckResult:
    if not shutil.which("node"):
        return CheckResult("node", Status.FAIL, "not installed", "brew install node")
    result = _capture(["node", "--version"])
    raw = (result.stdout or "").strip() if result else ""
    try:
        major = int(raw.lstrip("v").split(".")[0])
    except (ValueError, IndexError):
        return CheckResult("node", Status.WARN, f"unrecognised version: {raw or 'unknown'}")
    if major >= MIN_NODE_MAJOR:
        return CheckResult("node", Status.OK, raw)
    return CheckResult(
        "node",
        Status.FAIL,
        f"{raw} is too old (need v{MIN_NODE_MAJOR}+)",
        "brew install node",
    )


def check_appium() -> CheckResult:
    if not shutil.which("appium"):
        return CheckResult(
            "appium", Status.FAIL, "not installed", "ronin setup   (or: npm install -g appium)"
        )
    result = _capture(["appium", "--version"])
    version = (result.stdout or "").strip() if result else ""
    return CheckResult("appium", Status.OK, version or "installed")


def check_appium_driver(driver: str, needed: bool = True) -> CheckResult:
    """Checks one Appium driver. Drivers for platforms you don't target are informational."""
    name = f"driver: {driver}"
    if not shutil.which("appium"):
        return CheckResult(name, Status.SKIP, "appium not installed")
    result = _capture(["appium", "driver", "list", "--installed"])
    if result is None:
        return CheckResult(name, Status.WARN, "could not list drivers")
    if driver in _streams(result):
        return CheckResult(name, Status.OK, "installed")
    return CheckResult(
        name,
        Status.FAIL if needed else Status.WARN,
        "not installed",
        f"ronin setup   (or: appium driver install {driver})",
    )


def android_sdk_path() -> Optional[Path]:
    """Resolves the Android SDK from the environment, falling back to the macOS default."""
    for var in ANDROID_SDK_ENV_VARS:
        value = os.getenv(var)
        if value and Path(value).is_dir():
            return Path(value)
    if is_macos() and MACOS_DEFAULT_SDK.is_dir():
        return MACOS_DEFAULT_SDK
    return None


def check_android_sdk() -> CheckResult:
    sdk = android_sdk_path()
    if sdk is None:
        return CheckResult(
            "android sdk",
            Status.FAIL,
            "not found",
            "Install it via Android Studio, then: export ANDROID_HOME=$HOME/Library/Android/sdk",
        )
    if not any(os.getenv(var) for var in ANDROID_SDK_ENV_VARS):
        return CheckResult(
            "android sdk",
            Status.WARN,
            f"found at {sdk}, but ANDROID_HOME is not set",
            f'export ANDROID_HOME="{sdk}"',
        )
    return CheckResult("android sdk", Status.OK, str(sdk))


def _sdk_binary(sdk: Path, name: str) -> Optional[Path]:
    for relative in (f"platform-tools/{name}", f"emulator/{name}", f"tools/bin/{name}"):
        candidate = sdk / relative
        if candidate.exists():
            return candidate
    return None


def resolve_android_binary(name: str) -> Optional[Path]:
    """Finds an SDK binary on PATH, or inside the SDK as a fallback."""
    found = shutil.which(name)
    if found:
        return Path(found)
    sdk = android_sdk_path()
    return _sdk_binary(sdk, name) if sdk else None


def check_android_binary(name: str) -> CheckResult:
    if shutil.which(name):
        return CheckResult(name, Status.OK, "on PATH")
    sdk = android_sdk_path()
    if sdk and _sdk_binary(sdk, name):
        return CheckResult(
            name,
            Status.WARN,
            "in the SDK but not on PATH",
            PATH_EXPORT_HINT,
        )
    return CheckResult(
        name,
        Status.FAIL,
        "not found",
        "Install 'platform-tools' and 'emulator' from the Android Studio SDK Manager",
    )


def check_aapt() -> CheckResult:
    """aapt reads the package id and launch activity out of the APK under test."""
    sdk = android_sdk_path()
    if sdk is None:
        return CheckResult("aapt", Status.SKIP, "no android sdk")
    build_tools = sdk / "build-tools"
    if build_tools.is_dir():
        for candidate in sorted(build_tools.glob("*/aapt"), reverse=True):
            return CheckResult("aapt", Status.OK, str(candidate))
    return CheckResult(
        "aapt",
        Status.FAIL,
        "not found in build-tools",
        "Install any 'Android SDK Build-Tools' version from the SDK Manager",
    )


def check_avds(expected_device: Optional[str] = None) -> CheckResult:
    """Lists configured AVDs, warning when the device Ronin would target is absent."""
    from ronin.config import load_config

    expected = expected_device or load_config().get("device")
    emulator = resolve_android_binary("emulator")
    if emulator is None:
        return CheckResult("avds", Status.SKIP, "emulator not found")
    result = _capture([str(emulator), "-list-avds"])
    if result is None:
        return CheckResult("avds", Status.WARN, "could not list AVDs")
    avds = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if avds and expected not in avds:
        return CheckResult(
            "avds",
            Status.WARN,
            f"{', '.join(avds)} (no '{expected}')",
            f"Ronin targets '{expected}'; run: ronin config set device {avds[0]}",
        )
    if avds:
        return CheckResult("avds", Status.OK, ", ".join(avds))
    return CheckResult(
        "avds",
        Status.FAIL,
        "none configured",
        f"Create one with: ronin devices create {expected}",
    )


def check_xcode() -> CheckResult:
    if not is_macos():
        return CheckResult("xcode", Status.SKIP, "iOS automation requires macOS")
    result = _capture(["xcode-select", "-p"])
    path = (result.stdout or "").strip() if result else ""
    if result is not None and result.returncode == 0 and path:
        return CheckResult("xcode", Status.OK, path)
    return CheckResult(
        "xcode",
        Status.FAIL,
        "command-line tools not installed",
        "xcode-select --install",
    )


def check_simulators() -> CheckResult:
    if not is_macos():
        return CheckResult("simulators", Status.SKIP, "iOS automation requires macOS")
    result = _capture(["xcrun", "simctl", "list", "devices", "available"])
    if result is None:
        return CheckResult("simulators", Status.SKIP, "xcrun unavailable")
    iphones = [
        line.strip()
        for line in (result.stdout or "").splitlines()
        if "iPhone" in line and "unavailable" not in line
    ]
    if iphones:
        return CheckResult("simulators", Status.OK, f"{len(iphones)} iPhone simulator(s) available")
    return CheckResult(
        "simulators",
        Status.FAIL,
        "no iPhone simulators available",
        "Install a simulator runtime: Xcode > Settings > Components",
    )


def run_checks(platforms: tuple[str, ...] = ("android", "ios")) -> list[CheckSection]:
    """Runs every check for the requested platforms and groups the results."""
    sections = [
        CheckSection(
            "Ronin",
            [check_python(), check_ronin(), check_vision_deps(), check_credentials()],
        )
    ]

    appium = CheckSection("Appium", [check_node(), check_appium()])
    appium.results.append(check_appium_driver("uiautomator2", needed="android" in platforms))
    if is_macos():
        appium.results.append(check_appium_driver("xcuitest", needed="ios" in platforms))
    sections.append(appium)

    if "android" in platforms:
        sections.append(
            CheckSection(
                "Android",
                [
                    check_android_sdk(),
                    check_android_binary("adb"),
                    check_android_binary("emulator"),
                    check_aapt(),
                    check_avds(),
                ],
            )
        )

    if "ios" in platforms:
        sections.append(CheckSection("iOS", [check_xcode(), check_simulators()]))

    return sections
