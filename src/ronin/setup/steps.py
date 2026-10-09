"""The individual things `ronin setup` installs or creates.

Steps mutate the machine, so each one asks before acting, reports what it did
as a StepResult, and never raises — a failed step leaves a message behind and
the remaining steps still run.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ronin.agent.perception.image_descriptor import (
    OCR_LANGUAGES,
    YOLO_FILENAME,
    YOLO_REPO_ID,
)
from ronin.cli.console import FAIL, INFO, OK, SKIP, WARN
from ronin.defaults import DEFAULT_MODEL
from ronin.doctor import Status, check_appium_driver
from ronin.utils.subprocess_runner import run_command
from ronin.utils.vision_deps import vision_available

ENV_FILENAME = ".env"

ENV_TEMPLATE = f"""\
# Credentials for the models Ronin calls through LiteLLM.
# Ronin's default model is {DEFAULT_MODEL} — get a key at https://openrouter.ai/keys
OPENROUTER_API_KEY=

# Other providers work too; pass --model to use them.
# GEMINI_API_KEY=
# ANTHROPIC_API_KEY=
"""

PREFETCH_SCRIPT = f"""\
from huggingface_hub import hf_hub_download
import easyocr

hf_hub_download(repo_id={YOLO_REPO_ID!r}, filename={YOLO_FILENAME!r})
easyocr.Reader({OCR_LANGUAGES!r}, gpu=False)
"""


@dataclass
class StepResult:
    """What a step did, in the console's reporting vocabulary."""

    level: str
    label: str
    detail: str = ""
    hint: Optional[str] = None


@dataclass
class SetupContext:
    """Policy and callbacks a step needs, kept free of any rendering concern."""

    assume_yes: bool = False
    dry_run: bool = False
    confirm: Callable[[str], bool] = field(default=lambda _prompt: True)
    log: Callable[[str], None] = field(default=lambda _message: None)

    def approved(self, prompt: str) -> bool:
        return True if self.assume_yes else self.confirm(prompt)


def _run_visible(cmd: list[str], cwd: Optional[Path] = None) -> int:
    """Runs a long install command with its output left on screen."""
    try:
        result = run_command(
            cmd,
            capture_output=False,
            text=False,
            log_command=False,
            **({"cwd": str(cwd)} if cwd else {}),
        )
        return result.returncode
    except TypeError:
        # run_command does not forward cwd; fall back to subprocess directly.
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1


def _declined(label: str, command: list[str]) -> StepResult:
    return StepResult(WARN, label, "skipped", f"Run it yourself: {' '.join(command)}")


def _planned(label: str, command: list[str]) -> StepResult:
    return StepResult(INFO, label, f"would run: {' '.join(command)}")


def ensure_env_file(ctx: SetupContext, directory: Optional[Path] = None) -> StepResult:
    """Creates a .env template in the working directory when none exists."""
    target = (directory or Path.cwd()) / ENV_FILENAME
    if target.exists():
        return StepResult(OK, ENV_FILENAME, f"already present at {target}")
    if ctx.dry_run:
        return StepResult(INFO, ENV_FILENAME, f"would create {target}")
    try:
        target.write_text(ENV_TEMPLATE, encoding="utf-8")
    except OSError as exc:
        return StepResult(FAIL, ENV_FILENAME, f"could not write {target}: {exc}")
    return StepResult(
        WARN,
        ENV_FILENAME,
        f"created {target}",
        "Add your OPENROUTER_API_KEY to it (https://openrouter.ai/keys)",
    )


def install_appium(ctx: SetupContext) -> StepResult:
    """Installs Appium globally via npm when it is not already on PATH."""
    import shutil

    if shutil.which("appium"):
        result = run_command(["appium", "--version"], log_command=False)
        return StepResult(OK, "appium", (result.stdout or "").strip() or "installed")

    if not shutil.which("npm"):
        return StepResult(
            FAIL,
            "appium",
            "cannot install: npm not found",
            "Install Node.js 18+ first: brew install node",
        )

    command = ["npm", "install", "-g", "appium"]
    if ctx.dry_run:
        return _planned("appium", command)
    if not ctx.approved("Install Appium globally (npm install -g appium)?"):
        return _declined("appium", command)

    ctx.log("Installing Appium (this takes a minute)...")
    if _run_visible(command) != 0:
        return StepResult(FAIL, "appium", "npm install failed", f"Try manually: {' '.join(command)}")
    return StepResult(OK, "appium", "installed")


def install_driver(ctx: SetupContext, driver: str) -> StepResult:
    """Installs one Appium driver, skipping when Appium itself is missing."""
    label = f"driver: {driver}"
    existing = check_appium_driver(driver)
    if existing.status is Status.OK:
        return StepResult(OK, label, "already installed")
    if existing.status is Status.SKIP:
        return StepResult(SKIP, label, "appium not installed")

    command = ["appium", "driver", "install", driver]
    if ctx.dry_run:
        return _planned(label, command)
    if not ctx.approved(f"Install the Appium {driver} driver?"):
        return _declined(label, command)

    ctx.log(f"Installing the {driver} driver...")
    if _run_visible(command) != 0:
        return StepResult(FAIL, label, "install failed", f"Try manually: {' '.join(command)}")
    return StepResult(OK, label, "installed")


def install_vision(ctx: SetupContext) -> StepResult:
    """Adds the vision extra to whichever environment Ronin is installed in."""
    from ronin.setup.environment import detect_environment

    if vision_available():
        return StepResult(OK, "vision deps", "already installed")

    environment = detect_environment()
    command = environment.vision_install_command()
    if command is None:
        return StepResult(
            FAIL,
            "vision deps",
            f"cannot install into {environment.description}",
            "Install uv (https://docs.astral.sh/uv/) or add ultralytics, easyocr and "
            "huggingface-hub yourself",
        )

    if ctx.dry_run:
        return _planned("vision deps", command)
    if not ctx.approved(
        f"Install the vision extra into the {environment.kind.value} environment? "
        "It pulls PyTorch (hundreds of MB)"
    ):
        return _declined("vision deps", command)

    ctx.log(f"Installing ultralytics, easyocr and huggingface-hub into the {environment.description}...")
    if _run_visible(command, cwd=environment.install_cwd) != 0:
        return StepResult(
            FAIL, "vision deps", "install failed", f"Try manually: {' '.join(command)}"
        )
    return StepResult(OK, "vision deps", "installed")


def prefetch_models(ctx: SetupContext) -> StepResult:
    """Downloads the YOLO and OCR weights so the first run is not a silent wait."""
    if not vision_available():
        return StepResult(SKIP, "model weights", "vision deps not installed")
    if ctx.dry_run:
        return StepResult(INFO, "model weights", f"would prefetch {YOLO_REPO_ID} and EasyOCR")
    if not ctx.approved("Download the YOLO and EasyOCR weights now (several hundred MB)?"):
        return StepResult(
            WARN, "model weights", "skipped", "They will download on the first run instead"
        )

    # Runs out-of-process: the vision packages may have been installed moments
    # ago, and importing them here would mean importing into a mid-change env.
    command = [sys.executable, "-c", PREFETCH_SCRIPT]
    ctx.log("Downloading YOLO and EasyOCR weights (several hundred MB, cached afterwards)...")
    if _run_visible(command) != 0:
        return StepResult(
            WARN,
            "model weights",
            "prefetch failed",
            "They will download on the first run instead",
        )
    return StepResult(OK, "model weights", "cached")
