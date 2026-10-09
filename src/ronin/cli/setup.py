"""`ronin setup` — install what Ronin needs on this machine."""

import argparse
import sys
from typing import TYPE_CHECKING, List

from ronin.cli.console import FAIL, Console
from ronin.cli.doctor import summarize
from ronin.config import Config
from ronin.doctor import is_macos, run_checks

if TYPE_CHECKING:
    from ronin.setup import SetupContext, StepResult

PLATFORM_DRIVERS = {"android": "uiautomator2", "ios": "xcuitest"}


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "setup",
        help="Install what Ronin needs on this machine",
        description="Install Appium, its drivers and the vision extra, then verify the result.",
    )
    parser.add_argument(
        "--platform",
        choices=["android", "ios", "all"],
        help="Which platform to set up (default: the configured platform)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", dest="assume_yes", help="Install without prompting"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be installed, change nothing"
    )
    parser.add_argument(
        "--skip-models", action="store_true", help="Do not prefetch the YOLO and OCR weights"
    )
    parser.set_defaults(handler=run)


def _build_context(console: Console, assume_yes: bool, dry_run: bool) -> "SetupContext":
    from ronin.setup import SetupContext

    return SetupContext(
        assume_yes=assume_yes,
        dry_run=dry_run,
        confirm=console.confirm,
        log=lambda message: console.hint(message, indent=2),
    )


def _report(console: Console, results: List["StepResult"]) -> None:
    for result in results:
        console.result(result.level, result.label, result.detail, result.hint)


def run(args: argparse.Namespace, config: Config) -> int:
    """Installs what Ronin needs, then reports the resulting environment."""
    from ronin.setup import (
        ensure_env_file,
        install_appium,
        install_driver,
        install_vision,
        prefetch_models,
    )

    platforms = config.platforms(args.platform)
    assume_yes, dry_run, skip_models = args.assume_yes, args.dry_run, args.skip_models

    console = Console()
    console.title("Ronin setup" + (" (dry run)" if dry_run else ""))

    if not assume_yes and not dry_run and not sys.stdin.isatty():
        console.hint(
            "stdin is not a terminal — prompts read from it, and EOF counts as no. Pass -y to accept everything.",
            indent=2,
        )

    ctx = _build_context(console, assume_yes, dry_run)
    steps: List["StepResult"] = []

    console.heading("Project")
    steps.append(ensure_env_file(ctx))
    _report(console, steps[-1:])

    console.heading("Appium")
    appium_results = [install_appium(ctx)]
    for platform in platforms:
        driver = PLATFORM_DRIVERS[platform]
        if driver == "xcuitest" and not is_macos():
            continue
        appium_results.append(install_driver(ctx, driver))
    _report(console, appium_results)
    steps.extend(appium_results)

    console.heading("Vision")
    vision_results = [install_vision(ctx)]
    if skip_models:
        console.result("skip", "model weights", "skipped (--skip-models)")
    else:
        vision_results.append(prefetch_models(ctx))
    _report(console, vision_results)
    steps.extend(vision_results)

    console.heading("Verifying")
    sections = run_checks(platforms)
    outstanding = [r for s in sections for r in (*s.failures, *s.warnings)]
    if outstanding:
        for result in outstanding:
            console.result(result.status.value, result.name, result.detail, result.hint)
    else:
        console.result("ok", "all checks", "passed")

    exit_code = summarize(sections, console)
    if any(step.level == FAIL for step in steps):
        return 1
    return exit_code
