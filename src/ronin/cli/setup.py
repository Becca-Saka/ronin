"""Orchestration for `ronin setup`."""

import sys
from typing import List

from ronin.cli.console import FAIL, Console
from ronin.cli.doctor import summarize
from ronin.doctor import is_macos, run_checks
from ronin.setup import (
    SetupContext,
    StepResult,
    ensure_env_file,
    install_appium,
    install_driver,
    install_vision,
    prefetch_models,
)

PLATFORM_DRIVERS = {"android": "uiautomator2", "ios": "xcuitest"}


def _build_context(console: Console, assume_yes: bool, dry_run: bool) -> SetupContext:
    def confirm(prompt: str) -> bool:
        if not sys.stdin.isatty():
            return False
        answer = input(f"  {prompt} [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")

    return SetupContext(
        assume_yes=assume_yes,
        dry_run=dry_run,
        confirm=confirm,
        log=lambda message: console.hint(message, indent=2),
    )


def _report(console: Console, results: List[StepResult]) -> None:
    for result in results:
        console.result(result.level, result.label, result.detail, result.hint)


def run(
    platforms: tuple[str, ...],
    assume_yes: bool = False,
    dry_run: bool = False,
    skip_models: bool = False,
) -> int:
    """Installs what Ronin needs, then reports the resulting environment."""
    console = Console()
    console.title("Ronin setup" + (" (dry run)" if dry_run else ""))

    if not assume_yes and not dry_run and not sys.stdin.isatty():
        console.hint("Not running interactively — installs will be skipped. Pass -y to proceed.", indent=2)

    ctx = _build_context(console, assume_yes, dry_run)
    steps: List[StepResult] = []

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
