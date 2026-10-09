"""Ronin command-line entrypoint.

Dispatches the top-level subcommands and owns the process-wide concerns
(logging, .env loading, API key validation) that every subcommand needs.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from ronin import __version__
from ronin.config import Config, load_config

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def _require_credentials(*models: Optional[str]) -> None:
    """Fails fast when a requested model's credentials are absent from the environment.

    LiteLLM knows what each provider needs — one key for most, several for
    Bedrock, a base URL for Ollama — so defer to it rather than mapping
    prefixes to environment variables here. It reports nothing missing for
    models it cannot route, which keeps custom endpoints working.
    """
    import litellm

    for model in models:
        if not model:
            continue
        missing = litellm.validate_environment(model).get("missing_keys") or []
        if missing:
            raise SystemExit(
                f"Missing credentials for '{model}': {', '.join(missing)}.\n"
                "Add them to a .env file in this directory or export them."
            )


def _add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Device/model flags shared by the agent-driving subcommands.

    Defaults stay None so an explicit flag is distinguishable from an absent
    one; ronin.config supplies the value when the flag is omitted.
    """
    parser.add_argument("--platform", choices=["android", "ios"], help="Target platform")
    parser.add_argument("--device", help="Name of emulator/simulator/device")
    parser.add_argument("--model", help="LiteLLM model to use")
    parser.add_argument("--fast-model", help="Cheaper/faster LiteLLM model for per-step planning")
    parser.add_argument("--output-dir", help="Directory to save outputs and screenshots")
    parser.add_argument("--max-steps", type=int, help="Max steps per run")
    parser.add_argument("--settle-time", type=float, help="Seconds to wait after each action")
    parser.add_argument("--use-multimodal-vision", action="store_true", default=None, help="Allow the LLM to override detected coordinates")
    parser.add_argument("--headless", action="store_true", default=None, help="Run the emulator headlessly")


def _platforms(requested: Optional[str], config: Config) -> tuple[str, ...]:
    """Which platforms a check/install command covers.

    Without --platform, follow the configured target rather than assuming both:
    an Android-only user should not be told a missing iOS driver blocks them.
    """
    choice = requested or config.get("platform")
    if choice == "all":
        return ("android", "ios")
    return (choice,)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ronin",
        description="AI-driven mobile end-to-end testing agent.",
    )
    parser.add_argument("--version", action="version", version=f"ronin {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    subcommands.add_parser(
        "test",
        add_help=False,
        help="Run a journey file as a generated test suite",
        description="Run a journey file as a generated test suite. See 'ronin test --help'.",
    )

    doctor_parser = subcommands.add_parser(
        "doctor",
        help="Check that this machine can run Ronin",
        description="Report on every prerequisite Ronin needs, with the command to fix each gap.",
    )
    doctor_parser.add_argument(
        "--platform",
        choices=["android", "ios", "all"],
        help="Which platform's prerequisites to treat as required (default: the configured platform)",
    )
    doctor_parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit machine-readable output for CI"
    )

    devices_parser = subcommands.add_parser(
        "devices",
        help="List, create and choose devices",
        description="Show every AVD, simulator and attached device, or create a new AVD.",
    )
    devices_parser.add_argument(
        "--platform", choices=["android", "ios", "all"], help="Limit the listing to one platform"
    )
    devices_parser.add_argument("--all", action="store_true", help="Include watchOS and tvOS simulators")
    devices_parser.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable output")
    devices_actions = devices_parser.add_subparsers(dest="devices_action", metavar="<action>")
    devices_actions.add_parser("list", help="Show the available devices (default)")
    devices_create = devices_actions.add_parser("create", help="Create an Android virtual device")
    devices_create.add_argument("name", nargs="?", help="AVD name (default: the configured device)")
    devices_create.add_argument("--api", type=int, help="Android API level, e.g. 35")
    devices_create.add_argument("--profile", help="Hardware profile id, e.g. pixel_7")
    devices_create.add_argument("-y", "--yes", action="store_true", dest="assume_yes", help="Do not prompt")
    devices_default = devices_actions.add_parser("default", help="Make a device the default target")
    devices_default.add_argument("name", help="Device name to target by default")
    devices_default.add_argument(
        "--project",
        action="store_const",
        const="project",
        dest="scope",
        default="global",
        help="Write to ./ronin.yaml instead of the global config",
    )

    config_parser = subcommands.add_parser(
        "config",
        help="Show or change saved settings",
        description="Show the effective settings and where each came from, or change one.",
    )
    config_actions = config_parser.add_subparsers(dest="config_action", metavar="<action>")
    config_actions.add_parser("show", help="Print the effective settings and their origins")
    config_actions.add_parser("path", help="Print the config file locations")
    config_set = config_actions.add_parser("set", help="Save a setting")
    config_set.add_argument("key", help="Setting name, e.g. device")
    config_set.add_argument("value", help="Value to save")
    config_unset = config_actions.add_parser("unset", help="Remove a saved setting")
    config_unset.add_argument("key", help="Setting name to remove")
    for sub in (config_set, config_unset):
        sub.add_argument(
            "--project",
            action="store_const",
            const="project",
            dest="scope",
            default="global",
            help=f"Write to ./ronin.yaml instead of the global config",
        )

    setup_parser = subcommands.add_parser(
        "setup",
        help="Install what Ronin needs on this machine",
        description="Install Appium, its drivers and the vision extra, then verify the result.",
    )
    setup_parser.add_argument(
        "--platform",
        choices=["android", "ios", "all"],
        help="Which platform to set up (default: the configured platform)",
    )
    setup_parser.add_argument(
        "-y", "--yes", action="store_true", dest="assume_yes", help="Install without prompting"
    )
    setup_parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be installed, change nothing"
    )
    setup_parser.add_argument(
        "--skip-models", action="store_true", help="Do not prefetch the YOLO and OCR weights"
    )

    run_parser = subcommands.add_parser(
        "run",
        help="Drive the app towards a single goal, with no QA reporting",
        description="Drive the app towards a single plain-English goal.",
    )
    run_parser.add_argument("goal", help="What the agent should accomplish, in plain English")
    run_parser.add_argument("--app-path", required=True, help="Path to the app binary (.apk, .app or .ipa)")
    _add_device_arguments(run_parser)

    return parser


def _run_goal(args: argparse.Namespace, config: Config) -> int:
    from ronin.agent.device.platform import Platform
    from ronin.agent.device_agent import DeviceAgent

    model = config.resolve("model", args.model)
    fast_model = config.resolve("fast_model", args.fast_model)
    platform = config.resolve("platform", args.platform)
    app_path = args.app_path

    _require_credentials(model, fast_model)

    if not Path(app_path).exists():
        raise SystemExit(f"App binary not found: {app_path}")

    agent = DeviceAgent(
        device_name=config.resolve("device", args.device),
        app_path=app_path,
        platform=Platform.ANDROID if platform == "android" else Platform.IOS,
        model=model,
        fast_model=fast_model,
        assets_dir=Path(config.resolve("output_dir", args.output_dir)),
        max_steps=config.resolve("max_steps", args.max_steps),
        settle_time=config.resolve("settle_time", args.settle_time),
        use_multimodal_vision=config.resolve("use_multimodal_vision", args.use_multimodal_vision),
        headless=config.resolve("headless", args.headless),
    )
    try:
        result = agent.run_standalone(args.goal)
    except KeyboardInterrupt:
        logger.info("Agent execution interrupted by user")
        return 130

    if result.status == "goal_reached":
        logger.info("Goal reached in %d step(s).", result.steps_executed)
        return 0
    logger.error(
        "Goal not reached (%s) after %d step(s): %s",
        result.status,
        result.steps_executed,
        result.failure_reason or "no reason reported",
    )
    return 1


def _strip_log_level(argv: List[str]) -> List[str]:
    """Removes the top-level --log-level flag from args destined for a subparser."""
    cleaned: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--log-level":
            skip_next = True
            continue
        if token.startswith("--log-level="):
            continue
        cleaned.append(token)
    return cleaned


def _run_test_suite(argv: List[str], config: Config) -> int:
    from ronin.testing.cli import RoninCLI

    cli = RoninCLI(config)
    cli.parser.prog = "ronin test"
    args = cli.parser.parse_args(argv)
    _require_credentials(
        config.resolve("model", args.model), config.resolve("fast_model", args.fast_model)
    )
    # RoninCLI.run() re-parses and calls sys.exit() with the suite's status.
    asyncio.run(cli.run(argv))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parses the command line and dispatches to the requested subcommand."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    # 'test' delegates its own flags to the testing CLI, so split at the
    # subcommand rather than letting the top-level parser consume them.
    args, remaining = parser.parse_known_args(argv)
    _configure_logging(args.log_level)
    load_dotenv()
    config = load_config()

    if args.command == "doctor":
        from ronin.cli import doctor

        args = parser.parse_args(argv)
        return doctor.run(_platforms(args.platform, config), args.as_json)
    if args.command == "setup":
        from ronin.cli import setup

        args = parser.parse_args(argv)
        return setup.run(
            _platforms(args.platform, config), args.assume_yes, args.dry_run, args.skip_models
        )
    if args.command == "devices":
        from ronin.cli import devices

        args = parser.parse_args(argv)
        platforms = _platforms(args.platform, config) if args.platform else ("android", "ios")
        return devices.run(args, config, platforms)
    if args.command == "config":
        from ronin.cli import config as config_command

        args = parser.parse_args(argv)
        return config_command.run(args, config)
    if args.command == "test":
        return _run_test_suite(_strip_log_level(remaining), config)
    if args.command == "run":
        args = parser.parse_args(argv)
        return _run_goal(args, config)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
