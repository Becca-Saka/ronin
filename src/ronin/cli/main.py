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
from ronin.defaults import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATFORM,
    DEFAULT_SETTLE_TIME,
)

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
    """Device/model flags shared by the agent-driving subcommands."""
    parser.add_argument("--platform", default=DEFAULT_PLATFORM, choices=["android", "ios"], help="Target platform")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Name of emulator/simulator/device")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM model to use")
    parser.add_argument("--fast-model", default=None, help="Cheaper/faster LiteLLM model for per-step planning")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs and screenshots")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per run")
    parser.add_argument("--settle-time", type=float, default=DEFAULT_SETTLE_TIME, help="Seconds to wait after each action")
    parser.add_argument("--use-multimodal-vision", action="store_true", help="Allow the LLM to override detected coordinates")
    parser.add_argument("--headless", action="store_true", help="Run the emulator headlessly")


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
        default="all",
        choices=["android", "ios", "all"],
        help="Only check the prerequisites for one platform (default: all)",
    )
    doctor_parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit machine-readable output for CI"
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


def _run_goal(args: argparse.Namespace) -> int:
    from ronin.agent.device.platform import Platform
    from ronin.agent.device_agent import DeviceAgent

    _require_credentials(args.model, args.fast_model)

    if not Path(args.app_path).exists():
        raise SystemExit(f"App binary not found: {args.app_path}")

    agent = DeviceAgent(
        device_name=args.device,
        app_path=args.app_path,
        platform=Platform.ANDROID if args.platform == "android" else Platform.IOS,
        model=args.model,
        fast_model=args.fast_model,
        assets_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        settle_time=args.settle_time,
        use_multimodal_vision=args.use_multimodal_vision,
        headless=args.headless,
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


def _run_test_suite(argv: List[str]) -> int:
    from ronin.testing.cli import RoninCLI

    cli = RoninCLI()
    cli.parser.prog = "ronin test"
    args = cli.parser.parse_args(argv)
    _require_credentials(args.model, args.fast_model)
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

    if args.command == "doctor":
        from ronin.cli import doctor

        args = parser.parse_args(argv)
        platforms = ("android", "ios") if args.platform == "all" else (args.platform,)
        return doctor.run(platforms, args.as_json)
    if args.command == "test":
        return _run_test_suite(_strip_log_level(remaining))
    if args.command == "run":
        args = parser.parse_args(argv)
        return _run_goal(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
