"""`ronin run` — drive the app towards a single goal."""

import argparse
import logging
from pathlib import Path

from ronin.config import Config
from ronin.llm.credentials import require_credentials

logger = logging.getLogger(__name__)


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Device and model flags.

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
    parser.add_argument(
        "--use-multimodal-vision",
        action="store_true",
        default=None,
        help="Allow the LLM to override detected coordinates",
    )
    parser.add_argument(
        "--headless", action="store_true", default=None, help="Run the emulator headlessly"
    )


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "run",
        help="Drive the app towards a single goal, with no QA reporting",
        description="Drive the app towards a single plain-English goal.",
    )
    parser.add_argument("goal", help="What the agent should accomplish, in plain English")
    parser.add_argument(
        "--app-path", required=True, help="Path to the app binary (.apk, .app or .ipa)"
    )
    add_device_arguments(parser)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    from ronin.agent.device.platform import Platform
    from ronin.agent.device_agent import DeviceAgent

    model = config.resolve("model", args.model)
    fast_model = config.resolve("fast_model", args.fast_model)
    platform = config.resolve("platform", args.platform)

    require_credentials(model, fast_model)

    if not Path(args.app_path).exists():
        raise SystemExit(f"App binary not found: {args.app_path}")

    agent = DeviceAgent(
        device_name=config.resolve("device", args.device),
        app_path=args.app_path,
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
