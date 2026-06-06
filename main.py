"""
Main execution module for the AI-driven mobile E2E testing agent.
"""

import argparse
import logging
import os
import sys
from device.platform import Platform
from ronin_agent import RoninAgent

logger = logging.getLogger(__name__)


def main():
    """Main execution function coordinating agent instantiation and goal execution."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    os.environ["OPENROUTER_API_KEY"] = api_key
    goal = (
        "Sign in to the app with email 'test@email.com'. "
        "Click on the first notification item in the list. "
        "On the detail screen, type 'Awesome notification!' in the reply field, "
        "toggle the 'Pin notification' switch to true, "
        "and click the 'Submit' button."
    )
    app_path = "example/testing_app/build/app/outputs/flutter-apk/app-debug.apk"

    agent = RoninAgent(
        device_name="Pixel_8a",
        app_path=app_path,
        platform=Platform.ANDROID,
        use_multimodal_vision=False,
        max_steps=10,
        headless=False,
    )
    try:
        agent.execute(goal)
    except KeyboardInterrupt:
        logger.info("Agent execution interrupted by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ronin E2E mobile testing agent", add_help=False
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args, remaining = parser.parse_known_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    # Rebuild sys.argv for delegated parsers
    sys.argv = [sys.argv[0]] + remaining

    if len(sys.argv) > 1 and sys.argv[1] == "test-journey":
        sys.argv.pop(1)
        from ronin_testing.cli import main as cli_main

        cli_main()
    else:
        # Re-parse legacy arguments if not running a test journey
        legacy_parser = argparse.ArgumentParser(
            description="Ronin E2E mobile testing agent"
        )
        # Log level is already processed, but add it to legacy parser for completeness
        legacy_parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )
        legacy_parser.parse_args(remaining)
        main()
