"""
Main execution module for the AI-driven mobile E2E testing agent.
"""

import argparse
import logging
import os
from pathlib import Path

from device.platform import Platform
from ronin_agent import RoninAgent

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    os.environ["OPENROUTER_API_KEY"] = api_key

    goal = "I want to sign in to the app with email 'test@email.com'"

    agent = RoninAgent(
        device_name="Pixel_8a",
        app_path="example/testing_app/build/app/outputs/flutter-apk/app-debug.apk",
        platform=Platform.ANDROID,
    )
    agent.execute(goal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ronin E2E mobile testing agent")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    main()
