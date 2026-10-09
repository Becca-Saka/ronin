"""Ronin command-line entrypoint.

Each command owns its own parser and handler; this module only assembles them
and sets up the process-wide concerns every command needs — logging, .env
loading and configuration.
"""

import argparse
import logging
import sys
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

from ronin import __version__
from ronin.cli import config as config_command
from ronin.cli import devices, doctor, run, setup, test
from ronin.config import load_config

COMMANDS = (doctor, setup, devices, config_command, test, run)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ronin",
        description="AI-driven mobile end-to-end testing agent.",
    )
    parser.add_argument("--version", action="version", version=f"ronin {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity (default: INFO)",
    )

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")
    for command in COMMANDS:
        command.register(subcommands)
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Parses the command line and dispatches to the requested command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    args, extra = parser.parse_known_args(argv)
    _configure_logging(args.log_level)
    # usecwd: without it find_dotenv walks up from this module's directory,
    # so an installed Ronin would never see the .env next to your app.
    load_dotenv(find_dotenv(usecwd=True))

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2

    if not getattr(args, "passthrough", False):
        # Re-parse strictly, so an unknown flag is an error rather than ignored.
        args = parser.parse_args(argv)
    args.extra = extra

    return handler(args, load_config())


if __name__ == "__main__":
    sys.exit(main())
