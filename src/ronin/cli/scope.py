"""The --project flag, shared by the commands that write configuration."""

import argparse


def add_scope_argument(parser: argparse.ArgumentParser) -> None:
    """Adds --project, selecting ./ronin.yaml over the global config file."""
    parser.add_argument(
        "--project",
        action="store_const",
        const="project",
        dest="scope",
        default="global",
        help="Write to ./ronin.yaml instead of the global config",
    )
