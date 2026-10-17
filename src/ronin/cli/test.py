"""`ronin test` — run a journey file as a generated test suite."""

import argparse
from typing import List

from ronin.config import Config
from ronin.preflight import require_run


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "test",
        add_help=False,
        help="Run a journey file as a generated test suite",
        description="Run a journey file as a generated test suite. See 'ronin test --help'.",
    )
    parser.set_defaults(handler=run, passthrough=True)


def strip_log_level(argv: List[str]) -> List[str]:
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


def run(args: argparse.Namespace, config: Config) -> int:
    from ronin.testing.cli import RoninCLI

    argv = strip_log_level(args.extra)
    cli = RoninCLI(config)
    cli.parser.prog = "ronin test"
    parsed = cli.parser.parse_args(argv)
    require_run(
        config.resolve("model", parsed.model),
        config.resolve("fast_model", parsed.fast_model),
        config.resolve("grounding", parsed.grounding),
    )
    # RoninCLI.run() re-parses argv and answers with the suite's status.
    return cli.run(argv)
