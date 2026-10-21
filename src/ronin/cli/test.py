"""`ronin test` — run a journey file as a generated test suite."""

import argparse

from ronin.cli.console import configure_logging
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


def run(args: argparse.Namespace, config: Config) -> int:
    from ronin.testing.cli import RoninCLI

    cli = RoninCLI(config)
    cli.parser.prog = "ronin test"
    parsed = cli.parser.parse_args(args.extra)
    if parsed.log_level:
        # `ronin test --log-level X` would otherwise be dropped in silence: the
        # top-level parser never sees a flag written after the subcommand.
        configure_logging(parsed.log_level)
    require_run(
        config.resolve("model", parsed.model),
        config.resolve("fast_model", parsed.fast_model),
        config.resolve("grounding", parsed.grounding),
    )
    return cli.run_parsed(parsed)
