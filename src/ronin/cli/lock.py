"""`ronin lock` — write the suite a run will execute, with no device.

Generation needs the journey and a model. It does not need an emulator, an
app or a driver. So a separate command writes the lock, and leaves it to be
reviewed as a commit. A typo in a journey file then costs one model call
rather than a device boot.

Two jobs an ordinary run cannot do: lock with no device attached, and refresh
deliberately rather than as a side effect of testing.
"""

import argparse
import logging
from pathlib import Path

from ronin.app_discovery import resolve_journey_file
from ronin.cli.console import Console
from ronin.config import Config
from ronin.errors import UsageError
from ronin.preflight import require_lock

logger = logging.getLogger(__name__)


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "lock",
        help="Generate the test suite and write ronin.lock, with no device",
        description=(
            "Generate one scenario per declared cell and write ronin.lock "
            "beside the journey file. Commit the result: 'ronin test --locked' "
            "then runs exactly what was reviewed."
        ),
    )
    parser.add_argument(
        "journey_file",
        nargs="?",
        help="Path to the journey file. Found in this directory when omitted.",
    )
    parser.add_argument("--model", help="LiteLLM model to generate with")
    parser.add_argument(
        "--workflow",
        help="Which workflow(s) to lock, comma separated. Every one by default",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="refresh",
        help="Ask again for every cell, rather than only the missing and stale ones. "
        "Use it to take a new model's suite deliberately",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    """Writes the lock for every journey the file declares.

    Returns:
        0 when every journey was locked, 1 when a cell could not be covered,
        and 2 when the file itself could not be read.
    """
    from ronin.llm.llm_client import LLMClient
    from ronin.testing import lock as lockfile
    from ronin.testing import locking
    from ronin.testing.generator import ScenarioGenerator
    from ronin.testing.suite import load_workflows, select

    console = Console()
    model = config.resolve("model", args.model)
    require_lock(model)

    try:
        path = resolve_journey_file(args.journey_file)
        workflows = select(load_workflows(path), args.workflow)
    except UsageError as exc:
        console.write(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - any unreadable file is one outcome
        console.write(f"Could not read {args.journey_file or 'the journey file'}: {exc}")
        return 2

    target = lockfile.path_beside(Path(path))
    generator = ScenarioGenerator(LLMClient(model=model))
    uncovered = 0

    for workflow in workflows:
        for journey in workflow.journeys:
            resolution = locking.suite_for(
                journey,
                lambda wanted, cells: generator.generate_suite(wanted, cells),
                path=target,
                workflow=workflow.name,
                refresh=args.refresh,
            )
            label = f"{workflow.name}/{journey.id}" if workflow.name else journey.id
            console.write(f"{label}: {resolution.note}")
            for gap in resolution.uncovered:
                uncovered += 1
                console.write(f"  {gap.cell.key} not covered: {gap.reason}")

    console.write(f"Lock: {target}")
    return 1 if uncovered else 0
