"""`ronin config` — show or change saved settings."""

import argparse

from ronin.cli.console import BOLD, DIM, OK, WARN, Console
from ronin.config import (
    PROJECT_CONFIG_FILENAME,
    SETTINGS,
    Config,
    ConfigError,
    find_project_config,
    load_config,
    set_value,
    unset_value,
)


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "config",
        help="Show or change saved settings",
        description="Show the effective settings and where each came from, or change one.",
    )
    parser.set_defaults(handler=run)

    actions = parser.add_subparsers(dest="config_action", metavar="<action>")
    actions.add_parser("show", help="Print the effective settings and their origins")
    actions.add_parser("path", help="Print the config file locations")

    set_parser = actions.add_parser("set", help="Save a setting")
    set_parser.add_argument("key", help="Setting name, e.g. device")
    set_parser.add_argument("value", help="Value to save")

    unset_parser = actions.add_parser("unset", help="Remove a saved setting")
    unset_parser.add_argument("key", help="Setting name to remove")



def show(config: Config, console: Console) -> int:
    width = max(len(name) for name in SETTINGS)
    console.title("Ronin config")
    console.write()
    for name in sorted(SETTINGS):
        value = config.get(name)
        rendered = "—" if value is None else str(value)
        console.write(f"  {name.ljust(width)}  {rendered}")
        console.write(f"  {' ' * width}  {console.paint(config.origin(name), DIM)}")
    return 0


def paths(console: Console) -> int:
    """Where the settings come from, which is one file."""
    project = find_project_config()
    console.title("Ronin config file")
    console.write()
    if project is None:
        console.write(
            f"  —  (no {PROJECT_CONFIG_FILENAME} in this directory or above)"
        )
    else:
        console.write(f"  {project}  {console.paint('(exists)', DIM)}")
    return 0


def run(args: argparse.Namespace, config: Config) -> int:
    """Dispatches the config action, defaulting to show."""
    console = Console()
    action = getattr(args, "config_action", None) or "show"

    if action == "show":
        return show(config, console)
    if action == "path":
        return paths(console)

    try:
        if action == "set":
            path = set_value(args.key, args.value)
            updated = load_config()
            console.result(
                OK, args.key, f"{updated.get(args.key)!r} saved to {path}"
            )
        else:
            path = unset_value(args.key)
            updated = load_config()
            console.result(
                WARN,
                args.key,
                f"removed from {path}; now {updated.get(args.key)!r} ({updated.origin(args.key)})",
            )
    except ConfigError as exc:
        console.write(console.paint(str(exc), BOLD))
        return 2
    return 0
