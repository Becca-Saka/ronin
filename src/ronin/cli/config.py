"""`ronin config` — show or change saved settings."""

import argparse

from ronin.cli.console import BOLD, DIM, OK, WARN, Console
from ronin.cli.scope import add_scope_argument
from ronin.config import (
    GLOBAL_CONFIG_PATH,
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

    for action_parser in (set_parser, unset_parser):
        add_scope_argument(action_parser)


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
    project = find_project_config()
    console.title("Ronin config files")
    console.write()
    for label, path in (("global", GLOBAL_CONFIG_PATH), ("project", project)):
        if path is None:
            console.write(f"  {label:8} —  (no {PROJECT_CONFIG_FILENAME} in this directory or above)")
        else:
            state = "exists" if path.is_file() else "not created yet"
            console.write(f"  {label:8} {path}  {console.paint(f'({state})', DIM)}")
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
            path = set_value(args.key, args.value, scope=args.scope)
            updated = load_config()
            console.result(
                OK, args.key, f"{updated.get(args.key)!r} saved to {path}"
            )
        else:
            path = unset_value(args.key, scope=args.scope)
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
