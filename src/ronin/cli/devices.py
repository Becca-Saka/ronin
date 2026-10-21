"""`ronin devices` — list, create and choose devices."""

import argparse
import json
from dataclasses import asdict

from ronin.cli.console import FAIL, INFO, OK, SKIP, WARN, Console
from ronin.config import Config, set_value
from ronin.devices.android import (
    AndroidToolingError,
    best_profile,
    create_avd,
    device_profiles,
    install_system_image,
    pick_image,
)
from ronin.devices.discovery import (
    addressable,
    android_devices,
    list_devices,
    resolve,
)
from ronin.errors import UsageError


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "devices",
        help="List, create and choose devices",
        description="Show every AVD, simulator and attached device, or create a new AVD.",
    )
    parser.add_argument(
        "--platform", choices=["android", "ios", "all"], help="Limit the listing to one platform"
    )
    parser.add_argument("--all", action="store_true", help="Include watchOS and tvOS simulators")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit machine-readable output"
    )
    parser.set_defaults(handler=run)

    actions = parser.add_subparsers(dest="devices_action", metavar="<action>")
    actions.add_parser("list", help="Show the available devices (default)")

    create_parser = actions.add_parser("create", help="Create an Android virtual device")
    create_parser.add_argument("name", nargs="?", help="AVD name (default: the configured device)")
    create_parser.add_argument("--api", type=int, help="Android API level, e.g. 35")
    create_parser.add_argument("--profile", help="Hardware profile id, e.g. pixel_7")
    create_parser.add_argument(
        "-y", "--yes", action="store_true", dest="assume_yes", help="Do not prompt"
    )

    default_parser = actions.add_parser("default", help="Make a device the default target")
    default_parser.add_argument("name", help="Device name to target by default")


def _chosen(target, devices):
    """Which listed device the configured target names, or None.

    An ambiguous target names none of them. The run would refuse it, and
    marking either row would say Ronin had chosen.
    """
    if not target:
        return None
    try:
        return resolve(target, devices)
    except UsageError:
        return None


def show(devices, config: Config, console: Console) -> int:
    if not devices:
        console.write()
        console.result(WARN, "devices", "none found", "Create one with: ronin devices create")
        return 1

    target = config.get("device")
    # Marked by which device the target picks, not by what it is spelled as.
    # A name, a udid and a serial can all name one target, and two devices
    # can share a name, which would otherwise mark both rows.
    chosen = _chosen(target, devices)
    columns = ("", "NAME", "PLATFORM", "STATE", "RUNTIME", "ID")
    rows = [
        (
            "*" if device is chosen else " ",
            device.name,
            f"{device.platform}/{device.kind}",
            device.state,
            device.runtime or "",
            device.identifier or "",
        )
        for device in devices
    ]
    widths = [max(len(str(row[i])) for row in (columns, *rows)) for i in range(len(columns))]

    console.write()
    console.write("  " + "  ".join(str(c).ljust(w) for c, w in zip(columns, widths)).rstrip())
    for row in rows:
        console.write("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    console.write()
    console.hint(f"* configured target ({target}) — change it with: ronin config set device <name>", indent=2)
    return 0


def create(name: str, api, profile_request, assume_yes: bool, config: Config, console: Console) -> int:
    """Creates an AVD, offering to download a system image when none fits."""
    console.title(f"Creating AVD '{name}'")

    existing = {device.name for device in android_devices() if device.kind == "avd"}
    if name in existing and not console.confirm(f"AVD '{name}' already exists. Recreate it?", assume_yes):
        console.result(SKIP, name, "left untouched")
        return 0

    try:
        profile = best_profile(profile_request)
        if profile is None:
            available = [p for p in device_profiles() if p.startswith("pixel")] or device_profiles()[:8]
            console.result(
                FAIL,
                "profile",
                f"'{profile_request}' is not a profile this SDK knows",
                f"Available Pixel profiles: {', '.join(available)}",
            )
            return 1

        image, wanted = pick_image(api)
        if image is None:
            console.result(INFO, "system image", f"{wanted} is not installed")
            if not console.confirm(f"Download {wanted.package} now (1-2 GB)?", assume_yes):
                console.result(
                    FAIL,
                    "system image",
                    "declined",
                    f"Install it yourself: sdkmanager --install '{wanted.package}'",
                )
                return 1
            if install_system_image(wanted) != 0:
                console.result(FAIL, "system image", "download failed")
                return 1
            image, _ = pick_image(api)
            if image is None:
                console.result(FAIL, "system image", "still not available after install")
                return 1

        console.result(OK, "system image", str(image))
        console.result(OK, "hardware profile", profile)

        if create_avd(name, image, profile) != 0:
            console.result(FAIL, name, "avdmanager failed")
            return 1
    except AndroidToolingError as exc:
        console.result(FAIL, "android sdk", str(exc))
        return 1

    console.result(OK, name, "created")
    if config.get("device") != name:
        console.hint(f"Target it by default with: ronin config set device {name}", indent=4)
    return 0


def set_default(name: str, config: Config, console: Console) -> int:
    """Saves the target a run uses when none is named.

    A device is addressable by the name it was listed under or by its own id,
    and a person may paste either. The name resolves through the same
    function a session uses, so the two cannot disagree about which target a
    word picks.

    What is stored is the durable name of whatever was picked, which
    `addressable` decides. A listing name is not always one: an Android
    handset is listed by its model, and nothing can select that.
    """
    try:
        found = resolve(name, list_devices())
    except UsageError as ambiguous:
        console.result(FAIL, "device", f"'{name}' is not one target", str(ambiguous))
        return 1
    if found is None:
        console.result(
            FAIL, "device", f"'{name}' not found", "See what is available with: ronin devices"
        )
        return 1

    saved = addressable(found) or name
    path = set_value("device", saved)
    detail = f"{saved} saved to {path}"
    if saved != name:
        detail = f"{name} saved to {path} as {saved}"
    console.result(OK, "device", detail)
    return 0


def run(args: argparse.Namespace, config: Config) -> int:
    """Dispatches the devices action, defaulting to the listing."""
    console = Console()
    action = getattr(args, "devices_action", None) or "list"

    if action == "create":
        return create(
            args.name or config.get("device"),
            args.api,
            args.profile,
            args.assume_yes,
            config,
            console,
        )
    if action == "default":
        return set_default(args.name, config, console)

    platforms = config.platforms(args.platform) if args.platform else ("android", "ios")
    devices = list_devices(platforms, include_wearables=args.all)
    if args.as_json:
        print(json.dumps({"devices": [asdict(device) for device in devices]}, indent=2))
        return 0 if devices else 1
    console.title("Ronin devices")
    return show(devices, config, console)
