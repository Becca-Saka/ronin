"""Finding the launcher entry in an APK manifest.

`aapt dump badging` reports a launchable-activity only when the MAIN/LAUNCHER
intent-filter sits on an `<activity>`. An app that lets the user change its
icon puts that filter on `<activity-alias>` entries instead, one per icon, and
disables all but one. aapt then reports no launchable activity at all, Appium
falls back to an intent carrying no component, and Android refuses it because
it matches every launcher on the device.

Reading the manifest directly finds those aliases. `android:enabled` is what
separates the one icon in use from the rest; it defaults to true when absent.
"""

import re
from typing import NamedTuple, Optional

MAIN_ACTION = "android.intent.action.MAIN"
LAUNCHER_CATEGORY = "android.intent.category.LAUNCHER"

ELEMENT = re.compile(r"^(\s*)E: (\S+)")
NAME_ATTR = re.compile(r'A: .*?android:name\([^)]*\)="([^"]*)"')
ENABLED_ATTR = re.compile(r"A: .*?android:enabled\([^)]*\)=(\S+)")


class LauncherEntry(NamedTuple):
    """One `<activity>` or `<activity-alias>` that the launcher can start."""

    name: str
    enabled: bool


def _qualify(name: str, package: str) -> str:
    """Expands a manifest-relative class name against its package."""
    return f"{package}{name}" if name.startswith(".") else name


def launcher_entries(xmltree: str, package: str) -> list[LauncherEntry]:
    """Every activity or alias in the dump declaring MAIN and LAUNCHER.

    Args:
        xmltree: Output of `aapt2 dump xmltree --file AndroidManifest.xml`.
        package: The APK's package, used to expand relative class names.

    Returns:
        One entry per launcher declaration, in manifest order.
    """
    entries: list[LauncherEntry] = []
    name: Optional[str] = None
    enabled = True
    depth = -1
    nested = False
    main = launcher = False

    def close() -> None:
        if name and main and launcher:
            entries.append(LauncherEntry(_qualify(name, package), enabled))

    for line in xmltree.splitlines():
        element = ELEMENT.match(line)
        if element:
            indent, tag = len(element.group(1)), element.group(2)
            if depth >= 0 and indent <= depth:
                close()
                name, enabled, depth, nested, main, launcher = None, True, -1, False, False, False
            if tag in ("activity", "activity-alias"):
                name, enabled, depth, nested, main, launcher = None, True, indent, False, False, False
            elif depth >= 0:
                # An intent-filter, or anything else below the activity. Its own
                # android:name attributes are not the activity's name.
                nested = True
            continue

        if depth < 0:
            continue

        attribute = NAME_ATTR.search(line)
        if attribute:
            value = attribute.group(1)
            if not nested:
                name = name or value
            elif value == MAIN_ACTION:
                main = True
            elif value == LAUNCHER_CATEGORY:
                launcher = True
            continue

        state = ENABLED_ATTR.search(line)
        if state and not nested:
            enabled = state.group(1).strip().lower() != "false"

    close()
    return entries


def enabled_launcher(xmltree: str, package: str) -> Optional[str]:
    """The one launcher entry the device would show, or None when unclear.

    An app with several icons keeps exactly one alias enabled. Returning None
    for anything else keeps a disabled alias out of the launch intent, because
    starting one fails in a way that reads like the app is broken.
    """
    enabled = [entry.name for entry in launcher_entries(xmltree, package) if entry.enabled]
    return enabled[0] if len(enabled) == 1 else None
