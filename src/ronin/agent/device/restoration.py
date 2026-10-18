"""Giving the device back the way a scenario found it.

Scenarios share one device, and `SetupMode` restores the app rather than the
system around it. A scenario that goes offline, or moves the clock, or switches
the screen reader on, leaves it that way for every scenario after it. The next
one then tests a device nobody configured, and its verdict is about a state it
never declared.

So each scenario reads the states a capability could change, and writes back
the ones that moved. Reading first is what makes this honest. A state that could not be read is not
restored, because restoring it would write a value that was never observed.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

from ronin.agent.device import receipts
from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.driver_commands import DriverCommands
from ronin.agent.device.platform import Platform
from ronin.errors import DeviceUnavailable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceStateSnapshot:
    """The system states a capability can change, as they were found.

    Every field is optional. None means the state could not be read, and an
    unread state is left alone rather than guessed at.
    """

    airplane: Optional[str] = None
    wifi: Optional[str] = None
    data: Optional[str] = None
    battery: Optional[str] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None
    accessibility: Optional[str] = None
    """Every enabled accessibility service, not one service's state.

    The setting is one list, and a scenario that switches a screen reader
    on or off rewrites it. Keeping one boolean would restore the list to
    whatever that boolean implied and drop every other service in it.
    """

    def describe(self) -> str:
        known = {
            name: value
            for name, value in vars(self).items()
            if value is not None
        }
        return ", ".join(f"{name}={value}" for name, value in known.items()) or "nothing"


def _radio(radios: Optional[dict], name: str) -> Optional[str]:
    """One radio as "on" or "off", or None when the driver did not say."""
    if radios is None or name not in radios:
        return None
    return "on" if radios[name] else "off"


def snapshot(
    serial: str, package: Optional[str], commands: Optional[DriverCommands] = None
) -> DeviceStateSnapshot:
    """The states to put back, read before the scenario runs.

    The three radios are read together where the driver can read them. Airplane mode alone is not enough. A scenario that asks for wifi_only
        turns mobile data off and leaves airplane mode where it was. A
        restore watching only airplane mode then sees nothing move, and
        leaves data off for every scenario after it.
    """
    radios = commands.connectivity() if commands is not None else None
    return DeviceStateSnapshot(
        airplane=receipts.airplane_mode(serial),
        wifi=_radio(radios, "wifi"),
        data=_radio(radios, "data"),
        battery=receipts.battery_level(serial),
        timezone=receipts.timezone(serial),
        locale=receipts.app_locale(serial, package) if package else None,
        accessibility=AdbControl.enabled_accessibility_services(serial),
    )


def _put_back(
    what: str,
    write: Callable[[str], None],
    wanted: Optional[str],
    observed: Optional[str],
) -> None:
    """Write one state back, when it was read and has since moved.

    The writer is handed the value, rather than closing over it, so it is
    called only with a state that was actually observed.

    A failure here is reported and does not propagate. Restoration runs while a scenario is ending, often while an exception is on
    its way out. Losing that exception would hide why the scenario ended.
    """
    if wanted is None or observed is None or observed == wanted:
        return
    logger.info("Restoring %s: %s -> %s", what, observed, wanted)
    try:
        write(wanted)
    except (DeviceUnavailable, OSError) as exc:
        logger.warning("Could not restore %s to %s: %s", what, wanted, exc)


def _restore_radios(
    serial: str, found: DeviceStateSnapshot, commands: Optional[DriverCommands]
) -> None:
    """Put the three radios back, in one call where the driver can read them.

    Falls back to airplane mode alone, which is all adb can set. That cannot
    return wifi or data on its own, so a run without a driver keeps the narrower
    guarantee rather than a pretended one.
    """
    if commands is None or found.wifi is None or found.data is None:
        _put_back(
            "airplane mode",
            lambda value: AdbControl.set_airplane_mode(serial, value == "on"),
            found.airplane,
            receipts.airplane_mode(serial),
        )
        return

    now = commands.connectivity()
    if now is None:
        return
    wanted = {
        "wifi": found.wifi == "on",
        "data": found.data == "on",
        "airplane": found.airplane == "on",
    }
    current = {
        "wifi": bool(now.get("wifi")),
        "data": bool(now.get("data")),
        "airplane": bool(now.get("airplaneMode")),
    }
    if current == wanted:
        return
    logger.info("Restoring the radios: %s -> %s", current, wanted)
    try:
        commands.set_connectivity(**wanted)
    except DeviceUnavailable as exc:
        logger.warning("Could not restore the radios to %s: %s", wanted, exc)


def restore(
    serial: str,
    package: Optional[str],
    found: DeviceStateSnapshot,
    commands: Optional[DriverCommands] = None,
) -> None:
    """Put back every state that moved during the scenario."""
    _restore_radios(serial, found, commands)
    # Reset rather than written back: an override is either on or off, so
    # there is no level to restore it to, only the real battery.
    _put_back(
        "battery",
        lambda _: AdbControl.reset_battery(serial),
        found.battery,
        receipts.battery_level(serial),
    )
    _put_back(
        "timezone",
        lambda value: AdbControl.set_timezone(serial, value),
        found.timezone,
        receipts.timezone(serial),
    )
    if package:
        _put_back(
            "app locale",
            lambda value: AdbControl.set_app_locale(serial, package, value),
            found.locale,
            receipts.app_locale(serial, package),
        )
    _put_back(
        "the accessibility services",
        lambda value: AdbControl.set_accessibility_services(serial, value),
        found.accessibility,
        AdbControl.enabled_accessibility_services(serial),
    )


@contextmanager
def preserved(device_control, package: Optional[str] = None):
    """Hands the device back as the scenario found it, whatever ended it.

    Restoration is in a `finally`, so it runs after a pass, after a failure,
    and after an interruption. Only Android states are read and written: no
    capability Ronin applies to a simulator changes a state it can read back.
    """
    platform = getattr(device_control, "platform", None)
    serial = getattr(device_control, "serial", None) or ""
    if platform != Platform.ANDROID or not serial:
        yield None
        return

    package = package or getattr(device_control, "package_id", None)
    driver = getattr(device_control, "driver", None)
    commands = DriverCommands(driver) if driver is not None else None
    found = snapshot(serial, package, commands)
    logger.info("Device state before the scenario: %s", found.describe())
    try:
        yield found
    finally:
        restore(serial, package, found, commands)
