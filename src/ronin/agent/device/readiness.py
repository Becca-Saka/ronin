"""Confirming a device can be driven, before a run spends anything on it.

Scenario generation costs a model call and runs before anything touches the
device, so these checks have to come first.
"""

import logging

from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.platform import Platform
from ronin.errors import DeviceUnavailable

logger = logging.getLogger(__name__)

HEALTHY = "device"
"""What adb calls a device it can talk to. Anything else cannot be driven."""


def _startable(device_name: str) -> None:
    """Refuses a named device that is neither running nor one Ronin can start.

    The file names the device a run is for. If that device is not up and no
    emulator of that name exists, nothing can make the run meaningful.

    Raises:
        DeviceUnavailable: If no emulator of that name exists.
    """
    avds = AdbControl.list_avds()
    if avds is None:
        logger.warning(
            "Could not list emulators, so '%s' is taken on trust.", device_name
        )
        return
    if device_name in avds:
        logger.info("'%s' is not running. It will be started for this run.", device_name)
        return
    available = ", ".join(avds) if avds else "none"
    raise DeviceUnavailable(
        f"No device or emulator named '{device_name}'. Available: {available}.\n"
        f"Attach it, or set one that exists: ronin config set device <name>"
    )


def _android_ready(device_name: str) -> None:
    """Refuses to start when the named device cannot be driven.

    Raises:
        DeviceUnavailable: If the device is unusable, still booting, or
            cannot be started at all.
    """
    serial = AdbControl.resolve_emulator_serial(device_name)
    if serial is not None:
        if not AdbControl.boot_completed(serial):
            raise DeviceUnavailable(
                f"'{device_name}' ({serial}) has not finished booting.\n"
                "Wait for the home screen, then run again."
            )
        return

    unusable = {
        name: state
        for name, state in AdbControl.attached_devices().items()
        if state != HEALTHY
    }
    if unusable:
        # An offline device cannot report its AVD name, so there is no telling
        # whether it is the one the run was for.
        listed = ", ".join(f"{name} is {state}" for name, state in sorted(unusable.items()))
        raise DeviceUnavailable(
            f"No usable device named '{device_name}'. adb reports: {listed}.\n"
            "Reconnect it, or restart the emulator, then run again."
        )
    _startable(device_name)


def ensure_device_ready(platform: Platform, device_name: str) -> None:
    """Checks the device before a run commits to it.

    Raises:
        DeviceUnavailable: If the target cannot be driven.
    """
    if platform == Platform.ANDROID:
        _android_ready(device_name)
