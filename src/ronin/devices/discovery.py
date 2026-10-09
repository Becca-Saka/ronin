"""Enumerating the devices Ronin could target."""

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from ronin.doctor import is_macos, resolve_android_binary
from ronin.utils.subprocess_runner import run_command

RUNNING = "running"
SHUTDOWN = "shutdown"


@dataclass
class Device:
    """One targetable device: an AVD, a simulator, or a plugged-in handset."""

    name: str
    platform: str
    kind: str
    state: str
    identifier: Optional[str] = None
    runtime: Optional[str] = None


def _capture(cmd: list[str], timeout: float = 30.0) -> Optional[subprocess.CompletedProcess]:
    try:
        return run_command(cmd, timeout=timeout, log_command=False)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None


def _avd_name_for_serial(serial: str) -> Optional[str]:
    result = _capture(["adb", "-s", serial, "emu", "avd", "name"], timeout=10.0)
    if result is None or result.returncode != 0:
        return None
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return lines[0] if lines else None


def _model_for_serial(serial: str) -> str:
    result = _capture(["adb", "-s", serial, "shell", "getprop", "ro.product.model"], timeout=10.0)
    model = (result.stdout or "").strip() if result else ""
    return model or serial


def android_devices() -> list[Device]:
    """AVDs from the SDK, plus whatever adb currently sees."""
    devices: dict[str, Device] = {}

    emulator = resolve_android_binary("emulator")
    if emulator is not None:
        result = _capture([str(emulator), "-list-avds"])
        for name in (result.stdout or "").splitlines() if result else []:
            name = name.strip()
            if name:
                devices[name] = Device(name, "android", "avd", SHUTDOWN)

    result = _capture(["adb", "devices", "-l"])
    for line in (result.stdout or "").splitlines()[1:] if result else []:
        match = re.match(r"^(\S+)\s+(\S+)", line.strip())
        if not match:
            continue
        serial, state = match.groups()
        if state != "device":
            devices.setdefault(serial, Device(serial, "android", "physical", state, serial))
            continue
        if serial.startswith("emulator-"):
            avd = _avd_name_for_serial(serial)
            if avd and avd in devices:
                devices[avd].state = RUNNING
                devices[avd].identifier = serial
            else:
                devices[avd or serial] = Device(avd or serial, "android", "avd", RUNNING, serial)
        else:
            devices[serial] = Device(_model_for_serial(serial), "android", "physical", RUNNING, serial)

    return sorted(devices.values(), key=lambda device: device.name.lower())


def _runtime_label(runtime: str) -> str:
    identifier = runtime.split(".")[-1]
    family, _, version = identifier.partition("-")
    return f"{family} {version.replace('-', '.')}" if version else family


def ios_devices(include_wearables: bool = False) -> list[Device]:
    """Available simulators, with their UDIDs and runtimes."""
    if not is_macos():
        return []
    result = _capture(["xcrun", "simctl", "list", "devices", "available", "--json"])
    if result is None or result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return []

    devices = []
    for runtime, entries in (payload.get("devices") or {}).items():
        label = _runtime_label(runtime)
        if not include_wearables and not label.startswith("iOS"):
            continue
        for entry in entries:
            devices.append(
                Device(
                    name=entry.get("name", "?"),
                    platform="ios",
                    kind="simulator",
                    state=RUNNING if entry.get("state") == "Booted" else SHUTDOWN,
                    identifier=entry.get("udid"),
                    runtime=label,
                )
            )
    return sorted(devices, key=lambda device: (device.runtime or "", device.name))


def list_devices(
    platforms: tuple[str, ...] = ("android", "ios"), include_wearables: bool = False
) -> list[Device]:
    devices: list[Device] = []
    if "android" in platforms:
        devices.extend(android_devices())
    if "ios" in platforms:
        devices.extend(ios_devices(include_wearables))
    return devices
