"""Enumerating the devices Ronin could target."""

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence

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


# One line of `xctrace list devices`: a name, a release, and a udid. The host
# Mac is listed in the same section with no release, which is what tells the
# two apart.
_ATTACHED = re.compile(
    r"^(?P<name>.+?)\s+\((?P<release>\d[\d.]*)\)\s+\((?P<udid>[0-9A-Fa-f-]{8,})\)$"
)


def _attached_from_xctrace(output: str) -> list[Device]:
    """The devices in the `== Devices ==` section, and nothing below it.

    The simulators follow under their own header, and they are enumerated by
    simctl instead, which reports their boot state.
    """
    devices = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("=="):
            if devices or "Devices" not in line:
                break
            continue
        found = _ATTACHED.match(line)
        if found:
            devices.append(
                Device(
                    name=found["name"],
                    platform="ios",
                    kind="physical",
                    state=RUNNING,
                    identifier=found["udid"],
                    runtime=f"iOS {found['release']}",
                )
            )
    return devices


def ios_physical_devices() -> list[Device]:
    """Plugged-in iPhones and iPads, or an empty list where none answers.

    `xctrace` is asked first, although libimobiledevice is the lighter tool.
    It ships with Xcode, which a run already needs, and it reports the name
    and the release beside the udid. `idevice_id` reports a udid alone, so it
    is the fallback rather than the first choice.

    `system_profiler` is not asked. It enumerates USB hardware rather than
    automatable targets, and the serial number it reports is not the udid a
    session needs.
    """
    if not is_macos():
        return []

    result = _capture(["xcrun", "xctrace", "list", "devices"])
    if result is not None and result.returncode == 0:
        found = _attached_from_xctrace(result.stdout or "")
        if found:
            return found

    result = _capture(["idevice_id", "-l"], timeout=10.0)
    if result is None or result.returncode != 0:
        return []
    return [
        Device(udid, "ios", "physical", RUNNING, udid)
        for udid in (line.strip() for line in (result.stdout or "").splitlines())
        if udid
    ]


def _runtime_label(runtime: str) -> str:
    identifier = runtime.split(".")[-1]
    family, _, version = identifier.partition("-")
    return f"{family} {version.replace('-', '.')}" if version else family


def ios_simulators(include_wearables: bool = False) -> list[Device]:
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


def ios_devices(include_wearables: bool = False) -> list[Device]:
    """Every iOS target this machine can reach, attached ones first.

    A plugged-in iPhone is listed before the simulators because it is the
    scarcer target, and because a run that names neither should see it.
    """
    return ios_physical_devices() + ios_simulators(include_wearables)


def addressable(device: Device) -> Optional[str]:
    """How a run names this target again, after the listing is gone.

    Each kind has one durable name. A simulator and a physical iPhone are
    addressed by udid. An AVD is launched by its own name, and the serial it
    answers to is assigned at boot, so the name is the durable one. A
    plugged-in Android phone has only its serial: the listing shows a model
    name that nothing can select.
    """
    if device.platform == "android" and device.kind == "avd":
        return device.name
    return device.identifier


def resolve(name: str, devices: Sequence[Device]) -> Optional[Device]:
    """The one listed device this name picks, or None where none does.

    An exact identifier wins. It names one target by definition, and nothing
    stops another device being called the same thing.

    Raises:
        UsageError: Two targets carry this name. A physical iPhone and a
            simulator can both be called the same thing. Picking either one
            would run the suite against a device nobody chose.
    """
    from ronin.errors import UsageError

    for device in devices:
        if device.identifier and name == device.identifier:
            return device

    matches = [device for device in devices if device.name == name]
    if len(matches) > 1:
        spelled = "\n".join(
            f"  {device.identifier or device.name}  ({device.kind})"
            for device in matches
        )
        raise UsageError(
            f"'{name}' names {len(matches)} targets. Name one of these "
            f"instead:\n{spelled}"
        )
    return matches[0] if matches else None


def identifier_for(name: str, platform: str) -> Optional[str]:
    """The canonical id of the one listed target with this name, or None.

    A person types the name they read in `ronin devices`, and a session has
    to address the target itself. An id typed in place of a name answers with
    itself, and wins over a name, because an id names one target by
    definition.

    Raises:
        UsageError: Two targets carry this name. A physical iPhone and a
            simulator can both be called the same thing. Picking either one
            would run the suite against a device nobody chose.
    """
    found = resolve(name, list_devices((platform,)))
    return found.identifier if found else None


def list_devices(
    platforms: tuple[str, ...] = ("android", "ios"), include_wearables: bool = False
) -> list[Device]:
    devices: list[Device] = []
    if "android" in platforms:
        devices.extend(android_devices())
    if "ios" in platforms:
        devices.extend(ios_devices(include_wearables))
    return devices
