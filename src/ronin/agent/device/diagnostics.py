"""Explaining why an emulator failed to start."""

import logging
import shutil
import subprocess

from ronin.utils.platform_checks import MIN_EMULATOR_FREE_GB, free_disk_gb
from ronin.utils.subprocess_runner import run_command

logger = logging.getLogger(__name__)


def _avd_names() -> list[str]:
    emulator = shutil.which("emulator")
    if emulator is None:
        return []
    try:
        result = run_command([emulator, "-list-avds"], timeout=20, log_command=False)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def diagnose_avd_launch(device_name: str) -> str:
    """Returns the most likely reason the named AVD did not come up."""
    reasons = []

    free = free_disk_gb()
    if free < MIN_EMULATOR_FREE_GB:
        reasons.append(
            f"Only {free:.1f}GB free on this volume — the emulator refuses to boot below "
            f"about {MIN_EMULATOR_FREE_GB:.0f}GB. This is the usual cause."
        )

    names = _avd_names()
    if names and device_name not in names:
        reasons.append(
            f"No AVD named '{device_name}'. Available: {', '.join(names)}. "
            f"Run: ronin config set device {names[0]}"
        )
    elif not names and shutil.which("emulator"):
        reasons.append("No AVDs are configured. Run: ronin devices create")

    if not reasons:
        reasons.append(
            f"Start it manually to see the emulator's own error: emulator -avd {device_name}"
        )

    reasons.append("Run 'ronin doctor' to check the rest of the environment.")
    return "\n".join(f"  - {reason}" for reason in reasons)
