"""Creating Android virtual devices, installing what the SDK is missing."""

import platform as platform_module
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ronin.doctor import android_sdk_path
from ronin.utils.subprocess_runner import run_command

PREFERRED_TAGS = ("google_apis_playstore", "google_apis", "default", "aosp_atd")
FALLBACK_API_LEVEL = 35


@dataclass(frozen=True)
class SystemImage:
    """An emulator system image, installed or installable."""

    api: int
    tag: str
    abi: str

    @property
    def package(self) -> str:
        return f"system-images;android-{self.api};{self.tag};{self.abi}"

    def __str__(self) -> str:
        return f"android-{self.api} / {self.tag} / {self.abi}"


class AndroidToolingError(Exception):
    """Raised when the SDK is missing a tool this operation needs."""


def host_abi() -> str:
    return "arm64-v8a" if platform_module.machine() in ("arm64", "aarch64") else "x86_64"


def sdk_tool(name: str) -> Optional[Path]:
    sdk = android_sdk_path()
    if sdk is None:
        return None
    for relative in (f"cmdline-tools/latest/bin/{name}", f"tools/bin/{name}", f"cmdline-tools/bin/{name}"):
        candidate = sdk / relative
        if candidate.is_file():
            return candidate
    return None


def require_tool(name: str) -> Path:
    tool = sdk_tool(name)
    if tool is None:
        raise AndroidToolingError(
            f"{name} not found in the Android SDK. Install 'Android SDK Command-line Tools' "
            "from Android Studio > SDK Manager > SDK Tools."
        )
    return tool


def installed_system_images() -> list[SystemImage]:
    sdk = android_sdk_path()
    if sdk is None:
        return []
    images = []
    for path in sorted((sdk / "system-images").glob("android-*/*/*")):
        if not path.is_dir():
            continue
        match = re.match(r"android-(\d+)$", path.parent.parent.name)
        if match:
            images.append(SystemImage(int(match.group(1)), path.parent.name, path.name))
    return sorted(images, key=lambda image: (image.api, PREFERRED_TAGS.index(image.tag) if image.tag in PREFERRED_TAGS else 99), reverse=True)


def device_profiles() -> list[str]:
    tool = sdk_tool("avdmanager")
    if tool is None:
        return []
    try:
        result = run_command([str(tool), "list", "device"], timeout=60, log_command=False)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []
    return re.findall(r'^id: \d+ or "([^"]+)"', result.stdout or "", flags=re.MULTILINE)


def best_profile(requested: Optional[str] = None) -> Optional[str]:
    """Picks a hardware profile, preferring the newest Pixel this SDK knows."""
    profiles = device_profiles()
    if requested:
        return requested if requested in profiles else None
    pixels = [(int(match.group(1)), name) for name in profiles if (match := re.fullmatch(r"pixel_(\d+)", name))]
    if pixels:
        return max(pixels)[1]
    return profiles[0] if profiles else None


def pick_image(api: Optional[int] = None) -> tuple[Optional[SystemImage], SystemImage]:
    """Returns the installed image to use, and the image to install if none fits.

    The second value is what `ronin devices create` offers to download.
    """
    abi = host_abi()
    candidates = [image for image in installed_system_images() if image.abi == abi]
    if api is not None:
        candidates = [image for image in candidates if image.api == api]
    wanted = SystemImage(api or FALLBACK_API_LEVEL, PREFERRED_TAGS[1], abi)
    return (candidates[0] if candidates else None), wanted


def _run(cmd: list[str], feed: Optional[str] = None) -> int:
    """Runs an SDK tool, answering its interactive prompts when given a feed."""
    try:
        completed = subprocess.run(cmd, input=feed, text=True if feed is not None else None)
        return completed.returncode
    except (FileNotFoundError, OSError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1


def install_system_image(image: SystemImage) -> int:
    """Downloads a system image, accepting the SDK licences it prompts for."""
    sdkmanager = require_tool("sdkmanager")
    return _run([str(sdkmanager), "--install", image.package], feed="y\n" * 20)


def create_avd(name: str, image: SystemImage, profile: Optional[str]) -> int:
    """Creates an AVD, declining avdmanager's custom-hardware-profile prompt."""
    avdmanager = require_tool("avdmanager")
    command = [str(avdmanager), "create", "avd", "-n", name, "-k", image.package, "--force"]
    if profile:
        command += ["-d", profile]
    return _run(command, feed="no\n")
