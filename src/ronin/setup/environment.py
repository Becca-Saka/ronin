"""How this copy of Ronin was installed, and how to add packages to it.

`ronin setup` has to install the vision extra into Ronin's *own* environment,
which differs by install method: a uv-managed tool, a pipx venv, a project
checkout driven by uv.lock, or a plain virtualenv. Guessing wrong either fails
or installs packages that the next `uv sync` quietly removes.
"""

import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

VISION_PACKAGES = ("ultralytics", "easyocr", "huggingface-hub")


class InstallKind(str, Enum):
    CLONE = "clone"
    UV_TOOL = "uv-tool"
    PIPX = "pipx"
    VENV = "venv"
    SYSTEM = "system"


@dataclass
class Environment:
    """Where Ronin lives and the command that adds the vision extra to it."""

    kind: InstallKind
    prefix: Path
    project_root: Optional[Path] = None

    @property
    def description(self) -> str:
        if self.kind is InstallKind.CLONE:
            return f"project checkout at {self.project_root}"
        return f"{self.kind.value} environment at {self.prefix}"

    def vision_install_command(self) -> Optional[list[str]]:
        """The command that installs the vision extra, or None if we cannot tell.

        A checkout syncs from uv.lock, since installing into its venv directly
        would be undone by the next `uv sync`. Everything else gets the
        packages added to the environment currently running.
        """
        if self.kind is InstallKind.CLONE:
            if not shutil.which("uv"):
                return None
            return ["uv", "sync", "--extra", "vision"]

        if self.kind is InstallKind.PIPX and shutil.which("pipx"):
            return ["pipx", "inject", "ronin-ai", *VISION_PACKAGES]

        if shutil.which("uv"):
            return ["uv", "pip", "install", "--python", sys.executable, *VISION_PACKAGES]

        return [sys.executable, "-m", "pip", "install", *VISION_PACKAGES]

    @property
    def install_cwd(self) -> Optional[Path]:
        """Working directory the install command must run from."""
        return self.project_root if self.kind is InstallKind.CLONE else None


def _find_project_root() -> Optional[Path]:
    """Locates the checkout root when running from source (src/ronin/... layout)."""
    package_dir = Path(__file__).resolve().parent.parent  # .../src/ronin
    for candidate in (package_dir.parent.parent, package_dir.parent):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "ronin").is_dir():
            return candidate
    return None


def detect_environment() -> Environment:
    """Classifies the current interpreter's environment."""
    prefix = Path(sys.prefix)
    parts = set(prefix.parts)

    project_root = _find_project_root()
    if project_root is not None:
        return Environment(InstallKind.CLONE, prefix, project_root)
    if "tools" in parts and "uv" in parts:
        return Environment(InstallKind.UV_TOOL, prefix)
    if "pipx" in parts:
        return Environment(InstallKind.PIPX, prefix)
    if sys.prefix != sys.base_prefix:
        return Environment(InstallKind.VENV, prefix)
    return Environment(InstallKind.SYSTEM, prefix)
