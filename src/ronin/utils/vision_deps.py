"""Access to the optional vision dependencies.

YOLO detection and OCR pull in torch, which is large enough that it is not
part of the base install. These helpers keep the rest of the codebase
importable without them, and turn a missing install into an actionable
message instead of a bare ``ModuleNotFoundError`` surfacing mid-run.
"""

from importlib.util import find_spec

VISION_PACKAGES = ("ultralytics", "easyocr", "huggingface_hub")

INSTALL_HINT = (
    "Vision dependencies are not installed. Install them with:\n"
    "  uv sync --extra vision          (from a clone)\n"
    "  pip install 'ronin-ai[vision]'  (installed)"
)


class VisionDependencyError(ImportError):
    """Raised when YOLO/OCR are needed but the vision extra is not installed."""


def missing_vision_packages() -> list[str]:
    """Returns the vision packages that are not importable, in declaration order."""
    return [name for name in VISION_PACKAGES if find_spec(name) is None]


def vision_available() -> bool:
    """True when every optional vision dependency is importable."""
    return not missing_vision_packages()


def require_vision() -> None:
    """Raises VisionDependencyError if the vision extra is not installed."""
    missing = missing_vision_packages()
    if missing:
        raise VisionDependencyError(f"{INSTALL_HINT}\n\nMissing: {', '.join(missing)}")
