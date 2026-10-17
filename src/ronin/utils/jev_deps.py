"""Access to the optional decision-model dependency.

Grounding against a decision model needs its vendor SDK. A run that never
grounds never uses it. These helpers keep the rest of the codebase importable
without it, and turn a missing install into an actionable message instead of a
bare ``ModuleNotFoundError`` surfacing mid-run.
"""

from importlib.util import find_spec

JEV_PACKAGES = ("typesafe_sdk",)

INSTALL_HINT = (
    "The decision-model dependency is not installed. Install it with:\n"
    "  uv sync --extra jev          (from a clone)\n"
    "  pip install 'ronin-ai[jev]'  (installed)"
)


class JevDependencyError(ImportError):
    """Raised when JEV is needed but the jev extra is not installed."""


def missing_jev_packages() -> list[str]:
    """Returns the JEV packages that are not importable, in declaration order."""
    return [name for name in JEV_PACKAGES if find_spec(name) is None]


def jev_available() -> bool:
    """True when every optional JEV dependency is importable."""
    return not missing_jev_packages()


def require_jev() -> None:
    """Raises JevDependencyError if the jev extra is not installed."""
    missing = missing_jev_packages()
    if missing:
        raise JevDependencyError(f"{INSTALL_HINT}\n\nMissing: {', '.join(missing)}")
