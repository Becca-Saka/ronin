"""Resolved settings for every Ronin command.

Precedence, strongest first: an explicit CLI flag, a RONIN_* environment
variable, ./ronin.yaml in the working directory or any parent, the global
~/.ronin/config.yaml, then the built-in defaults. Each value remembers where
it came from so `ronin config` can explain itself.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from ronin.defaults import (
    DEFAULT_APP_PATH,
    DEFAULT_DEVICE,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLATFORM,
    DEFAULT_SETTLE_TIME,
)

PROJECT_CONFIG_FILENAME = "ronin.yaml"
GLOBAL_CONFIG_PATH = Path.home() / ".ronin" / "config.yaml"
ENV_PREFIX = "RONIN_"

DEFAULT_ORIGIN = "built-in default"


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


SETTINGS: dict[str, Callable[[Any], Any]] = {
    "platform": str,
    "device": str,
    "model": str,
    "fast_model": str,
    "app_path": str,
    "output_dir": str,
    "max_steps": int,
    "settle_time": float,
    "headless": _as_bool,
    "use_multimodal_vision": _as_bool,
}

BUILT_IN: dict[str, Any] = {
    "platform": DEFAULT_PLATFORM,
    "device": DEFAULT_DEVICE,
    "model": DEFAULT_MODEL,
    "fast_model": None,
    "app_path": DEFAULT_APP_PATH,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "max_steps": DEFAULT_MAX_STEPS,
    "settle_time": DEFAULT_SETTLE_TIME,
    "headless": False,
    "use_multimodal_vision": False,
}


class ConfigError(Exception):
    """Raised for an unknown setting or a value that will not coerce."""


@dataclass
class Config:
    """Effective settings, with the origin of each value."""

    values: dict[str, Any] = field(default_factory=dict)
    origins: dict[str, str] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)

    def get(self, name: str) -> Any:
        return self.values[name]

    def origin(self, name: str) -> str:
        return self.origins.get(name, DEFAULT_ORIGIN)

    def platforms(self, requested: Optional[str] = None) -> tuple[str, ...]:
        """Which platforms a command covers.

        Without an explicit choice, follow the configured target rather than
        assuming both: an Android-only user should not be told that a missing
        iOS driver blocks them.
        """
        choice = requested or self.values["platform"]
        return ("android", "ios") if choice == "all" else (choice,)

    def resolve(self, name: str, override: Any = None) -> Any:
        """An explicit CLI value wins; otherwise the resolved setting applies."""
        if override is None or override is False:
            return self.values[name]
        return override


def coerce(name: str, raw: Any) -> Any:
    if name not in SETTINGS:
        raise ConfigError(f"Unknown setting '{name}'. Known: {', '.join(sorted(SETTINGS))}")
    if raw is None:
        return None
    try:
        return SETTINGS[name](raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value for '{name}': {raw!r} ({exc})") from exc


def read_file(path: Path) -> dict[str, Any]:
    """Reads one config file, ignoring unknown keys so a newer file stays usable."""
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping of settings")
    return {key: value for key, value in loaded.items() if key in SETTINGS}


def find_project_config(start: Optional[Path] = None) -> Optional[Path]:
    """Looks for ronin.yaml in this directory and its parents."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _env_overrides() -> dict[str, tuple[Any, str]]:
    found = {}
    for name in SETTINGS:
        variable = f"{ENV_PREFIX}{name.upper()}"
        raw = os.getenv(variable)
        if raw is not None:
            found[name] = (raw, variable)
    return found


def load_config(start: Optional[Path] = None) -> Config:
    """Builds the effective configuration from every layer."""
    config = Config(values=dict(BUILT_IN))
    config.origins = {name: DEFAULT_ORIGIN for name in BUILT_IN}

    for path in (GLOBAL_CONFIG_PATH, find_project_config(start)):
        if path is None:
            continue
        settings = read_file(path)
        if settings:
            config.files.append(path)
        for name, raw in settings.items():
            config.values[name] = coerce(name, raw)
            config.origins[name] = str(path)

    for name, (raw, variable) in _env_overrides().items():
        config.values[name] = coerce(name, raw)
        config.origins[name] = variable

    return config


def config_path(scope: str = "global", start: Optional[Path] = None) -> Path:
    """The file a write targets: the global file, or ronin.yaml in this project."""
    if scope == "global":
        return GLOBAL_CONFIG_PATH
    existing = find_project_config(start)
    return existing or (start or Path.cwd()) / PROJECT_CONFIG_FILENAME


def set_value(name: str, raw: Any, scope: str = "global", start: Optional[Path] = None) -> Path:
    """Writes one setting, creating the file if needed, and returns its path."""
    value = coerce(name, raw)
    path = config_path(scope, start)
    settings = read_file(path)
    settings[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings, sort_keys=True), encoding="utf-8")
    return path


def unset_value(name: str, scope: str = "global", start: Optional[Path] = None) -> Path:
    """Removes one setting from a config file."""
    if name not in SETTINGS:
        raise ConfigError(f"Unknown setting '{name}'")
    path = config_path(scope, start)
    settings = read_file(path)
    settings.pop(name, None)
    if settings:
        path.write_text(yaml.safe_dump(settings, sort_keys=True), encoding="utf-8")
    elif path.is_file():
        path.unlink()
    return path
