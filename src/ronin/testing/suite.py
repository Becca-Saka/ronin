"""Reading a ronin.yaml into the workflows it describes.

A workflow owns an app, a device and an environment. The journeys inside it
are only the tests, so one file can cover several apps without a journey
having to say where it runs.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from ronin.config import SETTING_ALIASES, STRUCTURE_KEYS, ConfigError, expand_device
from ronin.testing.journey_env import resolve, substitute
from ronin.testing.models import UserJourney
from ronin.schema import RoninFile, explain

logger = logging.getLogger(__name__)

PATH_SETTINGS = ("app_path", "output_dir")
"""Settings read as paths, resolved against the file that named them."""


@dataclass(frozen=True)
class Workflow:
    """One app, one device, and the journeys to run against them."""

    name: str
    settings: dict[str, Any] = field(default_factory=dict)
    journeys: list[UserJourney] = field(default_factory=list)


def _normalise(block: dict[str, Any], where: str) -> dict[str, Any]:
    """Settings from one level of the file, under the names Ronin resolves."""
    expanded = expand_device(block, where)
    settings = {}
    for key, value in expanded.items():
        if key in STRUCTURE_KEYS:
            continue
        settings[SETTING_ALIASES.get(key, key)] = value
    return settings


def _anchor(settings: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolves path settings against the directory holding the file.

    A workflow names its app beside itself, so `ronin test` finds the same
    binary wherever it is run from.
    """
    anchored = dict(settings)
    for name in PATH_SETTINGS:
        raw = anchored.get(name)
        if raw and not Path(str(raw)).is_absolute():
            anchored[name] = str((base / str(raw)).resolve())
    return anchored


def _journey(data: dict[str, Any], shared_env: dict[str, Any], index: int) -> UserJourney:
    """One test, with the workflow's environment behind its own."""
    merged = dict(data)
    merged["env"] = {**shared_env, **(data.get("env") or {})}
    values, secrets = resolve(merged)
    description = data.get("journey", data.get("description", ""))
    return UserJourney(
        id=str(data.get("id", f"journey_{index:02d}")),
        name=str(data.get("name", "User Journey")),
        description=substitute(description, values),
        security_rules=[
            substitute(rule, values) for rule in data.get("security_rules", [])
        ],
        secrets=secrets,
    )


def _journeys_of(block: dict[str, Any], shared_env: dict[str, Any], where: str) -> list[UserJourney]:
    listed = block.get("journeys")
    if listed is None and "journey" in block:
        listed = [block]
    if listed is None:
        raise ConfigError(f"{where} has no journeys.")
    if not isinstance(listed, list):
        raise ConfigError(f"'journeys' in {where} must be a list.")
    return [_journey(item, shared_env, index + 1) for index, item in enumerate(listed)]


def _shared_env(block: dict[str, Any]) -> dict[str, Any]:
    declared = block.get("env") or {}
    if not isinstance(declared, dict):
        raise ConfigError("'env' must be a mapping of name to value.")
    return declared


def _validate(raw: dict[str, Any], path: Path) -> None:
    """Refuses a file Ronin cannot read, naming every key that is wrong.

    Every problem at once, so one pass over the file fixes them all.

    Raises:
        ConfigError: If the file does not match the ronin.yaml schema.
    """
    try:
        RoninFile.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(f"  {line}" for line in explain(exc))
        raise ConfigError(f"{path} cannot be read:\n{problems}") from None


def load_workflows(path: Path, raw: Any = None) -> list[Workflow]:
    """Every workflow in a ronin.yaml, in the order the file lists them.

    A file with no `workflows:` is one unnamed workflow, so a journey file
    written before workflows existed still runs.

    Raises:
        ConfigError: If the file cannot be read as workflows.
    """
    if raw is None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.resolve().parent

    if isinstance(raw, list):
        return [Workflow("", {}, [_journey(item, {}, i + 1) for i, item in enumerate(raw)])]
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping or a list of journeys.")
    _validate(raw, path)

    top = _normalise(raw, str(path))
    shared = _shared_env(raw)

    declared = raw.get("workflows")
    if declared is None:
        return [Workflow("", _anchor(top, base), _journeys_of(raw, shared, str(path)))]
    if not isinstance(declared, dict):
        raise ConfigError(f"'workflows' in {path} must be a mapping of name to workflow.")

    workflows = []
    for name, block in declared.items():
        where = f"workflow '{name}'"
        if not isinstance(block, dict):
            raise ConfigError(f"{where} in {path} must be a mapping.")
        settings = {**top, **_normalise(block, where)}
        env = {**shared, **_shared_env(block)}
        workflows.append(
            Workflow(str(name), _anchor(settings, base), _journeys_of(block, env, where))
        )
    return workflows


def select(workflows: list[Workflow], wanted: Optional[str]) -> list[Workflow]:
    """The workflows a run covers, or all of them.

    Raises:
        ConfigError: If a name was asked for and the file does not have it.
    """
    if not wanted:
        return workflows
    names = [name.strip() for name in wanted.split(",") if name.strip()]
    by_name = {workflow.name: workflow for workflow in workflows}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ConfigError(
            f"No workflow named {', '.join(missing)}. "
            f"This file has: {', '.join(by_name) or '(none named)'}"
        )
    return [by_name[name] for name in names]
