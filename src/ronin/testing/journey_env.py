"""Journey-declared values, and which of them are sensitive.

Guessing at secrets from quoted literals masked button labels and screen names
far more often than credentials. A journey declares its values instead:

    env:
      email: qa.tester@example.com
      password:
        value: ${RONIN_TEST_PASSWORD}
        secret: true

A plain value is ordinary; mark one secret where it is declared. A top-level
`secrets:` list of names does the same thing for values declared plainly.

`${VAR}` reads from the environment, so a real credential never has to live in
a file that gets committed. Journey text refers to a value as `{{email}}`.

An entry may also mark itself as one this app refuses:

    env:
      rejected_password:
        value: abc
        secret: true
        refused: true

An entry says nothing by default, which is almost every entry: the value is
meant to work. `refused: true` is a fact about the app that no reading of the
value could find, such as a password under its own length floor. A scenario
names the value, and the declaration answers for it, so nothing has to assert
a refusal it cannot support.
"""

import logging
import os
import re
from typing import Dict, FrozenSet, Tuple

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")
"""A name in journey text or in a step, such as `{{actors.editor.email}}`.

A dot reaches inside a declared account. One namespace then covers every
value a scenario may name, so nothing has to know where a name came from.
"""
ENV_REFERENCE = re.compile(r"\$\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}")

logger = logging.getLogger(__name__)


def _expand(name: str, value: str) -> str:
    """Resolves ${VAR} from the environment, refusing to run without it.

    Substituting an empty string would turn a missing password into a
    blank-password test and report the app as broken for failing it.
    """
    missing = [
        match.group(1)
        for match in ENV_REFERENCE.finditer(value)
        if os.environ.get(match.group(1)) is None
    ]
    if missing:
        raise ValueError(
            f"env.{name} refers to environment variable(s) that are not set: "
            + ", ".join(sorted(set(missing)))
        )
    return ENV_REFERENCE.sub(lambda match: os.environ[match.group(1)], value)


def _entry(name: str, declared) -> Tuple[str, bool, bool]:
    """Reads one env entry: the value, whether it is secret, whether it is refused."""
    if not isinstance(declared, dict):
        return _expand(name, str(declared)), False, False

    unexpected = set(declared) - {"value", "secret", "refused"}
    if unexpected:
        raise ValueError(
            f"env.{name} has unsupported key(s): " + ", ".join(sorted(unexpected))
        )
    if "value" not in declared:
        raise ValueError(f"env.{name} is a mapping, so it needs a 'value'.")
    return (
        _expand(name, str(declared["value"])),
        bool(declared.get("secret", False)),
        bool(declared.get("refused", False)),
    )


def resolve(data: dict) -> Tuple[Dict[str, str], Dict[str, str], FrozenSet[str]]:
    """A journey's declared values, which of them are secret, and which are refused."""
    declared = data.get("env") or {}
    if not isinstance(declared, dict):
        raise ValueError("'env' must be a mapping of name to value.")

    values: Dict[str, str] = {}
    secret_names = set()
    refused = set()
    for name, entry in declared.items():
        name = str(name)
        values[name], is_secret, is_refused = _entry(name, entry)
        if is_secret:
            secret_names.add(name)
        if is_refused:
            refused.add(name)

    listed = data.get("secrets") or []
    if isinstance(listed, str):
        listed = [listed]
    unknown = [name for name in listed if name not in values]
    if unknown:
        raise ValueError(
            "'secrets' names values missing from 'env': " + ", ".join(sorted(unknown))
        )
    secret_names.update(listed)

    secrets = {name: values[name] for name in sorted(secret_names) if values[name]}
    return values, secrets, frozenset(refused)


def substitute(text: str, values: Dict[str, str]) -> str:
    """Replaces {{name}} with its declared value, leaving unknown names alone."""
    if not text or not values:
        return text or ""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            logger.warning("Journey text refers to {{%s}}, which 'env' does not define.", name)
            return match.group(0)
        return values[name]

    return PLACEHOLDER.sub(replace, text)
