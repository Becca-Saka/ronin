#!/usr/bin/env python3
"""Enforce controlled language in the prose Ronin ships, with no dependencies.

The agent-development skill asks for short active sentences, one instruction
per sentence, and one name for one thing. Those rules are checked by the
ASD-STE100 skill's own linter, so this gate finds that linter and runs it
rather than holding a second copy of the rules that can disagree with it.

What this file owns is the extraction. A linter reading a `.py` file counts
identifiers and argv as words and reports on code nobody wrote as prose, so
only the docstrings and the comments are passed to it.

The linter is a personal tool rather than a dependency of this repository. A
checkout without it reports that and passes, because a gate that fails for a
missing tool stops work it cannot speak about.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().with_name("ste-baseline.toml")

LINTER_ENV = "RONIN_STE_LINT"
DEFAULT_LINTER = Path.home() / ".claude/skills/asd-ste100/scripts/ste-lint.py"

COMMENT = re.compile(r"\s*#\s?(.+)")
MACHINERY = ("noqa", "type:", "ruff:", "pylint", "mypy", "fmt:")


def _find_linter() -> Path | None:
    """The skill's linter, or None when this machine does not have it."""
    named = os.environ.get(LINTER_ENV)
    if named:
        path = Path(named).expanduser()
        return path if path.is_file() else None
    return DEFAULT_LINTER if DEFAULT_LINTER.is_file() else None


def prose_of(path: Path) -> str:
    """Every docstring and comment in one file, and nothing else.

    Field docstrings count: a bare string under an assignment documents the
    field above it, and Ronin uses them for the reasoning behind a default.
    """
    source = path.read_text()
    found: list[str] = []
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                found.append(doc)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                found.append(node.value.value)

    for line in source.splitlines():
        match = COMMENT.match(line)
        if match and not match.group(1).startswith(MACHINERY):
            found.append(match.group(1))

    return "\n\n".join(dict.fromkeys(found))


def hard_violations(linter: Path, prose: str) -> tuple[int, list[str]]:
    """What the skill's linter says about one file's prose.

    Returns the number of hard violations and the lines describing them.
    Advisory findings are reported and do not count, because the linter does
    not fail on them either.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(prose)
        temporary = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(linter), str(temporary)],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        temporary.unlink(missing_ok=True)

    output = result.stdout + result.stderr
    count = 0
    summary = re.search(r"(\d+) violations \((\d+) hard", output)
    if summary:
        count = int(summary.group(2))
    reported = [
        line.split(str(temporary) + ":")[-1]
        for line in output.splitlines()
        if str(temporary) in line
        and not any(soft in line for soft in ("passive-voice", "compound-tense",
                                              "present-perfect"))
    ]
    return count, reported


def _load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.is_file():
        return {}
    with BASELINE_PATH.open("rb") as baseline_file:
        return tomllib.load(baseline_file).get("allowed_hard_violations", {})


def _python_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.suffix == ".py":
            files.append(target)
    return [f for f in files if "__pycache__" not in f.parts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["src/ronin"])
    args = parser.parse_args()

    linter = _find_linter()
    if linter is None:
        print(
            f"STE gate skipped: no linter at {DEFAULT_LINTER}. "
            f"Set {LINTER_ENV} to run it."
        )
        return 0

    baseline = _load_baseline()
    failures: list[str] = []
    checked = 0

    for path in _python_files([Path(p) for p in args.paths]):
        prose = prose_of(path)
        if not prose.strip():
            continue
        checked += 1
        count, reported = hard_violations(linter, prose)
        relative = str(path.relative_to(REPOSITORY_ROOT)) if path.is_absolute() else str(path)
        allowed = baseline.get(relative, 0)
        if count > allowed:
            failures.append(f"{relative}: {count} hard violation(s), allowed {allowed}")
            failures.extend(f"    {line.strip()}" for line in reported[:6])

    if failures:
        print("\n".join(failures))
        print(f"\nSTE gate failed across {checked} file(s).")
        return 1

    print(f"STE gate passed for {checked} file(s) with prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
