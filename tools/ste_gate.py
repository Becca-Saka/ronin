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


SENTENCE_LENGTH = 40
"""How long an assigned string has to be before it counts as prose.

A name, a path and a log key are strings too. The rule has to separate those
from a prompt and a field description without holding a list of which names
carry prose, because a new prompt would not be on the list.
"""


def _assigned_prose(node: ast.AST) -> str | None:
    """A string assigned to a name, where it is long enough to be a sentence.

    A prompt lives in a module constant and a model-facing description lives
    in `Field(description=...)`. Both are read by a model, so both are prose
    this gate has to hold, and neither is a docstring or a comment.
    """
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    if not isinstance(node.value, ast.Constant):
        return None
    value = node.value.value
    if isinstance(value, str) and " " in value and len(value) >= SENTENCE_LENGTH:
        return value
    return None


def _described(node: ast.AST) -> list[str]:
    """Every `description=` a call passes as a plain string."""
    if not isinstance(node, ast.Call):
        return []
    return [
        keyword.value.value
        for keyword in node.keywords
        if keyword.arg == "description"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    ]


def prose_of(path: Path) -> str:
    """Every word in one file that a person or a model reads as English.

    Docstrings and comments, the long strings assigned to a name, and every
    `description=` a call carries. A prompt and a field description both
    reach a model, so a gate that read only docstrings passed the text that
    matters most.
    """
    source = path.read_text()
    found: list[str] = []
    tree = ast.parse(source)

    # `ast.get_docstring` dedents what it returns, and the same string read off
    # the node does not. Both reach the linter, the two copies differ by their
    # indentation, and every docstring is then counted twice. Collect the
    # nodes that hold a docstring and read each one once.
    carries_docstring = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                carries_docstring.add(id(first))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                found.append(doc)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and id(node) not in carries_docstring:
                found.append(node.value.value)

    for node in ast.walk(tree):
        assigned = _assigned_prose(node)
        if assigned:
            found.append(assigned)
        found.extend(_described(node))

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


def _named(path: Path) -> str:
    """The name a baseline entry uses: relative to the repository where it can be.

    A path outside the repository keeps its own spelling. `relative_to`
    raises on one, and a gate that crashes on the file it was handed reports
    nothing about the files it could have read.
    """
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


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
        relative = _named(path)
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
