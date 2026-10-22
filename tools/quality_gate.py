#!/usr/bin/env python3
"""Enforce Ronin's file-size ratchet, with no dependencies.

Complexity is ruff's C901, configured in pyproject.toml. Counting it here as
well gave one function two numbers from two algorithms that disagree. This one
charged for boolean operands and comprehensions, which McCabe ignores, and
skipped nested functions, which McCabe does not. File size is the part ruff has
no rule for, so that is the part which stays.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib

MAX_FILE_LINES = 550
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().with_name("quality-baseline.toml")


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else str(self.path)
        return f"{location}: {self.message}"


def _load_baseline() -> dict[str, dict[str, int]]:
    with BASELINE_PATH.open("rb") as baseline_file:
        return tomllib.load(baseline_file)


def _python_files(targets: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for target in targets:
        if target.is_dir():
            files.update(target.rglob("*.py"))
        elif target.suffix == ".py":
            files.add(target)
    return sorted(files)


def _relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return path.resolve()


def _check_file(path: Path, oversized_files: dict[str, int]) -> list[Violation]:
    relative_path = _relative(path)
    source = path.read_text(encoding="utf-8")
    line_count = len(source.splitlines())
    relative_name = relative_path.as_posix()
    allowed_lines = oversized_files.get(relative_name, MAX_FILE_LINES)
    violations: list[Violation] = []

    if line_count > allowed_lines:
        reason = "baseline ceiling" if relative_name in oversized_files else "limit"
        violations.append(
            Violation(
                relative_path,
                0,
                f"{line_count} lines exceeds {reason} of {allowed_lines}",
            )
        )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            REPOSITORY_ROOT / "src" / "ronin",
            REPOSITORY_ROOT / "tests",
            # The eval harness, and not the rest of `tools`: `ste_lint.py` is
            # vendored and sits above the ceiling.
            REPOSITORY_ROOT / "tools" / "eval",
            REPOSITORY_ROOT / "tools" / "eval_gate.py",
        ],
        help="Python files or directories to inspect (default: src/ronin and tests)",
    )
    args = parser.parse_args(argv)

    targets = [path if path.is_absolute() else REPOSITORY_ROOT / path for path in args.paths]
    violations: list[Violation] = []
    baseline = _load_baseline()
    oversized_files = baseline.get("oversized_files", {})
    for path in _python_files(targets):
        violations.extend(_check_file(path, oversized_files))

    if violations:
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        return 1

    print(f"Quality gate passed for {len(_python_files(targets))} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
