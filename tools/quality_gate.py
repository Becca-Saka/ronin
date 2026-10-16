#!/usr/bin/env python3
"""Enforce Ronin's dependency-free complexity and file-size ratchets."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib

MAX_COMPLEXITY = 21  # The project requirement is strictly less than 22.
MAX_FILE_LINES = 499  # The project requirement is strictly less than 500.
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


class _ComplexityCounter(ast.NodeVisitor):
    """Count decision points inside one function, excluding nested functions."""

    def __init__(self) -> None:
        self.value = 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.value += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.value += len(node.handlers)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.value += len(node.cases)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.value += 1 + len(node.ifs)
        self.generic_visit(node)


def _function_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    counter = _ComplexityCounter()
    for statement in node.body:
        counter.visit(statement)
    return counter.value


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


def _check_file(
    path: Path,
    oversized_files: dict[str, int],
    complex_functions: dict[str, int],
) -> list[Violation]:
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

    try:
        tree = ast.parse(source, filename=str(relative_path))
    except SyntaxError as error:
        violations.append(
            Violation(relative_path, error.lineno or 0, f"cannot parse: {error.msg}")
        )
        return violations

    functions = (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    for function in functions:
        complexity = _function_complexity(function)
        baseline_key = f"{relative_name}:{function.name}"
        allowed_complexity = complex_functions.get(baseline_key, MAX_COMPLEXITY)
        if complexity > allowed_complexity:
            reason = (
                "baseline ceiling"
                if baseline_key in complex_functions
                else "maximum"
            )
            violations.append(
                Violation(
                    relative_path,
                    function.lineno,
                    f"{function.name} has cyclomatic complexity {complexity}; {reason} is {allowed_complexity}",
                )
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[REPOSITORY_ROOT / "src" / "ronin"],
        help="Python files or directories to inspect (default: src/ronin)",
    )
    args = parser.parse_args(argv)

    targets = [path if path.is_absolute() else REPOSITORY_ROOT / path for path in args.paths]
    violations: list[Violation] = []
    baseline = _load_baseline()
    oversized_files = baseline.get("oversized_files", {})
    complex_functions = baseline.get("complex_functions", {})
    for path in _python_files(targets):
        violations.extend(_check_file(path, oversized_files, complex_functions))

    if violations:
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        return 1

    print(f"Quality gate passed for {len(_python_files(targets))} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
