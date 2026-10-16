#!/usr/bin/env python3
"""Enforce Ronin's type-checking ratchet on top of Pyright.

Pyright has no baseline of its own, so every file stays checked and the count
of recorded errors per file is what may not grow. Suppressing whole files would
hide new mistakes in exactly the modules that already have the most.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().with_name("type-baseline.toml")


@dataclass(frozen=True)
class Violation:
    path: str
    found: int
    allowed: int

    def render(self) -> str:
        if self.allowed:
            return f"{self.path}: {self.found} type errors exceeds baseline of {self.allowed}"
        return f"{self.path}: {self.found} type errors; no baseline allowance"


def _load_baseline() -> dict[str, int]:
    with BASELINE_PATH.open("rb") as baseline_file:
        return tomllib.load(baseline_file).get("files", {})


def _run_pyright(paths: list[str]) -> dict:
    executable = shutil.which("pyright")
    if executable is None:
        raise SystemExit(
            "pyright is not on PATH. Install the dev group: uv sync --group dev"
        )
    completed = subprocess.run(
        [executable, "--outputjson", "--level", "error", *paths],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise SystemExit(
            f"pyright produced no report (exit {completed.returncode}):\n{completed.stderr}"
        ) from None


def _errors_per_file(report: dict) -> Counter[str]:
    counts: Counter[str] = Counter()
    for diagnostic in report.get("generalDiagnostics", []):
        if diagnostic.get("severity") != "error":
            continue
        path = Path(diagnostic["file"])
        try:
            relative = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            relative = path
        counts[relative.as_posix()] += 1
    return counts


def _render_baseline(counts: Counter[str]) -> str:
    lines = [
        "# Files with recorded Pyright errors. Values are maximum error counts,",
        "# not targets. Lower a value whenever a file improves, delete the entry",
        "# when it reaches zero, and do not add new entries.",
        "[files]",
    ]
    lines += [f'"{path}" = {count}' for path, count in sorted(counts.items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/ronin"],
        help="Paths to type check (default: src/ronin)",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Record the current errors as the baseline instead of enforcing it",
    )
    args = parser.parse_args(argv)

    counts = _errors_per_file(_run_pyright(args.paths))

    if args.write_baseline:
        BASELINE_PATH.write_text(_render_baseline(counts), encoding="utf-8")
        print(f"Recorded {sum(counts.values())} type errors across {len(counts)} files.")
        return 0

    baseline = _load_baseline()
    violations = [
        Violation(path, found, baseline.get(path, 0))
        for path, found in sorted(counts.items())
        if found > baseline.get(path, 0)
    ]
    if violations:
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        return 1

    improved = {
        path: allowed
        for path, allowed in baseline.items()
        if counts.get(path, 0) < allowed
    }
    for path, allowed in sorted(improved.items()):
        print(f"{path} is now at {counts.get(path, 0)} errors; lower its baseline of {allowed}.")

    print(f"Type gate passed with {sum(counts.values())} baselined errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
