#!/usr/bin/env python3
"""Break the judging and the gate on purpose, and check something notices.

    python -m tools.mutation_sweep           every mutant
    python -m tools.mutation_sweep --only crash

No mutation tool is configured for this repository, and a general one would
spend most of its time on code where a surviving mutant means nothing. This
holds a written list instead: each entry is a defect somebody could really
introduce, in the logic that decides a verdict or decides whether the gate
passes.

A mutant is **caught** when the check that owns it fails. The judge's mutants
are owned by the eval corpus, so the gate has to fail. The gate's own mutants
are owned by its tests, so pytest has to fail.

The edit is applied to the file, the check is run, and the file is put back in
a `finally`. A sweep that is interrupted still restores what it changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = Path(__file__).resolve().with_name("mutants.json")

JUDGE = "judge"
"""Owned by the eval corpus: the gate has to fail."""

GATE = "gate"
"""Owned by the gate's own tests: pytest has to fail."""

GATE_TESTS = (
    "tests/tools/test_eval_gate.py",
    "tests/tools/test_eval_tiers.py",
    "tests/tools/test_eval_corpus.py",
    "tests/judge/test_auditor_answers.py",
)


@dataclass(frozen=True)
class Mutant:
    """One defect, and where it goes."""

    name: str
    owner: str
    path: str
    find: str
    replace: str


def _python() -> str:
    """The interpreter running this, so a sweep uses the same environment."""
    return sys.executable


def _checked(mutant: Mutant) -> subprocess.CompletedProcess:
    if mutant.owner == JUDGE:
        return subprocess.run(
            [_python(), "-m", "tools.eval_gate"],
            cwd=ROOT, capture_output=True, text=True,
        )
    return subprocess.run(
        [_python(), "-m", "pytest", *GATE_TESTS, "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )


def _applied(mutant: Mutant) -> tuple[bool, str]:
    """Runs one mutant, and says whether the check caught it."""
    path = ROOT / mutant.path
    source = path.read_text(encoding="utf-8")
    if mutant.find not in source:
        return False, "the code it edits has moved. Rewrite the mutant."
    try:
        path.write_text(source.replace(mutant.find, mutant.replace, 1), encoding="utf-8")
        done = _checked(mutant)
    finally:
        path.write_text(source, encoding="utf-8")
    if done.returncode != 0:
        return True, _told(done.stdout)
    return False, "nothing failed"


def _told(output: str) -> str:
    """The line of a failure worth putting in a sweep's own report."""
    for line in output.splitlines():
        if "wrong:" in line or "unanswered:" in line or " failed" in line:
            return line.strip()
    return "the check failed"


def _read() -> list[Mutant]:
    loaded = json.loads(MUTANTS.read_text(encoding="utf-8"))
    return [Mutant(**entry) for entry in loaded]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="Run the mutants whose name holds this.")
    args = parser.parse_args(argv)

    mutants = [m for m in _read() if not args.only or args.only in m.name]
    caught = 0
    stale = 0
    for mutant in mutants:
        was_caught, detail = _applied(mutant)
        if was_caught:
            caught += 1
            print(f"caught    [{mutant.owner}] {mutant.name}  ->  {detail}")
        elif "has moved" in detail:
            stale += 1
            print(f"STALE     [{mutant.owner}] {mutant.name}  ->  {detail}")
        else:
            print(f"SURVIVED  [{mutant.owner}] {mutant.name}  ->  {detail}")

    print(f"\n{caught} of {len(mutants)} mutants caught.")
    if stale:
        print(f"{stale} mutant(s) no longer match the code. A stale mutant tests nothing.")
    return 0 if caught == len(mutants) else 1


if __name__ == "__main__":
    raise SystemExit(main())
