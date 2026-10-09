"""Rendering for `ronin doctor`."""

import json
from dataclasses import asdict
from typing import List

from ronin.cli.console import BOLD, Console
from ronin.doctor import CheckSection, Status, run_checks


def render(sections: List[CheckSection], console: Console) -> None:
    """Prints the sectioned report, with fix hints under anything not OK."""
    for section in sections:
        console.heading(section.title)
        for result in section.results:
            console.result(
                result.status.value,
                result.name,
                result.detail,
                result.hint if result.status in (Status.WARN, Status.FAIL) else None,
            )


def summarize(sections: List[CheckSection], console: Console) -> int:
    """Prints the closing summary and returns the process exit code."""
    failures = [r for s in sections for r in s.failures]
    warnings = [r for s in sections for r in s.warnings]

    console.write()
    if not failures and not warnings:
        console.count(Status.OK.value, "No issues found. Ronin is ready.")
        return 0

    if warnings:
        console.count(Status.WARN.value, f"{len(warnings)} warning(s):")
        for result in warnings:
            console.bullet(Status.WARN.value, f"{result.name}: {result.detail}")
    if failures:
        console.count(Status.FAIL.value, f"{len(failures)} blocking issue(s):")
        for result in failures:
            console.bullet(Status.FAIL.value, f"{result.name}: {result.detail}")
        console.write()
        console.write(f"Fix the above, then re-run {console.paint('ronin doctor', BOLD)}.")
        return 1

    console.write()
    console.write("Warnings are safe to ignore if the affected platform is not your target.")
    return 0


def as_json_payload(sections: List[CheckSection]) -> dict:
    """Serializes the report for CI consumers."""
    return {
        "sections": [
            {
                "title": section.title,
                "results": [
                    {**asdict(result), "status": result.status.value}
                    for result in section.results
                ],
            }
            for section in sections
        ],
        "blocking": sum(len(section.failures) for section in sections),
    }


def run(platforms: tuple[str, ...], as_json: bool) -> int:
    """Executes the checks and emits either the human report or JSON."""
    sections = run_checks(platforms)

    if as_json:
        payload = as_json_payload(sections)
        print(json.dumps(payload, indent=2))
        return 1 if payload["blocking"] else 0

    console = Console()
    console.title("Ronin doctor")
    render(sections, console)
    return summarize(sections, console)
