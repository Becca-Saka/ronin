"""A live record of what a suite is doing, written as it goes.

A long suite is otherwise opaque until its report appears at the end. This
file lists every scenario up front and updates each one as it finishes, so it
can be read — or tailed — while the run is still going.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List

from ronin.testing.models import GeneratedTestCase, ScenarioReport, UserJourney
from ronin.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)


class ScenarioProgress:
    """Writes the scenario plan and keeps its statuses current."""

    def __init__(
        self,
        output_dir: Path,
        journey: UserJourney,
        scenarios: List[GeneratedTestCase],
        scrubber: SecretScrubber | None = None,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", journey.id).strip("._") or "journey"
        self.path = output_dir / f"scenarios_{safe_id}_{timestamp}.json"
        self.journey = journey
        self._scrubber = scrubber
        self.started_at = datetime.now()
        self.entries = {
            scenario.test_id: {
                "position": index + 1,
                "test_id": scenario.test_id,
                "name": self._scrub(scenario.scenario_name),
                "cell": scenario.cell,
                "category": scenario.category.value,
                "scenario_type": scenario.scenario_type.value,
                "priority": scenario.priority.value,
                "status": "pending",
                "steps_planned": len(scenario.steps_instructions),
                "steps_instructions": [self._scrub(value) for value in scenario.steps_instructions],
                "expected_outcomes": [
                    self._scrub(outcome.statement)
                    for outcome in scenario.expected_outcomes
                ],
            }
            for index, scenario in enumerate(scenarios)
        }

    def start(self, test_id: str) -> None:
        entry = self.entries.get(test_id)
        if entry is not None:
            entry["status"] = "running"
            entry["started_at"] = datetime.now().isoformat(timespec="seconds")
        self.write()

    def _scrub(self, value: str | None) -> str:
        return self._scrubber.scrub(value) if self._scrubber else value or ""

    def record(self, test_id: str, report: ScenarioReport) -> None:
        entry = self.entries.get(test_id)
        if entry is not None:
            entry["status"] = report.status
            entry["steps_executed"] = report.steps_executed
            entry["duration_seconds"] = round(report.duration, 1)
            entry["failure_reason"] = self._scrub(report.failure_reason)
            entry["issues_found"] = len(report.detected_issues or [])
            entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.write()

    def abort(self, reason: str) -> None:
        """Marks everything unfinished as aborted, so a crash leaves no 'running' row."""
        for entry in self.entries.values():
            if entry["status"] in ("pending", "running"):
                entry["status"] = "ABORTED"
                entry["failure_reason"] = self._scrub(reason)
                entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.write()

    def write(self) -> None:
        entries = list(self.entries.values())
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1

        payload = {
            "journey": {
                "id": self.journey.id,
                "name": self._scrub(self.journey.name),
            },
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "total": len(entries),
            "counts": counts,
            "scenarios": entries,
        }
        try:
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as exc:
            logger.warning("Could not update the scenario plan at %s: %s", self.path, exc)
