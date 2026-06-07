"""
Per-step action history with outcome observations.
Each entry carries what was planned, what actually happened, and the LLM's own assessment.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ActionResult:
    """System-observed outcome of a single executed action."""

    success: bool
    observation: str
    error: Optional[str] = None


@dataclass
class ActionEntry:
    """One step in the action history: what was planned, what happened, and what the LLM thought."""

    step_num: int
    action_desc: str
    result: ActionResult
    evaluation: str = ""  # filled in at the next step via update_last_evaluation()


class AgentHistory:
    """Rolling action history with outcomes, formatted for LLM consumption."""

    def __init__(self):
        self._entries: List[ActionEntry] = []

    def append(self, entry: ActionEntry) -> None:
        self._entries.append(entry)

    def update_last_evaluation(self, evaluation: str) -> None:
        """Attach the LLM's self-assessment of the previous step to that entry."""
        if self._entries and evaluation:
            self._entries[-1].evaluation = evaluation

    def format_for_prompt(self) -> str:
        if not self._entries:
            return "No actions executed yet."
        lines = []
        for e in self._entries:
            line = f"[Step {e.step_num}] {e.action_desc} → {e.result.observation}"
            if e.evaluation:
                line += f" | Self-assessment: {e.evaluation}"
            lines.append(line)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
