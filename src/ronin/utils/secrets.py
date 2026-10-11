"""Keeping test data out of logs without guessing what looks secret.

The planner writes free text — "type 'password123' into the Password field" —
which lands in logs, trails and reports. Rather than pattern-matching for
secrets, the values a scenario actually uses are collected from its own
instructions and from what the agent types, then removed wherever they appear.
"""

import re
from typing import Iterable, Optional

QUOTED = re.compile(r"'([^']{3,})'|\"([^\"]{3,})\"")
MASK = "[REDACTED]"
MIN_LENGTH = 3


class SecretScrubber:
    """Removes a run's known input values from any text about to be logged."""

    def __init__(self):
        self._values: set[str] = set()

    def learn_from_instructions(self, instructions: Optional[Iterable[str] | str]) -> None:
        """Collects quoted literals, which is how scenarios carry their test data."""
        if not instructions:
            return
        text = instructions if isinstance(instructions, str) else "\n".join(instructions)
        for match in QUOTED.finditer(text):
            value = match.group(1) or match.group(2)
            if value and len(value) >= MIN_LENGTH:
                self._values.add(value)

    def learn(self, value: Optional[str]) -> None:
        if value and len(value) >= MIN_LENGTH:
            self._values.add(value)

    def scrub(self, text: Optional[str]) -> str:
        if not text:
            return text or ""
        for value in sorted(self._values, key=len, reverse=True):
            text = text.replace(value, MASK)
        return text
