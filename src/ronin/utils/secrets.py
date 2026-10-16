"""Keeping declared secrets out of what Ronin writes down.

The planner writes free text — "type 'password123' into the Password field" —
which lands in logs, trails and reports, and those outlive the run: they are
committed, uploaded as CI artifacts, attached to reviews. A journey says which
of its values are sensitive (see journey_env) and those values are removed
from anything persisted.

Only from what is persisted. What the planner reads is never masked: it has to
recognise the fields it types into, and a security rule like "the card number
must never be shown in plain text" can only be audited by an agent allowed to
see a card number.
"""

from typing import Optional

MIN_LENGTH = 3


class SecretScrubber:
    """Replaces a journey's declared secret values with the name that declared them.

    `{{password}}` rather than `[REDACTED]`: a reader of the report can still
    follow what the test did and which value went where, without the value
    itself ever being written down.
    """

    def __init__(self):
        self._named: dict[str, str] = {}

    def learn_secrets(self, secrets: Optional[dict]) -> None:
        """Takes the name-to-value pairs a journey declared sensitive."""
        for name, value in (secrets or {}).items():
            if value and len(value) >= MIN_LENGTH:
                self._named[value] = name

    def _masks(self) -> list[tuple[str, str]]:
        pairs = [(value, "{{" + name + "}}") for value, name in self._named.items()]
        return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)

    def scrub(self, text: Optional[str]) -> str:
        """Masks every declared secret value found in persisted text."""
        if not text:
            return text or ""
        for value, mask in self._masks():
            text = text.replace(value, mask)
        return text

    def contains_secret(self, text: Optional[str]) -> bool:
        """Whether any declared secret appears in the text, for image masking."""
        if not text:
            return False
        return any(value in text for value in self._named)
