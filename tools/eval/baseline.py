"""The baseline, and what a sweep of the corpus is allowed to answer.

The baseline names the items that answer wrongly today. Each entry carries the
reason it is tolerated, and the version of the fixture it was granted to. A
sweep fails when an item answers wrongly and no entry accounts for it. It also
fails when an entry names an item the corpus no longer holds.

This is a ratchet, in the sense the other gates here use: the recorded number
of accepted failures may fall and may not rise.

Why not a rate. The corpus is small, and it can only grow as fast as a person
labels one. At this size a false-pass percentage moves by whole points on a
single item, so a change in it says nothing about whether the auditor got
better. What an item did is a fact. What the population does is not knowable
from here, and reporting it as if it were would be the error this gate exists
to prevent.

The baseline is read and never written by the gate. A new tolerated item is a
decision, and a decision belongs in a commit a person made.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.eval.corpus import SCREEN_TIER, VERDICT_TIER
from tools.eval.tiers import CORRECT, FLAKY, WRONG, Answer

BASELINE_PATH = Path(__file__).resolve().parents[1] / "eval-baseline.toml"

KNOWN = "known_wrong"
ALLOWED_FLAKY = "known_flaky"


@dataclass(frozen=True)
class Exemption:
    """One tolerated item: why, and which version of the fixture.

    Bound to the fixture's content, not to its name. An exemption granted to
    a case has to lapse when the case changes, or an item can be edited into
    something else under a name that is already excused.
    """

    reason: str
    fixture: str

    def covers(self, answer: Answer) -> bool:
        return self.fixture == answer.digest


@dataclass(frozen=True)
class Baseline:
    """The items whose wrong or unstable answers are already accounted for."""

    known_wrong: dict[str, Exemption]
    known_flaky: dict[str, Exemption]

    @classmethod
    def read(cls, path: Path = BASELINE_PATH) -> "Baseline":
        try:
            with path.open("rb") as handle:
                loaded = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise BaselineError(f"{path.name}: {exc}") from exc
        return cls(
            known_wrong=_exemptions(loaded.get(KNOWN, {}), path),
            known_flaky=_exemptions(loaded.get(ALLOWED_FLAKY, {}), path),
        )

    def _entry(self, answer: Answer) -> Optional[Exemption]:
        """The entry granted for the state this answer is in, if any.

        An exemption covers the state it was written for and no other.
        `known_wrong` says a rule is answered wrongly. It does not say the
        auditor may stop answering it, and silence read as a wrong answer
        would be excused by it.
        """
        if answer.state == WRONG:
            return self.known_wrong.get(answer.item)
        if answer.state == FLAKY:
            return self.known_flaky.get(answer.item)
        return None

    def tolerates(self, answer: Answer) -> bool:
        """Whether this answer is one the baseline already accounts for."""
        entry = self._entry(answer)
        return entry is not None and entry.covers(answer)

    def granted(self, item: str) -> Optional[Exemption]:
        """Any entry naming this item, whatever state it was granted for."""
        return self.known_wrong.get(item) or self.known_flaky.get(item)

    def tolerated_version(self, answer: Answer) -> bool:
        """Whether an entry names this item and matches the fixture it answered."""
        entry = self.granted(answer.item)
        return entry is not None and entry.covers(answer)

    def lapsed(self, answer: Answer) -> bool:
        """Whether an entry names this item and no longer matches its fixture."""
        entry = self.granted(answer.item)
        return entry is not None and not entry.covers(answer)

    @property
    def items(self) -> set:
        return set(self.known_wrong) | set(self.known_flaky)


class BaselineError(Exception):
    """A baseline file an exemption cannot be read out of."""


def _tier_of(item: str) -> str:
    """Which corpus an item belongs to, read from its name.

    A screen rule is reported as `fixture/expected:0`, and a verdict fixture
    may not hold a slash in its id. So the two namespaces do not overlap, and
    a sweep can tell which of its entries it was in a position to answer.
    """
    return SCREEN_TIER if "/" in item else VERDICT_TIER


def _exemptions(loaded: dict, path: Path) -> dict[str, Exemption]:
    """One table per item, each naming a reason and the fixture it covers."""
    entries = {}
    for item, given in loaded.items():
        if not isinstance(given, dict) or "reason" not in given or "fixture" not in given:
            raise BaselineError(
                f"{path.name}: '{item}' needs a reason and the fixture hash it "
                "covers. Run the gate to read the hash off the failure."
            )
        entries[item] = Exemption(reason=given["reason"], fixture=given["fixture"])
    return entries


@dataclass(frozen=True)
class Reading:
    """What one sweep of the corpus found, against the baseline.

    `tiers` names the tiers this sweep actually ran. A baseline entry for a
    tier that did not run is left alone: the free run reads the verdict
    corpus only, and every screen entry would otherwise read as a fixture
    somebody deleted.
    """

    answers: list[Answer]
    baseline: Baseline
    tiers: frozenset = frozenset()

    @property
    def regressions(self) -> list[Answer]:
        """Items answering wrongly that the baseline does not account for."""
        return [
            answer
            for answer in self.answers
            if not answer.is_correct and not self.baseline.tolerates(answer)
        ]

    @property
    def repaired(self) -> list[str]:
        """Baselined items whose own fixture now answers correctly.

        Reported rather than failed. A person removes the line, in the commit
        that earned it. An item whose fixture changed is not repaired: it is
        a different case under an old name, and `lapsed` is where it reads.
        """
        return sorted(
            answer.item
            for answer in self.answers
            if answer.is_correct and self.baseline.tolerated_version(answer)
        )

    @property
    def absent(self) -> list[str]:
        """Baselined items this sweep could have answered and did not hold.

        Failed rather than reported. Deleting a fixture is how a failing case
        stops failing without being fixed, and a line naming nothing would
        let that pass.
        """
        present = {answer.item for answer in self.answers}
        return sorted(self._mine - present)

    @property
    def _mine(self) -> set:
        """The baselined items belonging to the tiers this sweep ran."""
        return {item for item in self.baseline.items if _tier_of(item) in self.tiers}

    @property
    def lapsed(self) -> list[Answer]:
        """Items whose exemption named a different version of the fixture."""
        return [answer for answer in self.answers if self.baseline.lapsed(answer)]

    @property
    def holds(self) -> bool:
        """Whether this sweep is one the gate passes.

        A lapsed exemption fails even where the item now answers correctly.
        Otherwise editing a label to match the implementation would clear the
        item and leave the stale entry standing, which is the weakening this
        binding exists to catch.
        """
        return not self.regressions and not self.absent and not self.lapsed

    def render(self) -> str:
        """The sweep as lines a person reads in a build log."""
        lines = [
            f"{len([a for a in self.answers if a.state == CORRECT])}"
            f" of {len(self.answers)} corpus items answered correctly."
        ]
        for answer in self.answers:
            if answer.is_correct:
                continue
            tolerated = " (baselined)" if self.baseline.tolerates(answer) else ""
            lines.append(f"  {answer.state}{tolerated}: {answer.item} — {answer.detail}")
        for answer in self.lapsed:
            lines.append(
                f"  the exemption for {answer.item} covers an older fixture. "
                f"It now hashes to {answer.digest}."
            )
        for item in self.repaired:
            lines.append(f"  now correct, so take it out of the baseline: {item}")
        for item in self.absent:
            lines.append(f"  baselined and no longer in the corpus: {item}")
        return "\n".join(lines)
