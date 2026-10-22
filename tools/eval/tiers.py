"""Running the corpus, and saying what each item answered.

Two tiers, because two different things can regress and they are not
measurable in the same way.

**Tier 1** puts stored evidence through the deterministic judge. No model, no
device. The same evidence gives the same verdict every time, so one pass is
enough and an item is either correct or wrong.

**Tier 2** puts a screenshot and a rule to the auditor. A model answers, so
the same question can answer differently twice. Each rule is asked several
times and classed by what came back: correct every time, wrong every time, or
disagreeing with itself.

Nothing here reports a rate. An item is named, its state is named, and the
baseline decides what that is allowed to be. At this corpus size a percentage
would move by whole points on one item, which reads as a result and is noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from ronin.testing.judge import decide, rules

from tools.eval.corpus import ScreenFixture, VerdictFixture

logger = logging.getLogger(__name__)

CORRECT = "correct"
WRONG = "wrong"
FLAKY = "flaky"
UNANSWERED = "unanswered"
"""The rule was put to the auditor and no verdict naming it came back."""

REPLICATES = 3
"""How many times one rule is asked. Three separates a flake from an answer."""


@dataclass(frozen=True)
class Audit:
    """One call to the auditor about one screenshot.

    Kept per replicate rather than per item. An item's state is an aggregate,
    and an aggregate cannot say which call produced it. Without the route and
    the prompt behind each call, a result that moved between two sweeps
    cannot be attributed to the prompt, the model or the provider.
    """

    fixture: str
    digest: str
    replicate: int
    answered: dict
    """Each wanted rule id, and the status this call gave it. In memory only.

    Derived from the assessment, so it is not written down. What a reader
    needs is the answer itself, and a rule that got no verdict is a rule the
    answer does not mention.
    """

    assessment: Optional[dict] = None
    requested_model: Optional[str] = None
    served_model: Optional[str] = None
    """What answered, which can differ from what was asked for. A provider
    that quietly serves something else is one reason a result moves."""

    prompt_sha256: Optional[str] = None
    schema_sha256: Optional[str] = None
    cost: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "fixture": self.fixture,
            "digest": self.digest,
            "replicate": self.replicate,
            "assessment": self.assessment,
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "prompt_sha256": self.prompt_sha256,
            "schema_sha256": self.schema_sha256,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class Answer:
    """What one corpus item answered, and how that reads."""

    item: str
    state: str
    detail: str
    digest: str = ""
    """The fixture this answered, as a hash. A baseline entry is bound to it."""

    @property
    def is_correct(self) -> bool:
        return self.state == CORRECT


def judge_verdicts(fixtures: list[VerdictFixture]) -> list[Answer]:
    """Tier 1. Every verdict fixture, through the judge that has no model."""
    answers = []
    for fixture in fixtures:
        answers.append(_one_verdict(fixture))
    return answers


def _one_verdict(fixture: VerdictFixture) -> Answer:
    try:
        verdict = decide.determine_status(**fixture.arguments)
    except Exception as exc:  # noqa: BLE001 - a fixture must not stop the sweep
        return Answer(
            fixture.id,
            WRONG,
            f"the judge raised {type(exc).__name__}: {exc}",
            fixture.digest,
        )
    disagreement = fixture.expect.disagreement(*verdict)
    if disagreement is None:
        return Answer(fixture.id, CORRECT, verdict.status, fixture.digest)
    return Answer(fixture.id, WRONG, disagreement, fixture.digest)


def audit_screens(
    fixtures: list[ScreenFixture],
    auditor_for: Callable[[ScreenFixture], object],
    replicates: int = REPLICATES,
    audits: Optional[list] = None,
    call_for: Optional[Callable[[], object]] = None,
) -> list[Answer]:
    """Tier 2. Every rule of every screen fixture, asked `replicates` times.

    Args:
        fixtures: The screens to put to the auditor.
        auditor_for: Builds one auditor for one fixture. Called once per
            replicate, so no answer is served from a previous one's cache.
        replicates: How many times each screen is asked.
        audits: Filled with one `Audit` per call, where a caller wants the
            provenance of each answer rather than the states alone.
        call_for: Hands back the call the auditor just made. The auditor
            returns an assessment and not the call behind it, so a caller
            that wants the route supplies this and keeps the two joined.
    """
    answers = []
    for fixture in fixtures:
        answers.extend(_one_screen(fixture, auditor_for, replicates, audits, call_for))
    return answers


def _one_screen(
    fixture: ScreenFixture,
    auditor_for: Callable[[ScreenFixture], object],
    replicates: int,
    audits: Optional[list] = None,
    call_for: Optional[Callable[[], object]] = None,
) -> list[Answer]:
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, not {replicates}")
    screenshot = fixture.screenshot.read_bytes()
    wanted = _every_id(fixture)
    seen: dict[str, list[Optional[str]]] = {}
    for replicate in range(replicates):
        given, answer = _asked_once(fixture, auditor_for, screenshot)
        if audits is not None:
            record = call_for() if call_for else None
            audits.append(_audit(fixture, replicate, given, wanted, record, answer))
        for identifier in wanted:
            seen.setdefault(identifier, []).append(given.get(identifier))
    return [
        _classed(f"{fixture.id}/{identifier}", answer, seen[identifier], fixture.digest)
        for identifier, answer in wanted.items()
    ]


def _asked_once(
    fixture: ScreenFixture, auditor_for: Callable[[ScreenFixture], object], screenshot: bytes
) -> tuple[dict[str, Optional[str]], object]:
    """One look at the screen: a status per rule id, and the whole answer.

    `screen_id` is None so the auditor's own cache is never consulted. A
    replicate served from a cache is the same answer counted twice, which
    would read as stability the model did not show.
    """
    auditor = auditor_for(fixture)
    assessment = auditor.audit(screenshot, None)  # type: ignore[attr-defined]
    answered = {}
    for verification in assessment.rule_verifications:
        if verification.rule_id:
            answered[verification.rule_id] = verification.status
    return answered, assessment


def _audit(
    fixture: ScreenFixture, replicate: int, given: dict, wanted: dict, record, answer=None
) -> Audit:
    """One call written down, with what the client reported about it.

    The route is read off the call the client recorded. A sweep that named
    only the model it asked for would be silent about a provider serving
    something else, which is one of the things a moved result can mean.
    """
    answered = {identifier: given.get(identifier) for identifier in wanted}
    whole = _whole(answer)
    if record is None:
        return Audit(
            fixture=fixture.id, digest=fixture.digest, replicate=replicate,
            answered=answered, assessment=whole,
        )
    return Audit(
        fixture=fixture.id,
        digest=fixture.digest,
        replicate=replicate,
        answered=answered,
        assessment=whole,
        requested_model=record.model,
        served_model=record.served_model,
        prompt_sha256=record.prompt_sha256,
        schema_sha256=record.schema_sha256,
        cost=record.cost,
    )


def _whole(answer) -> Optional[dict]:
    """One assessment as plain data, where the caller handed one over."""
    if answer is None:
        return None
    return answer.model_dump(mode="json")


def _every_id(fixture: ScreenFixture) -> dict[str, str]:
    """Each rule's id, mapped to the answer it must give.

    The index is the rule's position among the rules of its own kind, which
    is how `rules.labelled` numbers them for the prompt. Counted here rather
    than matched on the text, because two rules in one fixture may read alike
    and a lookup by text would answer for the wrong one.
    """
    wanted = {}
    counted = {rules.EXPECTED: 0, rules.SECURITY: 0}
    for rule in fixture.rules:
        kind = rules.EXPECTED if rule.kind == "expected" else rules.SECURITY
        wanted[rules.rule_id(kind, counted[kind])] = rule.expect
        counted[kind] += 1
    return wanted


def _classed(
    item: str, wanted: str, given: list[Optional[str]], digest: str = ""
) -> Answer:
    """One rule's state across its replicates.

    Silence is read before anything else. A look that named no verdict for
    this rule produced no evidence about it, which is not the same as a
    verdict that disagrees with another one. Read as disagreement it would be
    tolerated by a `known_flaky` line, and a rule the auditor stopped
    answering at all would be covered by it.
    """
    if any(answer is None for answer in given):
        return Answer(item, UNANSWERED, f"answered {_counted(given)}", digest)
    matched = [answer == wanted for answer in given]
    if all(matched):
        return Answer(item, CORRECT, wanted, digest)
    if not any(matched):
        return Answer(item, WRONG, f"answered {_counted(given)}, wanted {wanted}", digest)
    return Answer(item, FLAKY, f"answered {_counted(given)}, wanted {wanted}", digest)


def _counted(given: list[Optional[str]]) -> str:
    """The answers that came back, each with how often."""
    tally: dict[str, int] = {}
    for answer in given:
        name = answer or "nothing"
        tally[name] = tally.get(name, 0) + 1
    return ", ".join(f"{name} x{count}" for name, count in sorted(tally.items()))
