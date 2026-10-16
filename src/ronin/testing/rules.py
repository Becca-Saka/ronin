"""The identifiers a verdict comes back under.

Three places depend on this format, and on all three using the same format. The
planner reads the rules. The auditor reads the rules. The gate matches a verdict
to its rule by id. A separate copy in each file would let one drift. Drift fails silently: the
verdicts stop matching, and every scenario reports missing evidence while its
assertions visibly pass. So the scheme lives here, and nowhere else.
"""

from typing import Iterable, List, Set

EXPECTED = "expected"
SECURITY = "security"
NONE_SPECIFIED = "None specified."


def rule_id(kind: str, index: int) -> str:
    """The id for one rule, e.g. `security:1`."""
    return f"{kind}:{index}"


def ids(rules: Iterable, kind: str) -> Set[str]:
    """Every id of this kind. The gate matches verdicts against these."""
    return {rule_id(kind, index) for index, _ in enumerate(rules or [])}


def is_kind(identifier: str, kind: str) -> bool:
    """Whether an id belongs to this kind of rule."""
    return (identifier or "").startswith(f"{kind}:")


def statement_of(rule) -> str:
    """The words of a rule, whether it arrived as a string or an outcome."""
    return rule if isinstance(rule, str) else rule.statement


def labelled(rules: Iterable, kind: str) -> List[str]:
    """The rules as prompt lines. Each line carries the id its verdict must use."""
    return [
        f"- [{rule_id(kind, index)}] {statement_of(rule)}"
        for index, rule in enumerate(rules or [])
        if rule is not None
    ]


def listing(rules: Iterable, kind: str) -> str:
    """One prompt section: the rules of this kind, or a note that none exist."""
    return "\n".join(labelled(rules, kind)) or NONE_SPECIFIED


def unprovable(outcomes: Iterable) -> List[str]:
    """Ids of outcomes nothing Ronin records can settle.

    Named rather than counted as missing. An outcome about how long the app
    took has no evidence anywhere in a run, and reporting it as unevidenced
    reads as a scenario that fell short rather than one that asked for
    something no part of this system measures.
    """
    from ronin.testing.models import AssertionScope, ExpectedOutcome

    return [
        rule_id(EXPECTED, index)
        for index, outcome in enumerate(outcomes or [])
        if isinstance(outcome, ExpectedOutcome)
        and outcome.scope is AssertionScope.RUN_TIMING
    ]


def on_screen_only(outcomes: Iterable) -> List:
    """The outcomes one screenshot could settle, keeping every index.

    An outcome the auditor cannot answer is blanked rather than dropped,
    because the id a verdict carries is its position in this list. Removing one
    would renumber the rest and move every later verdict onto the wrong rule.
    """
    from ronin.testing.models import AssertionScope, ExpectedOutcome

    kept = []
    for outcome in outcomes or []:
        on_screen = (
            outcome.scope is AssertionScope.CURRENT_SCREEN
            if isinstance(outcome, ExpectedOutcome)
            else True
        )
        kept.append(outcome if on_screen else None)
    return kept


def every_rule(expected_outcomes: Iterable, security_rules: Iterable) -> str:
    """Both kinds in one list, for a prompt that keeps them together."""
    lines = labelled(expected_outcomes, EXPECTED) + labelled(security_rules, SECURITY)
    return "\n".join(lines) or NONE_SPECIFIED
