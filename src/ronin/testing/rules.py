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


def ids(rules: Iterable[str], kind: str) -> Set[str]:
    """Every id of this kind. The gate matches verdicts against these."""
    return {rule_id(kind, index) for index, _ in enumerate(rules or [])}


def is_kind(identifier: str, kind: str) -> bool:
    """Whether an id belongs to this kind of rule."""
    return (identifier or "").startswith(f"{kind}:")


def labelled(rules: Iterable[str], kind: str) -> List[str]:
    """The rules as prompt lines. Each line carries the id its verdict must use."""
    return [f"- [{rule_id(kind, index)}] {rule}" for index, rule in enumerate(rules or [])]


def listing(rules: Iterable[str], kind: str) -> str:
    """One prompt section: the rules of this kind, or a note that none exist."""
    return "\n".join(labelled(rules, kind)) or NONE_SPECIFIED


def every_rule(expected_outcomes: Iterable[str], security_rules: Iterable[str]) -> str:
    """Both kinds in one list, for a prompt that keeps them together."""
    lines = labelled(expected_outcomes, EXPECTED) + labelled(security_rules, SECURITY)
    return "\n".join(lines) or NONE_SPECIFIED
