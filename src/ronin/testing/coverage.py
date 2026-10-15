"""Accounting for security rules across a whole journey.

Scoping a rule to the scenarios that can reach it stops a sign-in test being
failed for not evidencing a checkout rule. Taken alone that leaves a hole: a
rule no scenario ever reaches disappears quietly. The suite closes it — every
rule has to be evidenced by at least one scenario, or be named as unevidenced.
"""

from typing import Iterable, List

from ronin.testing.models import ScenarioReport


def unevidenced_security_rules(
    security_rules: List[str], scenarios: Iterable[ScenarioReport]
) -> List[str]:
    """Returns the rules no scenario managed to check where they could apply."""
    evidenced = set()
    for scenario in scenarios:
        for verification in scenario.rule_verifications:
            rule_id = verification.rule_id or ""
            if (
                rule_id.startswith("security:")
                and verification.status == "passed"
                and verification.applicable
            ):
                evidenced.add(rule_id)
    return [
        rule
        for index, rule in enumerate(security_rules)
        if f"security:{index}" not in evidenced
    ]
