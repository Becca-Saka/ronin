"""Turning a run and its evidence into one word about the scenario.

Nothing here touches a device or a model. It reads what a run recorded and
decides what that adds up to, so the ranking between a crash, an infeasible
stop and missing evidence lives in one place.
"""

import logging
from typing import List, Optional

from ronin.agent.step.outcomes import DeviceRunResult
from ronin.testing import rules
from ronin.testing.models import QAIssue, RuleVerification
from ronin.testing.qa_planner import QAStepResponse

logger = logging.getLogger(__name__)


def _terminal_outcome(
    run_result: DeviceRunResult,
    stop_reason: Optional[str],
    verifications: List[RuleVerification],
    issues: List[QAIssue],
    outputs: List[QAStepResponse],
    findings: Optional[List] = None,
):
    """Classify outcomes that cannot become a successful verification."""

    # Ranked first: a crash is something the app did, and the rest are
    # statements about the run. A scenario that crashed the app has found a
    # defect, whatever else the planner concluded about being able to proceed.
    if any(issue.issue_type == "crash_screen" for issue in issues):
        return "FAILED", "App crashed: crash screen detected."
    if run_result.status == "hook_stop" and stop_reason and "infeasible" in stop_reason.lower():
        return "SKIPPED", stop_reason
    if run_result.status == "exception":
        return "ERROR", run_result.failure_reason or "Execution exception"
    if run_result.status in ("loop_detected", "step_limit"):
        return "UNVERIFIED", run_result.failure_reason
    critical = next(
        (
            issue
            for issue in issues
            if issue.severity == "critical"
            and not _unsupported_exposure(issue, findings)
        ),
        None,
    )
    if critical:
        return "FAILED", f"Critical QA issue: {critical.description}"
    violated = _violated_security_rule(verifications)
    if violated:
        return "FAILED", f"Security rule violated: {violated.rule}"
    failed = next(
        (
            verification
            for verification in _last_applicable(verifications).values()
            if verification.status == "failed"
        ),
        None,
    )
    if failed:
        return "FAILED", f"Rule validation failed: {failed.rule}"
    if outputs and not outputs[-1].is_scenario_feasible:
        return "SKIPPED", outputs[-1].infeasibility_reason or "Scenario infeasible"
    return None


EXPOSURE = "sensitive_data_exposure"


def _unsupported_exposure(issue, findings) -> bool:
    """Whether an exposure claim has no reading of the screen behind it.

    The auditor answers twice about a screen: in prose, where it may call a
    bare "CVV:" caption a plain-text CVV, and as a reading of each sensitive
    field, where it must choose between a legible value, a mask, a caption and
    nothing. The reading is the evidence. A claim the reading does not support
    is reported and does not fail the scenario.

    With no reading at all there is nothing to weigh, so the claim stands.
    """
    if issue.issue_type != EXPOSURE:
        return False
    here = [finding for finding in (findings or []) if finding.screen_name == issue.current_screen]
    return bool(here) and not any(finding.is_exposure() for finding in here)


def _last_applicable(verifications: List[RuleVerification]) -> dict:
    """The last answer from a screen the rule could apply to, per rule.

    Every rule is evaluated on every screen, so a scenario accumulates many
    answers per rule. Screens the rule cannot apply to answer "untested", and
    those say nothing about the rule. Reading them as the verdict would let
    walking away from a field decide whether the field worked.
    """
    latest = {}
    for verification in verifications:
        if verification.rule_id and verification.applicable:
            latest[verification.rule_id] = verification
    return latest


def _violated_security_rule(verifications: List[RuleVerification]):
    """A security rule failing anywhere is a violation, however the run ended.

    Unlike an expected outcome, these describe what must never happen: a
    password exposed at step 4 stays exposed, whatever step 9 shows.
    """
    return next(
        (
            verification
            for verification in verifications
            if verification.status == "failed"
            and rules.is_kind(verification.rule_id, rules.SECURITY)
        ),
        None,
    )


def _missing_required_ids(
    verifications: List[RuleVerification],
    required_outcomes: List[str],
    security_rules: List[str],
) -> list[str]:
    """Lists the required assertions a scenario ended without evidencing.

    The two kinds are judged differently because they mean different things.

    An expected outcome is evidenced once some screen showed it holding. Many
    outcomes can only be seen in passing — a field accepts a long address, and
    the next screen has no such field. Letting the newest answer decide would
    let leaving that screen erase the proof that it worked. A later "failed"
    still counts, because the outcome stopped holding where it could be seen.

    A security rule describes what must never happen. The planner reports on
    every rule on every screen, so a mention alone proves nothing. A rule
    counts as required only once some screen could have broken it. Once
    evidenced there, it stays evidenced, whatever later screens show.
    """
    expected_ids = rules.ids(required_outcomes, rules.EXPECTED)
    security_ids = rules.ids(security_rules, rules.SECURITY)
    applicable = {
        verification.rule_id
        for verification in verifications
        if verification.rule_id and verification.applicable
    }
    evidenced = {
        verification.rule_id
        for verification in verifications
        if verification.rule_id
        and verification.applicable
        and verification.status == "passed"
    }
    latest = _last_applicable(verifications)
    settled = {
        rule_id
        for rule_id in expected_ids
        if (verification := latest.get(rule_id))
        and verification.status == "passed"
    }
    required_ids = expected_ids | (security_ids & applicable)
    return sorted(required_ids - settled - (evidenced & security_ids))
