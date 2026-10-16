"""Audits one screen against the rules and reports what it shows.

The auditor receives the screenshot and the rules, and nothing else. Not the
planned action, not the history, not what the scenario expects next. A verdict
counts only if the thing that gave it could disagree with the planner. Withhold
the plan and it can.

It reads the clean screenshot, not the numbered one. Marks cover the interface,
and a drawn box hides the defects the auditor looks for: clipped text,
overlapping controls, and overflow stripes.

One assessment covers every rule on one screen. The auditor reads a given
screen once.

It judges rules and reports defects. It does not judge whether the scenario
finished: that needs the goal, and withholding the goal is the point.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from ronin.llm.llm_client import LLMClient
from ronin.llm.messages import MultiPartMessage
from ronin.testing import rules
from ronin.testing.models import QAIssue, RuleVerification
from functools import partial

from ronin.utils.redactor import safe_text

logger = logging.getLogger(__name__)


class Assessment(BaseModel):
    """What one look at one screen says about the rules."""

    screen_name: str = Field(
        ..., description="Descriptive name of the screen, e.g. 'Login Screen'."
    )
    rule_verifications: List[RuleVerification] = Field(
        default_factory=list,
        description="One entry per rule listed, carrying that rule's bracketed id.",
    )
    detected_issues: List[QAIssue] = Field(
        default_factory=list,
        description="Visual defects visible on this screen.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_common_provider_shapes(cls, value):
        """Normalize equivalent shapes produced despite advisory JSON schemas."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        verifications = normalized.get("rule_verifications")
        if isinstance(verifications, dict):
            normalized["rule_verifications"] = [
                {
                    "rule": rule,
                    "status": status,
                    "reasoning": "Provider returned a compact rule-status mapping.",
                }
                for rule, status in verifications.items()
            ]
        for verification in normalized.get("rule_verifications") or []:
            if isinstance(verification, dict) and "reasoning" not in verification:
                verification["reasoning"] = verification.pop(
                    "details", "No reasoning supplied by the provider."
                )
        for issue in normalized.get("detected_issues") or []:
            if not isinstance(issue, dict):
                continue
            if "description" not in issue and "issue" in issue:
                issue["description"] = issue.pop("issue")
            issue.setdefault("issue_type", "qa_issue")
        return normalized



PROMPT = """You are auditing one screenshot of a mobile app.

You are not driving the app and you do not know what it was asked to do. Judge
only what this screen shows.

### Rules to audit
{rules}

### How to answer
1. `screen_name`: what this screen is, from what is on it.
2. `rule_verifications`: one entry per rule above, copying its bracketed id into
   `rule_id`.
   - "passed" means you can see this screen satisfying the rule, now.
   - "failed" means you can see this screen breaking it, now. A rule is not
     failed because a field could show something once it is filled, because a
     label suggests a value might appear beside it, or because the layout
     could expose something under other conditions. Report what is rendered,
     not what the design permits. If the violating content is not on the
     screen in front of you, the rule is not failed.
   - "untested" means this screen does not show you either way. A rule about a
     screen you are not looking at is "untested", never "failed".
   - `applicable` is false when this screen holds nothing the rule could apply
     to — a card-masking rule where there is no card field. Pair it with
     "untested". Never mark a rule passed because its subject is absent, and
     never from what you expect a field to do once it is filled. Judge the
     pixels in front of you.
   - `reasoning`: what you saw, not what you infer.
3. `detected_issues`: visual defects on this screen — clipped or overlapping
   text, layout overflow stripes, a blank or crashed screen, or sensitive data
   shown in plain text. Each needs a `severity`.
"""


class QAAuditor:
    """Asks a model what one screen shows, with no knowledge of the plan."""

    def __init__(self, llm_client: LLMClient, expected_outcomes, security_rules, scrubber=None):
        self._llm = llm_client
        self._rules = rules.every_rule(expected_outcomes, security_rules)
        self._scrubber = scrubber
        self._by_screen: dict[str, Assessment] = {}

    def audit(self, screenshot: bytes, screen_id: Optional[str]) -> Assessment:
        """What this screen shows. Asks once per distinct screen.

        A screen that did not change cannot give a different answer, so an
        identical fingerprint reuses the answer. The match is exact on purpose.
        A looser one would reuse an assessment across the small change the
        auditor exists to catch — a value appearing in a field that was empty.

        Expect few hits. Typing changes the fingerprint, so a form pays per
        step. One recorded run of 122 audits reused nothing.
        """
        if screen_id and screen_id in self._by_screen:
            logger.debug("Screen %s already audited; reusing it.", screen_id[:12])
            return self._kept(self._by_screen[screen_id])

        message = MultiPartMessage()
        message.add_text("The screen to audit:")
        message.add_image(screenshot)
        message.add_text(PROMPT.format(rules=self._rules))

        logger.info("Auditing the current screen against the rules...")
        assessment: Assessment = self._llm.generate_response([message], Assessment)
        self._clean(assessment)

        if screen_id:
            self._by_screen[screen_id] = assessment
        return self._kept(assessment)

    @staticmethod
    def _kept(assessment: Assessment) -> Assessment:
        """A copy, because a caller withdraws findings from what it receives.

        The step loop drops a crash reported on the first look, then checks the
        screen again. Returning the stored object would carry that withdrawal
        into every later look at the same screen, and a crash that persists
        would stay invisible.
        """
        return assessment.model_copy(deep=True)

    def _clean(self, assessment: Assessment) -> None:
        """Masks declared secrets in what the auditor wrote, before anything keeps it."""
        if self._scrubber is None:
            return
        scrub = partial(safe_text, scrubber=self._scrubber)
        assessment.screen_name = scrub(assessment.screen_name)
        for verification in assessment.rule_verifications:
            verification.rule = scrub(verification.rule)
            verification.reasoning = scrub(verification.reasoning)
        for issue in assessment.detected_issues:
            issue.description = scrub(issue.description)
