"""Which half of the loop owns an answer, and when an answer still counts."""

import unittest
from types import SimpleNamespace


class AuditorOwnershipTests(unittest.TestCase):
    """Which half of the loop owns which answer."""

    def test_the_planner_is_not_asked_for_verdicts(self):
        """A planner grading the work it just chose is not a second opinion."""
        from ronin.testing.qa_planner import QAStepResponse

        fields = QAStepResponse.model_fields

        self.assertNotIn("rule_verifications", fields)
        self.assertNotIn("detected_issues", fields)

    def test_the_auditor_is_asked_for_them(self):
        from ronin.testing.qa_auditor import Assessment

        fields = Assessment.model_fields

        self.assertIn("rule_verifications", fields)
        self.assertIn("detected_issues", fields)

    def test_completion_stays_with_the_planner(self):
        """Only the planner is told the goal, so only it can say the goal is met."""
        from ronin.testing.qa_auditor import Assessment
        from ronin.testing.qa_planner import QAStepResponse

        self.assertIn("goal_satisfied", QAStepResponse.model_fields)
        self.assertNotIn("goal_satisfied", Assessment.model_fields)
        self.assertNotIn("goal_evidence", Assessment.model_fields)

    def test_feasibility_stays_with_the_planner(self):
        """Whether the agent can proceed is not a statement about the app."""
        from ronin.testing.qa_auditor import Assessment
        from ronin.testing.qa_planner import QAStepResponse

        self.assertIn("is_scenario_feasible", QAStepResponse.model_fields)
        self.assertNotIn("is_scenario_feasible", Assessment.model_fields)


class CrashRankingTests(unittest.TestCase):
    """A crash is something the app did. The rest describe the run."""

    def _verdict_for(self, **overrides):
        from ronin.testing.models import QAIssue
        from ronin.testing.qa_agent import _terminal_outcome

        values = {
            "run_result": SimpleNamespace(status="hook_stop", failure_reason=None),
            "stop_reason": "Scenario is infeasible: the screen cannot be reached",
            "verifications": [],
            "issues": [
                QAIssue(
                    issue_type="crash_screen",
                    severity="critical",
                    description="blank screen",
                )
            ],
            "outputs": [],
        }
        values.update(overrides)
        return _terminal_outcome(**values)

    def test_a_crash_outranks_an_infeasible_verdict(self):
        """Reported as skipped, a crash looks like a scenario nobody ran."""
        status, reason = self._verdict_for()

        self.assertEqual(status, "FAILED")
        self.assertIn("crash", reason.lower())

    def test_infeasibility_still_reports_when_nothing_crashed(self):
        status, _reason = self._verdict_for(issues=[])

        self.assertEqual(status, "SKIPPED")


class DurableOutcomeEvidenceTests(unittest.TestCase):
    """An outcome seen in passing stays proven after the screen is left.

    A long address is accepted on the checkout screen. The order status screen
    that follows has no address field, so the rule reports untested there.
    Reading that as the verdict lets walking away erase the proof.
    """

    def _verification(self, rule_id, status, applicable=True):
        from ronin.testing.models import RuleVerification

        return RuleVerification(
            rule="The address field accepts the long input",
            rule_id=rule_id,
            status=status,
            applicable=applicable,
            reasoning="because",
        )

    def _missing(self, verifications):
        from ronin.testing.qa_agent import _missing_required_ids

        return _missing_required_ids(
            verifications, ["The address field accepts the long input"], []
        )

    def test_a_pass_survives_a_screen_the_outcome_cannot_be_seen_on(self):
        missing = self._missing(
            [
                self._verification("expected:0", "passed"),
                self._verification("expected:0", "untested", applicable=False),
            ]
        )

        self.assertEqual(missing, [])

    def test_an_outcome_never_seen_is_still_missing(self):
        missing = self._missing(
            [self._verification("expected:0", "untested", applicable=False)]
        )

        self.assertEqual(missing, ["expected:0"])

    def test_an_outcome_that_later_broke_where_it_could_be_seen_is_missing(self):
        """A failure on a screen the rule applies to still overrides the pass."""
        missing = self._missing(
            [
                self._verification("expected:0", "passed"),
                self._verification("expected:0", "failed"),
            ]
        )

        self.assertEqual(missing, ["expected:0"])

    def test_an_early_not_yet_does_not_block_a_later_pass(self):
        missing = self._missing(
            [
                self._verification("expected:0", "untested"),
                self._verification("expected:0", "passed"),
            ]
        )

        self.assertEqual(missing, [])


class ContradictedExposureTests(unittest.TestCase):
    """The auditor reports rules and defects together, and they can disagree.

    They can disagree about one screen: every masking rule passed, and the
    same answer called a bare "CVV:" label a plain-text CVV.
    """

    def _issue(self, kind="sensitive_data_exposure", screen="Checkout Details"):
        from ronin.testing.models import QAIssue

        issue = QAIssue(
            issue_type=kind,
            severity="critical",
            description="The CVV is visible in plain text",
        )
        issue.current_screen = screen
        return issue

    def _verification(self, status, screen="Checkout Details", rule_id="security:2"):
        from ronin.testing.models import RuleVerification

        verification = RuleVerification(
            rule="The CVV must never be displayed in plain text",
            rule_id=rule_id,
            status=status,
            applicable=True,
            reasoning="only 'CVV:' is visible",
        )
        verification.screen_name = screen
        return verification

    def _verdict_for(self, issues, verifications):
        from ronin.testing.qa_agent import _terminal_outcome

        return _terminal_outcome(
            run_result=SimpleNamespace(status="goal_reached", failure_reason=None),
            stop_reason=None,
            verifications=verifications,
            issues=issues,
            outputs=[],
        )

    def test_an_exposure_claim_the_rules_deny_does_not_fail_the_scenario(self):
        outcome = self._verdict_for([self._issue()], [self._verification("passed")])

        self.assertIsNone(outcome)

    def test_an_exposure_claim_the_rules_confirm_still_fails(self):
        outcome = self._verdict_for([self._issue()], [self._verification("failed")])

        self.assertEqual(outcome[0], "FAILED")

    def test_an_exposure_claim_on_a_screen_with_no_rules_still_fails(self):
        """Nothing checked it, so there is no verdict to weigh against."""
        outcome = self._verdict_for([self._issue()], [])

        self.assertEqual(outcome[0], "FAILED")

    def test_rules_from_another_screen_do_not_excuse_it(self):
        outcome = self._verdict_for(
            [self._issue(screen="Checkout Details")],
            [self._verification("passed", screen="Login Screen")],
        )

        self.assertEqual(outcome[0], "FAILED")

    def test_a_crash_is_never_excused_this_way(self):
        """Only exposure claims are weighed against masking rules."""
        outcome = self._verdict_for(
            [self._issue(kind="crash_screen")], [self._verification("passed")]
        )

        self.assertEqual(outcome[0], "FAILED")
