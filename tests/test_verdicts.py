"""What counts as evidence that a scenario passed."""

import unittest
from types import SimpleNamespace


class VerdictTests(unittest.TestCase):
    def _agent(self):
        from ronin.testing.qa_agent import QAAgent

        return QAAgent.__new__(QAAgent)

    def _result(self, status="goal_reached", reason=None):
        return SimpleNamespace(status=status, failure_reason=reason)

    def _output(self, feasible=True):
        """What the planner answers: whether it can proceed."""
        return SimpleNamespace(
            is_scenario_feasible=feasible, infeasibility_reason=None
        )

    def _assessment(self):
        """What the auditor answers: rules and defects, never the plan's goal."""
        return SimpleNamespace(rule_verifications=[], detected_issues=[])

    def _verdict(self, **overrides):
        values = {
            "run_result": self._result(),
            "stop_reason": None,
            "all_verifications": [],
            "all_issues": [],
            "qa_outputs": [self._output()],
            "assessments": [self._assessment()],
            "required_outcomes": [],
            "security_rules": [],
        }
        values.update(overrides)
        return self._agent()._determine_status(**values)

    def test_model_done_without_observable_evidence_is_not_a_pass(self):
        """The planner wanting to be done is not evidence that it is.

        Completion needs a rule the auditor confirmed on a screen it judged
        without the plan. A scenario naming no expected outcome offers nothing
        to confirm, so the planner's say-so is all that is left.
        """
        status, reason = self._verdict(required_outcomes=[])

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("planner's own word", reason)

    def test_missing_required_verification_is_not_a_pass(self):
        status, reason = self._verdict(required_outcomes=["Order is confirmed"])

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("expected:0", reason)

    def test_security_rule_a_scenario_never_reached_is_not_required(self):
        from ronin.testing.models import RuleVerification

        status, reason = self._verdict(
            required_outcomes=["Order is confirmed"],
            security_rules=["Card is masked on checkout"],
            all_verifications=[
                RuleVerification(
                    rule="Order is confirmed",
                    rule_id="expected:0",
                    status="passed",
                    reasoning="CONFIRMING is visible",
                )
            ],
        )

        self.assertEqual(status, "PASSED", reason)

    def test_security_rule_the_planner_reported_untested_is_required(self):
        from ronin.testing.models import RuleVerification

        status, reason = self._verdict(
            required_outcomes=[],
            security_rules=["Card is masked on checkout"],
            all_verifications=[
                RuleVerification(
                    rule="Card is masked on checkout",
                    rule_id="security:0",
                    status="untested",
                    reasoning="Could not read the field",
                )
            ],
        )

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("security:0", reason)

    def test_all_required_verifications_can_pass(self):
        from ronin.testing.models import RuleVerification

        status, reason = self._verdict(
            required_outcomes=["Order is confirmed"],
            security_rules=["Card is masked"],
            all_verifications=[
                RuleVerification(
                    rule="Order is confirmed",
                    rule_id="expected:0",
                    status="passed",
                    reasoning="CONFIRMING is visible",
                ),
                RuleVerification(
                    rule="Card is masked",
                    rule_id="security:0",
                    status="passed",
                    reasoning="No PAN is visible",
                ),
            ],
        )

        self.assertEqual((status, reason), ("PASSED", None))

    def test_harness_exception_is_distinct_from_app_failure(self):
        status, _reason = self._verdict(
            run_result=self._result("exception", "connection refused")
        )

        self.assertEqual(status, "ERROR")

    def test_all_skipped_suite_is_not_a_pass(self):
        from ronin.testing.runner import aggregate_journey_status

        self.assertEqual(aggregate_journey_status(0, 0), "FAILED")
        self.assertEqual(aggregate_journey_status(1, 0), "PASSED")
        self.assertEqual(aggregate_journey_status(1, 1), "FAILED")


class FirstLookTests(unittest.TestCase):
    """Verdicts read off a screen that may still be painting are withdrawn once."""

    def _response(self, feasible=True):
        return SimpleNamespace(
            is_scenario_feasible=feasible,
            action=SimpleNamespace(type="done", reason="stop"),
        )

    def _assessment(self, crash=False):
        from ronin.testing.models import QAIssue

        issues = []
        if crash:
            issues.append(
                QAIssue(
                    issue_type="crash_screen",
                    severity="critical",
                    description="Blank screen",
                    current_screen="Blank Screen",
                )
            )
        return SimpleNamespace(detected_issues=issues)

    def test_a_crash_on_the_first_look_is_withdrawn(self):
        from ronin.testing.qa_agent import _defer_first_look

        response, assessment = self._response(), self._assessment(crash=True)

        self.assertTrue(_defer_first_look(response, assessment, first_look=True))
        self.assertEqual(assessment.detected_issues, [])
        self.assertEqual(response.action.type, "wait")

    def test_a_crash_seen_later_stands(self):
        from ronin.testing.qa_agent import _defer_first_look

        response, assessment = self._response(), self._assessment(crash=True)

        self.assertFalse(_defer_first_look(response, assessment, first_look=False))
        self.assertEqual(len(assessment.detected_issues), 1)

    def test_infeasibility_on_the_first_look_is_withdrawn(self):
        from ronin.testing.qa_agent import _defer_first_look

        response, assessment = self._response(feasible=False), self._assessment()

        self.assertTrue(_defer_first_look(response, assessment, first_look=True))
        self.assertTrue(response.is_scenario_feasible)
        self.assertEqual(response.action.type, "wait")

    def test_infeasibility_seen_later_stands(self):
        from ronin.testing.qa_agent import _defer_first_look

        response, assessment = self._response(feasible=False), self._assessment()

        self.assertFalse(_defer_first_look(response, assessment, first_look=False))
        self.assertFalse(response.is_scenario_feasible)

    def test_a_healthy_first_look_is_left_alone(self):
        from ronin.testing.qa_agent import _defer_first_look

        response, assessment = self._response(), self._assessment()

        self.assertFalse(_defer_first_look(response, assessment, first_look=True))
        self.assertEqual(response.action.type, "done")

    def test_withdrawing_a_verdict_replaces_the_done_that_came_with_it(self):
        """Clearing the verdict alone changes the label. The loop still acts on `done`."""
        from ronin.agent.actuation.actions import ActionType
        from ronin.testing.qa_agent import _defer_first_look

        response, assessment = self._response(), self._assessment(crash=True)

        _defer_first_look(response, assessment, first_look=True)

        self.assertEqual(response.action.type, ActionType.WAIT)
        self.assertGreater(response.action.duration, 0)


class VerificationAggregationTests(unittest.TestCase):
    """Every rule is re-evaluated on every screen, so answers pile up per rule."""

    def _verification(self, rule_id, status, applicable=True, rule="A rule"):
        from ronin.testing.models import RuleVerification

        return RuleVerification(
            rule=rule,
            rule_id=rule_id,
            status=status,
            applicable=applicable,
            reasoning="because",
        )

    def _verdict(self, **overrides):
        from ronin.testing.qa_agent import QAAgent

        values = {
            "run_result": SimpleNamespace(status="goal_reached", failure_reason=None),
            "stop_reason": None,
            "all_verifications": [],
            "all_issues": [],
            "qa_outputs": [
                SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)
            ],
            "assessments": [
                SimpleNamespace(rule_verifications=[], detected_issues=[])
            ],
            "required_outcomes": [],
            "security_rules": [],
        }
        values.update(overrides)

        # A scenario passes only once the auditor evidenced an expected
        # outcome, so a test about any other rule needs one in place to reach
        # the behaviour it is checking.
        if "required_outcomes" not in overrides:
            values["required_outcomes"] = ["Order is confirmed"]
            values["all_verifications"] = list(values["all_verifications"]) + [
                self._verification("expected:0", "passed", rule="Order is confirmed")
            ]
        return QAAgent.__new__(QAAgent)._determine_status(**values)

    def test_an_outcome_checked_too_early_does_not_condemn_the_scenario(self):
        status, reason = self._verdict(
            required_outcomes=["Order is confirmed"],
            all_verifications=[
                self._verification("expected:0", "failed"),
                self._verification("expected:0", "failed"),
                self._verification("expected:0", "passed"),
            ],
        )

        self.assertEqual(status, "PASSED", reason)

    def test_an_outcome_that_ends_failed_fails_the_scenario(self):
        status, reason = self._verdict(
            required_outcomes=["Order is confirmed"],
            all_verifications=[
                self._verification("expected:0", "passed"),
                self._verification("expected:0", "failed"),
            ],
        )

        self.assertEqual(status, "FAILED")
        self.assertIn("Rule validation failed", reason)

    def test_a_security_rule_broken_once_stays_broken(self):
        status, reason = self._verdict(
            security_rules=["Password is never shown"],
            all_verifications=[
                self._verification("security:0", "failed", rule="Password is never shown"),
                self._verification("security:0", "passed", rule="Password is never shown"),
            ],
        )

        self.assertEqual(status, "FAILED")
        self.assertIn("Security rule violated", reason)

    def test_a_rule_passed_where_it_could_not_apply_is_not_evidence(self):
        status, reason = self._verdict(
            required_outcomes=["Order is confirmed"],
            all_verifications=[
                self._verification("expected:0", "passed", applicable=False)
            ],
        )

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("expected:0", reason)

    def test_a_dynamic_assertion_does_not_decide_the_scenario(self):
        status, reason = self._verdict(
            all_verifications=[self._verification(None, "failed", rule="Button is blue")]
        )

        self.assertEqual(status, "PASSED", reason)


class SecurityScopingTests(VerificationAggregationTests):
    """The planner reports every rule on every screen, so a mention proves nothing."""

    def test_a_rule_only_ever_reported_non_applicable_is_not_required(self):
        status, reason = self._verdict(
            security_rules=["Card is masked on checkout"],
            all_verifications=[
                self._verification("security:0", "untested", applicable=False),
                self._verification("security:0", "untested", applicable=False),
            ],
        )

        self.assertEqual(status, "PASSED", reason)

    def test_a_rule_applicable_but_never_passed_is_required(self):
        status, reason = self._verdict(
            security_rules=["Card is masked on checkout"],
            all_verifications=[
                self._verification("security:0", "untested", applicable=True)
            ],
        )

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("security:0", reason)

    def test_evidence_survives_later_screens_the_rule_does_not_reach(self):
        status, reason = self._verdict(
            security_rules=["Card is masked on checkout"],
            all_verifications=[
                self._verification("security:0", "passed", applicable=True),
                self._verification("security:0", "untested", applicable=False),
            ],
        )

        self.assertEqual(status, "PASSED", reason)


class SecurityCoverageTests(unittest.TestCase):
    """Scoping rules per scenario leaves a hole, and the suite has to close it."""

    def _scenario(self, verifications):
        from ronin.testing.models import RuleVerification, ScenarioReport

        return ScenarioReport(
            test_id="TS_1",
            scenario_name="Scenario",
            classification="happy_path",
            status="PASSED",
            rule_verifications=[
                RuleVerification(
                    rule=rule,
                    rule_id=rule_id,
                    status=status,
                    applicable=applicable,
                    reasoning="because",
                )
                for rule, rule_id, status, applicable in verifications
            ],
        )

    def _unevidenced(self, rules, scenarios):
        from ronin.testing.coverage import unevidenced_security_rules

        return unevidenced_security_rules(rules, scenarios)

    def test_a_rule_no_scenario_checked_is_reported(self):
        scenarios = [self._scenario([("Password masked", "security:0", "passed", True)])]

        self.assertEqual(
            self._unevidenced(["Password masked", "Card masked"], scenarios),
            ["Card masked"],
        )

    def test_a_rule_evidenced_by_any_scenario_counts(self):
        scenarios = [
            self._scenario([("Card masked", "security:1", "untested", True)]),
            self._scenario([("Card masked", "security:1", "passed", True)]),
        ]

        self.assertEqual(self._unevidenced(["Password masked", "Card masked"], scenarios), ["Password masked"])

    def test_a_pass_where_the_rule_could_not_apply_is_not_coverage(self):
        scenarios = [self._scenario([("Card masked", "security:0", "passed", False)])]

        self.assertEqual(self._unevidenced(["Card masked"], scenarios), ["Card masked"])

    def test_every_rule_evidenced_leaves_nothing_to_report(self):
        scenarios = [self._scenario([("Card masked", "security:0", "passed", True)])]

        self.assertEqual(self._unevidenced(["Card masked"], scenarios), [])


class AbortReportingTests(unittest.TestCase):
    """A run that dies before any scenario exists must not look like a pass."""

    def _runner(self):
        from ronin.testing.runner import RoninTestRunner

        return RoninTestRunner.__new__(RoninTestRunner)

    def test_generation_failure_is_reported_as_a_failed_test(self):
        reports = []

        failed = self._runner()._record_aborted([], reports, RuntimeError("no scenarios"))

        self.assertEqual(failed, 1)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].status, "ERROR")
        self.assertEqual(reports[0].test_id, "scenario_generation")
        self.assertIn("no scenarios", reports[0].failure_reason)

    def test_unfinished_scenarios_are_each_recorded(self):
        from ronin.testing.models import GeneratedTestCase, ScenarioReport

        planned = [
            GeneratedTestCase(
                test_id=f"TS_{n}",
                scenario_name=f"S{n}",
                classification="happy_path",
                steps_instructions=["x"],
                expected_outcomes=[],
            )
            for n in (1, 2)
        ]
        reports = [
            ScenarioReport(
                test_id="TS_1", scenario_name="S1", classification="happy_path", status="PASSED"
            )
        ]

        failed = self._runner()._record_aborted(planned, reports, KeyboardInterrupt())

        self.assertEqual(failed, 1)
        self.assertEqual([r.test_id for r in reports], ["TS_1", "TS_2"])


class RuleIdentityTests(unittest.TestCase):
    """The id names the rule, whichever field the auditor put it in.

    The gate matches a verdict to its rule by id alone, so an id read from
    nowhere discards an answer the auditor gave.
    """

    def _verification(self, **fields):
        from ronin.testing.models import RuleVerification

        return RuleVerification(status="passed", reasoning="because", **fields)

    def test_an_id_in_the_id_field_is_kept(self):
        answer = self._verification(rule="Order is confirmed", rule_id="expected:3")

        self.assertEqual(answer.rule_id, "expected:3")

    def test_an_id_in_the_rule_field_is_read_from_there(self):
        """The auditor answers this way more often than not."""
        answer = self._verification(rule="expected:0")

        self.assertEqual(answer.rule_id, "expected:0")

    def test_the_prompt_line_it_was_read_from_still_names_its_rule(self):
        """The rules reach the auditor as `- [expected:0] ...` and come back whole."""
        answer = self._verification(rule="[expected:11] Checkout shows the CVV field")

        self.assertEqual(answer.rule_id, "expected:11")

    def test_a_security_id_is_recovered_the_same_way(self):
        answer = self._verification(rule="security:2 — the CVV must stay masked")

        self.assertEqual(answer.rule_id, "security:2")

    def test_an_assertion_with_no_id_keeps_none(self):
        """The auditor may raise a rule of its own, and those have no id."""
        answer = self._verification(rule="The subtotal adds up")

        self.assertIsNone(answer.rule_id)

    def test_a_number_that_is_not_an_id_is_not_read_as_one(self):
        answer = self._verification(rule="The cart shows 2 items at 3:15")

        self.assertIsNone(answer.rule_id)

    def test_the_id_field_wins_when_both_carry_one(self):
        """It is the field the prompt asked for, so it is the deliberate answer."""
        answer = self._verification(rule="[expected:9] Payment screen", rule_id="expected:9")

        self.assertEqual(answer.rule_id, "expected:9")


class EvidenceReachesTheGateTests(unittest.TestCase):
    """An answered outcome must settle its rule, not report as unevidenced."""

    def _verdict(self, verifications, required):
        from ronin.testing.qa_agent import QAAgent

        return QAAgent.__new__(QAAgent)._determine_status(
            run_result=SimpleNamespace(status="goal_reached", failure_reason=None),
            stop_reason=None,
            all_verifications=verifications,
            all_issues=[],
            qa_outputs=[
                SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)
            ],
            assessments=[SimpleNamespace(rule_verifications=[], detected_issues=[])],
            required_outcomes=required,
            security_rules=[],
        )

    def test_an_outcome_answered_in_the_rule_field_passes_the_scenario(self):
        from ronin.testing.models import RuleVerification

        answered = RuleVerification(
            rule="expected:0", status="passed", reasoning="The screen shows it"
        )

        status, reason = self._verdict([answered], ["Sign-in screen is visible"])

        self.assertEqual(status, "PASSED")
        self.assertIsNone(reason)
