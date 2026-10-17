"""Outcomes about where the run went, which no one screenshot can settle."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ronin.testing import transitions
from ronin.testing.models import AssertionScope, ExpectedOutcome, RuleVerification

SIGN_IN = "Sign-in screen"
LOCATION = "Location Access screen"
HOME = "Home screen"


def screen(name):
    return SimpleNamespace(screen_name=name)


def outcome(statement, scope=AssertionScope.TRANSITION):
    return ExpectedOutcome(statement=statement, scope=scope)


def verdicts(*answers):
    """An LLM that answers with these verdicts and records what it was asked."""
    answer = transitions.TransitionVerdicts(
        verdicts=[
            RuleVerification(
                rule=rule, rule_id=rule, status=status, reasoning=reasoning
            )
            for rule, status, reasoning in answers
        ]
    )
    return SimpleNamespace(generate_response=Mock(return_value=answer))


def asked(llm):
    (messages, _model), _ = llm.generate_response.call_args
    return messages[0].content


class ScopeSelectionTests(unittest.TestCase):
    """The two selectors must divide the outcomes, not overlap or drop any."""

    def test_a_transition_is_claimed_only_by_the_transition_selector(self):
        from ronin.testing import rules

        outcomes = [outcome("Tapping Login reaches Location Access")]

        self.assertEqual(rules.transitions_only(outcomes), outcomes)
        self.assertEqual(rules.on_screen_only(outcomes), [None])

    def test_a_bare_statement_belongs_to_the_screen_selector(self):
        """It carries no scope, and a claim about one screen is the common case."""
        from ronin.testing import rules

        outcomes = ["The cart shows a subtotal"]

        self.assertEqual(rules.on_screen_only(outcomes), outcomes)
        self.assertEqual(rules.transitions_only(outcomes), [None])

    def test_every_outcome_is_claimed_by_exactly_one_selector(self):
        """Claimed twice is counted twice. Claimed by neither can never settle."""
        from ronin.testing import rules

        outcomes = [
            "The cart shows a subtotal",
            outcome("Tapping Login reaches Location Access"),
            outcome("The app stays responsive", AssertionScope.RUN_TIMING),
        ]

        on_screen = rules.on_screen_only(outcomes)
        moved = rules.transitions_only(outcomes)

        for index, _ in enumerate(outcomes):
            claims = [bool(on_screen[index]), bool(moved[index])]
            # A run_timing outcome is claimed by neither, and `rules.unprovable`
            # is what keeps the gate from demanding it.
            self.assertLessEqual(sum(claims), 1, f"outcome {index} claimed twice")

    def test_an_index_survives_the_blanking_around_it(self):
        """The id a verdict carries is the outcome's position, so it cannot shift."""
        from ronin.testing import rules

        outcomes = [
            "A screen claim",
            "Another screen claim",
            outcome("Tapping Place Order reaches Order Status"),
        ]

        listed = rules.listing(rules.transitions_only(outcomes), rules.EXPECTED)

        self.assertIn("[expected:2]", listed)
        self.assertNotIn("[expected:0]", listed)


class SettlingTests(unittest.TestCase):
    """What the order of screens can and cannot answer."""

    def _settle(self, llm, outcomes, names):
        return transitions.settle(llm, outcomes, [screen(name) for name in names])

    def test_a_screen_the_run_reached_settles_the_claim(self):
        llm = verdicts(("expected:0", "passed", f"{SIGN_IN} then {LOCATION}"))

        settled = self._settle(
            llm,
            [outcome("Tapping Login reaches the Location Access screen")],
            [SIGN_IN, LOCATION, HOME],
        )

        self.assertEqual([v.rule_id for v in settled], ["expected:0"])
        self.assertEqual(settled[0].status, "passed")

    def test_the_screens_are_sent_in_the_order_they_were_visited(self):
        """A claim that one screen follows another needs the order, not the set."""
        llm = verdicts(("expected:0", "passed", LOCATION))

        self._settle(llm, [outcome("Login reaches Location Access")], [SIGN_IN, LOCATION])

        sent = asked(llm)
        self.assertLess(sent.index(SIGN_IN), sent.index(LOCATION))

    def test_one_visit_read_twice_is_named_once(self):
        """The auditor reads a screen on arrival and on departure."""
        llm = verdicts(("expected:0", "passed", LOCATION))

        self._settle(
            llm,
            [outcome("Login reaches Location Access")],
            [SIGN_IN, SIGN_IN, LOCATION],
        )

        self.assertEqual(asked(llm).count(f"1. {SIGN_IN}"), 1)
        self.assertIn(f"2. {LOCATION}", asked(llm))

    def test_coming_back_to_a_screen_is_named_again(self):
        """Leaving and returning is two arrivals, and a claim may rest on that."""
        llm = verdicts(("expected:0", "passed", HOME))

        self._settle(
            llm, [outcome("Back from the cart reaches Home")], [HOME, "My Cart", HOME]
        )

        self.assertIn(f"1. {HOME}", asked(llm))
        self.assertIn(f"3. {HOME}", asked(llm))

    def test_a_verdict_without_an_id_is_dropped(self):
        """The gate matches on the id, so an unnamed verdict settles nothing."""
        answer = transitions.TransitionVerdicts(
            verdicts=[
                RuleVerification(rule="Something happened", status="passed", reasoning="yes")
            ]
        )
        llm = SimpleNamespace(generate_response=Mock(return_value=answer))

        settled = self._settle(llm, [outcome("Login reaches Location Access")], [SIGN_IN])

        self.assertEqual(settled, [])

    def test_a_scenario_with_no_transition_outcome_is_not_judged(self):
        llm = verdicts()

        settled = self._settle(llm, ["The cart shows a subtotal"], [SIGN_IN])

        self.assertEqual(settled, [])
        llm.generate_response.assert_not_called()

    def test_a_run_that_named_no_screen_is_not_judged(self):
        """With nothing to read, a verdict would be invention."""
        llm = verdicts()

        settled = self._settle(llm, [outcome("Login reaches Location Access")], [None, ""])

        self.assertEqual(settled, [])
        llm.generate_response.assert_not_called()

    def test_the_verdict_is_filed_against_the_screen_it_names(self):
        llm = verdicts(("expected:0", "passed", f"reached {LOCATION} after {SIGN_IN}"))

        settled = self._settle(
            llm, [outcome("Login reaches Location Access")], [SIGN_IN, LOCATION]
        )

        self.assertEqual(settled[0].screen_name, LOCATION)

    def test_secrets_are_scrubbed_from_what_the_judge_wrote(self):
        llm = verdicts(("expected:0", "passed", "typed hunter2 and arrived"))
        scrubber = SimpleNamespace(scrub=lambda text: (text or "").replace("hunter2", "***"))

        settled = transitions.settle(
            llm,
            [outcome("Login reaches Location Access")],
            [screen(SIGN_IN)],
            scrubber=scrubber,
        )

        self.assertNotIn("hunter2", settled[0].reasoning)


class GateIntegrationTests(unittest.TestCase):
    """A settled transition has to reach the verdict as the auditor's does."""

    def _status(self, verifications, outcomes):
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
            required_outcomes=outcomes,
            security_rules=[],
        )

    def test_a_scenario_passes_once_both_kinds_are_evidenced(self):
        outcomes = ["The cart shows a subtotal", outcome("Login reaches Location Access")]
        evidence = [
            RuleVerification(rule="expected:0", status="passed", reasoning="seen"),
            RuleVerification(rule="expected:1", status="passed", reasoning="in order"),
        ]

        status, reason = self._status(evidence, outcomes)

        self.assertEqual(status, "PASSED")
        self.assertIsNone(reason)

    def test_an_unjudged_transition_still_holds_the_scenario_back(self):
        """A failed judgement must not read as a pass."""
        outcomes = ["The cart shows a subtotal", outcome("Login reaches Location Access")]
        evidence = [RuleVerification(rule="expected:0", status="passed", reasoning="seen")]

        status, reason = self._status(evidence, outcomes)

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("expected:1", reason)


class AnsweredClaimsOnlyTests(unittest.TestCase):
    """A judge may answer the claims it was shown, and no others."""

    def _settle(self, answer, outcomes):
        llm = SimpleNamespace(generate_response=Mock(return_value=answer))
        return transitions.settle(llm, outcomes, [screen(SIGN_IN), screen(LOCATION)])

    def _answer(self, *answers):
        return transitions.TransitionVerdicts(
            verdicts=[
                RuleVerification(rule=rid, rule_id=rid, status=status, reasoning="r")
                for rid, status in answers
            ]
        )

    def test_a_claim_it_was_never_shown_is_ignored(self):
        """expected:0 here is an on-screen outcome, judged by the auditor."""
        outcomes = ["The cart shows a subtotal", outcome("Login reaches Location Access")]

        settled = self._settle(
            self._answer(("expected:0", "passed"), ("expected:1", "passed")), outcomes
        )

        self.assertEqual([v.rule_id for v in settled], ["expected:1"])

    def test_a_fabricated_security_verdict_is_ignored(self):
        """A failed security rule fails the scenario, so an invented one is worse
        than a false pass."""
        outcomes = [outcome("Login reaches Location Access")]

        settled = self._settle(
            self._answer(("security:1", "failed"), ("expected:0", "passed")), outcomes
        )

        self.assertEqual([v.rule_id for v in settled], ["expected:0"])

    def test_an_id_past_the_end_of_the_list_is_ignored(self):
        outcomes = [outcome("Login reaches Location Access")]

        settled = self._settle(self._answer(("expected:7", "passed")), outcomes)

        self.assertEqual(settled, [])
