"""The id scheme three files agree on, and the drift it exists to prevent."""

import unittest

from ronin.agent.device.launch import SetupMode


class RuleIdTests(unittest.TestCase):
    def test_an_id_names_its_kind_and_position(self):
        from ronin.testing import rules

        self.assertEqual(rules.rule_id(rules.SECURITY, 1), "security:1")
        self.assertEqual(rules.rule_id(rules.EXPECTED, 0), "expected:0")

    def test_ids_cover_every_rule(self):
        from ronin.testing import rules

        self.assertEqual(
            rules.ids(["a", "b"], rules.EXPECTED), {"expected:0", "expected:1"}
        )

    def test_a_kind_is_recognised_from_its_id(self):
        from ronin.testing import rules

        self.assertTrue(rules.is_kind("security:3", rules.SECURITY))
        self.assertFalse(rules.is_kind("expected:3", rules.SECURITY))
        self.assertFalse(rules.is_kind(None, rules.SECURITY))


class RuleListingTests(unittest.TestCase):
    def test_both_kinds_are_listed_with_ids(self):
        from ronin.testing import rules

        listing = rules.every_rule(["Order confirmed"], ["Card masked", "CVV masked"])

        self.assertIn("[expected:0] Order confirmed", listing)
        self.assertIn("[security:0] Card masked", listing)
        self.assertIn("[security:1] CVV masked", listing)

    def test_a_scenario_with_no_rules_says_so(self):
        from ronin.testing import rules

        self.assertEqual(rules.every_rule([], []), rules.NONE_SPECIFIED)
        self.assertEqual(rules.listing([], rules.SECURITY), rules.NONE_SPECIFIED)


class PromptAgreementTests(unittest.TestCase):
    """The planner and the auditor must name a rule the same way.

    They are told the rules separately and both echo an id back, which the gate
    matches on. Drift between the two files would not fail anywhere — every
    verdict would simply stop matching, and every scenario would report missing
    evidence while its assertions visibly passed.
    """

    OUTCOMES = ["Order is confirmed"]
    SECURITY = ["The card number is masked"]

    def _planner_prompt(self):
        from types import SimpleNamespace

        from ronin.agent.plan.planner import GROUNDING_COORDINATES
        from ronin.testing.qa_planner import QAPlanner

        planner = QAPlanner.__new__(QAPlanner)
        planner.grounding = GROUNDING_COORDINATES
        planner._test_case = SimpleNamespace(expected_outcomes=self.OUTCOMES, setup=SetupMode.CLEAR_DATA)
        planner._security_rules = self.SECURITY
        return planner._build_qa_prompt(
            goal_lines=["Sign in"],
            gui_elements=None,
            history="",
            trail=None,
            drift_warning=None,
            loop_nudge=None,
        )

    def _auditor_prompt(self):
        """What the auditor actually sends, not what it ought to send."""
        from types import SimpleNamespace
        from unittest.mock import Mock

        from ronin.testing.qa_auditor import Assessment, QAAuditor

        llm = SimpleNamespace(
            generate_response=Mock(return_value=Assessment(screen_name="Login"))
        )
        QAAuditor(llm, self.OUTCOMES, self.SECURITY).audit(b"png", "screen")

        (messages, _model), _ = llm.generate_response.call_args
        return " ".join(
            part.get("text", "") for part in messages[0].to_dict()["content"]
        )

    def test_both_prompts_name_every_rule_identically(self):
        from ronin.testing import rules

        planner, auditor = self._planner_prompt(), self._auditor_prompt()

        for line in rules.labelled(self.OUTCOMES, rules.EXPECTED) + rules.labelled(
            self.SECURITY, rules.SECURITY
        ):
            self.assertIn(line, planner, "the planner names this rule differently")
            self.assertIn(line, auditor, "the auditor names this rule differently")
