"""What the planner is told about the scenario it is walking.

The planner carries out instructions. Another component judges the result,
and the planner's own verdicts are discarded. Shown the outcomes, it steers
the app until they hold, and the outcome then records a pass over a state the
planner arranged rather than one the app produced.

So these hold three places at once: the prompt text, the call site that
builds the planner, and the response schema, because a field description
travels to a model and can undo what a prompt says.
"""

import unittest

from fakes import qa_prompt, walk

from ronin.agent.device.launch import SetupMode
from types import SimpleNamespace
from unittest.mock import Mock

SENTINEL_OUTCOME = "the sign-in screen remains visible"
SENTINEL_SECURITY = "the card number is masked"


class PlannerBoundaryTests(unittest.TestCase):
    def _case_with_rules(self):
        """A scenario whose outcome is a sentinel, so a leak is unmistakable."""
        from ronin.testing.models import ExpectedOutcome, GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_001",
            scenario_name="Sign in",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Enter the email.", "Tap Login."),
            expected_outcomes=[ExpectedOutcome(statement=SENTINEL_OUTCOME)],
        )

    """What the planner is told, and what it must never be told.

    The planner carries a scenario out. The auditor judges it. A planner that
    reads the assertions can satisfy an assertion instead of the scenario, and
    its own verdicts reach no report.
    """


    def test_the_planner_is_told_no_rule_at_all(self):
        """Not the words, not the ids, not a heading that invites them."""
        prompt = qa_prompt()

        for rule in (SENTINEL_OUTCOME, SENTINEL_SECURITY):
            self.assertNotIn(rule, prompt)
        for naming in ("expected:", "security:", "Expected Outcomes", "Security Rules"):
            self.assertNotIn(naming, prompt)

    def test_a_scenario_carrying_rules_sends_none_of_them(self):
        """The real path: a scenario with rules, through to the sent message.

        Asserting on a planner built without rules proves little, because
        there was nothing to leak. This gives the scenario sentinel rules and
        reads what the planner actually sent.
        """
        from ronin.testing.qa_planner import QAPlanner

        test_case = self._case_with_rules()
        llm = SimpleNamespace(generate_response=Mock())

        # Built the way QAAgent builds it, from the scenario's setup alone.
        planner = QAPlanner(llm_client=llm, setup=test_case.setup)
        sent = planner._build_qa_prompt(
            goal_lines=test_case.steps_instructions,
            gui_elements=None,
            history="",
            trail=None,
            drift_warning=None,
            loop_nudge=None,
        )

        self.assertNotIn(SENTINEL_OUTCOME, sent)
        self.assertNotIn(SENTINEL_SECURITY, sent)
        self.assertIn("Tap Login", sent)

    def test_what_qa_agent_hands_the_planner_carries_no_rules(self):
        """The call site, not the prompt. A spy reads what was really passed.

        Reading the source would pass on a comment and fail on a reformat.
        This builds the planner through the real constructor and records the
        arguments it was given.
        """
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from ronin.agent.device.platform import Platform
        from ronin.agent.memory.history import AgentHistory
        from ronin.agent.run.agent import DeviceRunResult
        from ronin.testing.models import SetupMode
        from ronin.testing.qa_agent import QAAgent
        from ronin.utils.secrets import SecretScrubber

        agent = QAAgent.__new__(QAAgent)
        agent._fast_llm = object()
        agent._grounding = "marks"
        agent._device_agent = SimpleNamespace(
            max_steps=40,
            replay_ledger=None,
            scrubber=SecretScrubber(),
            run=Mock(
                return_value=DeviceRunResult(
                    status="step_limit",
                    steps_executed=1,
                    failure_reason="limit",
                    history=AgentHistory(),
                )
            ),
        )
        recorded = {}

        with patch("ronin.testing.qa_agent.QAPlanner") as planner:
            planner.side_effect = lambda **passed: recorded.update(passed) or Mock()
            agent.execute_scenario(
                self._case_with_rules(),
                [SENTINEL_SECURITY],
                SimpleNamespace(platform=Platform.ANDROID, serial=""),
                SimpleNamespace(reset=Mock()),
                Path("unused"),
                session_trail=object(),
            )

        self.assertEqual(recorded["setup"], SetupMode.CLEAR_DATA)
        self.assertNotIn("test_case", recorded)
        self.assertNotIn("security_rules", recorded)
        self.assertNotIn("expected_outcomes", recorded)

    def test_no_model_facing_field_asks_the_planner_to_judge(self):
        """A field description travels to the model inside the JSON schema.

        One that asks whether an outcome is observable undoes a prompt that
        says not to judge, and the prompt is the half a reader notices.
        """
        from ronin.testing.qa_planner import QAStepResponse

        schema = QAStepResponse.model_json_schema()["properties"]

        for field in ("goal_satisfied", "goal_evidence", "action", "is_scenario_feasible"):
            described = schema[field].get("description", "")
            self.assertNotIn("goal's stated outcome", described)
            self.assertNotIn("Judge the goal", described)
            self.assertNotIn("cannot be reached from here", described)

    def test_the_planner_is_asked_only_for_fields_it_can_answer(self):
        """The prompt requests only fields the planner's response model has."""
        prompt = qa_prompt()

        for field in ("detected_issues", "rule_verifications"):
            self.assertNotIn(field, prompt)

    def test_the_planner_is_told_the_state_the_scenario_starts_in(self):
        self.assertIn("data was cleared", qa_prompt(setup=SetupMode.CLEAR_DATA))
        self.assertIn("Sign in", qa_prompt())

    def test_no_blanket_claim_about_how_the_app_started(self):
        """`setup.describe()` says what is true. PRESERVE is not a fresh launch."""
        prompt = qa_prompt(setup=SetupMode.PRESERVE)

        self.assertNotIn("freshly launched", prompt)


if __name__ == "__main__":
    unittest.main()
