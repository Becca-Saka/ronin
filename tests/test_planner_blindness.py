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
                SimpleNamespace(
                    platform=Platform.ANDROID,
                    serial="",
                    os_version=None,
                    package_id=None,
                    driver=None,
                ),
                SimpleNamespace(reset=Mock()),
                Path("unused"),
                session_trail=object(),
            )

        self.assertEqual(recorded["setup"], SetupMode.CLEAR_DATA)
        self.assertNotIn("test_case", recorded)
        self.assertNotIn("security_rules", recorded)
        self.assertNotIn("expected_outcomes", recorded)

    def test_every_judging_field_rewrites_the_wording_it_inherits(self):
        """A field description travels to the model inside the JSON schema.

        An inherited one that asks whether the goal was reached undoes a
        prompt saying not to judge, and the prompt is the half a reader
        notices.

        Compared against the base description rather than against remembered
        words. A deleted override then fails here, and a reworded one does
        not. The fields are named because which of them carry the risk is a
        judgement, and the check on each is derived.
        """
        from ronin.agent.plan.planner import DeviceStepResponse
        from ronin.testing.qa_planner import QAStepResponse

        base = DeviceStepResponse.model_json_schema()["properties"]
        qa = QAStepResponse.model_json_schema()["properties"]

        for field in ("goal_satisfied", "goal_evidence", "action", "is_scenario_feasible"):
            described = qa[field].get("description", "")
            self.assertTrue(described, f"{field} reaches the model undescribed")
            self.assertNotEqual(
                described, base.get(field, {}).get("description"), field
            )

    def test_the_planner_is_asked_only_for_fields_it_can_answer(self):
        """The prompt requests only fields the planner's response model has.

        Asking for one the model lacks spends tokens on a verdict pydantic
        drops. The premise is asserted first, so a field added to the model
        fails here rather than leaving the rule below it enforcing nothing.
        """
        from ronin.testing.qa_planner import QAStepResponse

        prompt = qa_prompt()

        for field in ("detected_issues", "rule_verifications"):
            self.assertNotIn(field, QAStepResponse.model_fields, field)
            self.assertNotIn(field, prompt, field)

    def test_the_planner_is_told_the_state_the_scenario_starts_in(self):
        self.assertIn(
            SetupMode.CLEAR_DATA.describe(), qa_prompt(setup=SetupMode.CLEAR_DATA)
        )
        self.assertIn("Sign in", qa_prompt())

    def test_the_prompt_claims_only_what_this_setup_did(self):
        """`describe()` says what is true, and PRESERVE is not a fresh launch.

        Every mode's own words come off the enum, so a new mode or a reworded
        one is covered without this test being told about it.
        """
        for mode in SetupMode:
            prompt = qa_prompt(setup=mode)

            self.assertIn(mode.describe(), prompt, mode.value)
            for other in SetupMode:
                if other is not mode:
                    self.assertNotIn(other.describe(), prompt, other.value)


if __name__ == "__main__":
    unittest.main()
