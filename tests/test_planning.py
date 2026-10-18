"""What Ronin asks the model, and what it records about the answer."""

import unittest

from ronin.agent.device.launch import SetupMode
from types import SimpleNamespace
from unittest.mock import Mock
from ronin.agent.run.agent import (
    AgentHistory,
    DeviceAgent,
    RollingLoopDetector,
    RunContext,
)


SENTINEL_OUTCOME = "the sign-in screen remains visible"
SENTINEL_SECURITY = "the card number is masked"


def qa_prompt(goal_lines=("Sign in",), setup=SetupMode.CLEAR_DATA):
    """The prompt a real QAPlanner builds, through its real constructor.

    Through the constructor, so a field it stopped taking cannot be assigned
    back in and go on looking present.
    """
    from ronin.testing.qa_planner import QAPlanner

    llm = SimpleNamespace(generate_response=Mock())
    planner = QAPlanner(llm_client=llm, setup=setup)
    return planner._build_qa_prompt(
        goal_lines=list(goal_lines),
        gui_elements=None,
        history="",
        trail=None,
        drift_warning=None,
        loop_nudge=None,
    )


class ProviderNormalizationTests(unittest.TestCase):
    """Schemas are advisory, so equivalent shapes have to be accepted."""

    def test_the_planner_accepts_a_coordinate_pair(self):
        from ronin.testing.qa_planner import QAStepResponse

        response = QAStepResponse.model_validate(
            {
                "current_screen": "Login",
                "observation": "Login form",
                "reasoning": "Focus email",
                "action": {"type": "tap", "coordinates": [10, 20], "reason": "focus"},
            }
        )

        self.assertEqual(response.action.coordinates.x, 10)

    def test_the_auditor_accepts_a_compact_rule_mapping(self):
        from ronin.testing.qa_auditor import Assessment

        assessment = Assessment.model_validate(
            {
                "screen_name": "Login",
                "rule_verifications": {"Login is visible": "passed"},
                "detected_issues": [{"issue": "Debug banner", "severity": "low"}],
            }
        )

        self.assertEqual(assessment.rule_verifications[0].status, "passed")
        self.assertEqual(assessment.detected_issues[0].issue_type, "qa_issue")
        self.assertEqual(assessment.detected_issues[0].description, "Debug banner")


class PlannerAgreementTests(unittest.TestCase):
    """QAPlanner subclasses DevicePlanner. Their action rules drifted apart once."""

    def _qa_prompt(self):
        return qa_prompt()

    def _device_prompt(self):
        from ronin.agent.plan.planner import DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        return planner._build_prompt(
            goal_lines=["Sign in"],
            gui_elements=None,
            history="",
            trail=None,
            drift_warning=None,
            loop_nudge=None,
        )

    def test_both_planners_carry_the_same_action_rules(self):
        from ronin.agent.plan.planner import DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        shared = planner._action_vocabulary() + planner._action_constraints()

        self.assertIn(shared, self._device_prompt())
        self.assertIn(shared, self._qa_prompt())

    def test_neither_planner_asks_for_a_focusing_tap_before_typing(self):
        for prompt in (self._device_prompt(), self._qa_prompt()):
            self.assertIn("do not spend a separate", prompt)
            self.assertNotIn("Step A: tap", prompt)


class RunnerWiringTests(unittest.TestCase):
    """A name used only inside `finally` is never exercised by importing the module."""

    def test_every_name_the_runner_calls_is_bound(self):
        import ronin.testing.runner as runner

        for name in ("unevidenced_security_rules", "aggregate_journey_status", "QAAgent"):
            self.assertTrue(hasattr(runner, name), f"{name} is not importable")


class VisionWeightsTests(unittest.TestCase):
    """The weights decide what the agent can see, so which weights must be fixed."""

    def test_the_detector_is_pinned_to_a_commit(self):
        from ronin.agent.perception.detector import YOLO_REVISION

        self.assertRegex(YOLO_REVISION, r"^[0-9a-f]{40}$")

    def test_setup_prefetches_exactly_what_a_run_loads(self):
        """Otherwise setup warms one set of weights and the run downloads another."""
        from ronin.agent.perception.detector import (
            YOLO_FILENAME,
            YOLO_REPO_ID,
            YOLO_REVISION,
        )
        from ronin.setup.steps import PREFETCH_SCRIPT

        for value in (YOLO_REPO_ID, YOLO_FILENAME, YOLO_REVISION):
            self.assertIn(repr(value), PREFETCH_SCRIPT)

    def test_the_reported_box_keeps_its_model_facing_meaning(self):
        """The model answers in this schema, so its description is prompt text."""
        from ronin.testing.models import QAIssue

        description = QAIssue.model_json_schema()["$defs"]["BoundingBox"]["description"]

        self.assertEqual(
            description, "Represents coordinates highlighting a visual issue."
        )


def _element(x_min, y_min, x_max, y_max, text=""):
    return SimpleNamespace(
        element="View",
        text=text,
        bounding_box=SimpleNamespace(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ),
    )


class _RecordingPlanner:
    def plan_step(self, **kwargs):
        self.kwargs = kwargs
        return "planned"


class KeyboardPlanningTests(unittest.TestCase):
    def test_keyboard_open_uses_current_reflowed_elements(self):
        agent = DeviceAgent.__new__(DeviceAgent)
        planner = _RecordingPlanner()
        current_field = _element(60, 816, 1014, 966, "Email Address")
        snapshot = SimpleNamespace(
            gui_elements=[current_field],
            marks=[],
            raw_img_bytes=b"image",
            planner_img_bytes=b"image",
        )
        context = RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            keyboard_open=True,
            screen_height=2400,
        )

        result = agent._plan_step(
            planner=planner,
            goal="Enter an email",
            snapshot=snapshot,
            ctx=context,
            session_trail=None,
            loop_nudge=None,
        )

        self.assertEqual(result, "planned")
        self.assertIs(planner.kwargs["gui_elements"][0], current_field)
        box = planner.kwargs["gui_elements"][0].bounding_box
        self.assertEqual(((box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2), (537, 891))


class ElementFormattingTests(unittest.TestCase):
    """The restaurant list that caused a one-row mis-tap in a real run."""

    def _planner(self):
        from ronin.agent.plan.planner import DevicePlanner

        return DevicePlanner.__new__(DevicePlanner)

    def _row(self, name, cuisine, y):
        return [
            _element(200, y - 30, 520, y + 2, name),
            _element(200, y + 28, 560, y + 60, cuisine),
            _element(820, y, 910, y + 60, "Menu"),
        ]

    def test_repeated_labels_carry_their_row(self):
        elements = self._row("Sweet Treats", "Desserts 0.5 km", 576) + self._row(
            "Sushi Zen", "Japanese 0.8 km", 805
        )

        lines = self._planner()._format_elements(elements).splitlines()
        menus = [line for line in lines if "'Menu'" in line]

        self.assertEqual(len(menus), 2)
        self.assertIn("Sweet Treats", menus[0])
        self.assertIn("Sushi Zen", menus[1])

    def test_unique_labels_are_left_alone(self):
        elements = [_element(0, 0, 100, 50, "Login")]

        self.assertNotIn("in row with", self._planner()._format_elements(elements))

    def test_duplicate_detections_collapse(self):
        elements = [
            _element(820, 805, 910, 865, "Menu"),
            _element(821, 806, 911, 866, "Menu"),
        ]

        self.assertEqual(len(self._planner()._format_elements(elements).splitlines()), 1)

    def test_elements_read_top_to_bottom(self):
        elements = [
            _element(0, 900, 100, 950, "bottom"),
            _element(0, 100, 100, 150, "top"),
            _element(500, 100, 600, 150, "top right"),
        ]

        lines = self._planner()._format_elements(elements).splitlines()

        self.assertIn("top'", lines[0])
        self.assertIn("top right", lines[1])
        self.assertIn("bottom", lines[2])


class StepBudgetTests(unittest.TestCase):
    def _agent(self, ceiling):
        from ronin.testing.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        agent._device_agent = SimpleNamespace(max_steps=ceiling)
        return agent

    def _case(self, instruction_count):
        return SimpleNamespace(steps_instructions=["do a thing"] * instruction_count)

    def test_short_scenarios_get_the_floor(self):
        self.assertEqual(self._agent(40)._step_budget(self._case(5)), 15)

    def test_long_scenarios_scale_with_their_instructions(self):
        """The 19-instruction journey that ran out of steps at 20."""
        self.assertEqual(self._agent(40)._step_budget(self._case(19)), 38)

    def test_the_configured_ceiling_always_wins(self):
        self.assertEqual(self._agent(10)._step_budget(self._case(19)), 10)
        self.assertEqual(self._agent(10)._step_budget(self._case(2)), 10)


class ScenarioGenerationTests(unittest.TestCase):
    def _generator(self, responses):
        from ronin.testing.generator import ScenarioGenerator

        client = SimpleNamespace(generate_response=Mock(side_effect=responses))
        return ScenarioGenerator(client), client

    def _journey(self):
        from ronin.testing.models import UserJourney

        return UserJourney(id="j", name="J", description="sign in", security_rules=[])

    def _suite(self, count):
        from ronin.testing.generator import GeneratedTestCasesList
        from ronin.testing.models import GeneratedTestCase

        return GeneratedTestCasesList(
            scenarios=[
                GeneratedTestCase(
                    test_id=f"TS_{n}",
                    scenario_name="S",
                    classification="happy_path",
                    steps_instructions=["open app"],
                    expected_outcomes=["the app opens"],
                )
                for n in range(count)
            ]
        )

    def test_the_schema_itself_forbids_an_empty_suite(self):
        """minItems reaches the model, so it is told an empty list is invalid."""
        from pydantic import ValidationError

        from ronin.testing.generator import GeneratedTestCasesList

        with self.assertRaises(ValidationError):
            GeneratedTestCasesList(scenarios=[])

        schema = GeneratedTestCasesList.model_json_schema()
        self.assertEqual(schema["properties"]["scenarios"]["minItems"], 1)
        self.assertEqual(schema["properties"]["scenarios"]["maxItems"], 10)

    def test_a_scenario_without_an_outcome_is_refused(self):
        """A scenario with no outcome is a schema mismatch, not a valid scenario.

        The gate judges a scenario by its outcomes, so one that declares none
        can only report as unverified. Refusing it here is what starts the
        retry: `LLMClient.generate_response` sends the validation error back,
        and the generator asks again.
        """
        from pydantic import ValidationError

        from ronin.testing.models import GeneratedTestCase

        with self.assertRaises(ValidationError):
            GeneratedTestCase(
                test_id="TS_1",
                scenario_name="S",
                classification="happy_path",
                steps_instructions=["open app"],
                expected_outcomes=[],
            )

        schema = GeneratedTestCase.model_json_schema()
        self.assertIn("expected_outcomes", schema["required"])
        self.assertEqual(schema["properties"]["expected_outcomes"]["minItems"], 1)

    def test_an_empty_response_is_retried(self):
        from pydantic import ValidationError

        rejected = ValidationError.from_exception_data("GeneratedTestCasesList", [])
        generator, client = self._generator([rejected, self._suite(2)])

        scenarios = generator.generate_scenarios(self._journey())

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(client.generate_response.call_count, 2)

    def test_repeated_empty_responses_abort_the_run(self):
        from pydantic import ValidationError

        rejected = ValidationError.from_exception_data("GeneratedTestCasesList", [])
        generator, client = self._generator([rejected] * 3)

        with self.assertRaisesRegex(RuntimeError, "Scenario generation failed"):
            generator.generate_scenarios(self._journey())

        self.assertEqual(client.generate_response.call_count, 3)


class StructuredRetryTests(unittest.TestCase):
    def _client(self):
        from ronin.llm.llm_client import LLMClient

        return LLMClient(model="test/model")

    def _model(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: int

        return Answer

    def test_a_schema_mismatch_is_corrected_once(self):
        from ronin.llm.messages import TextMessage

        client = self._client()
        answer = self._model()
        client._complete = Mock(
            side_effect=[
                __import__("pydantic").ValidationError.from_exception_data("Answer", []),
                answer(value=7),
            ]
        )

        result = client.generate_response([TextMessage(content="hi")], answer)

        self.assertEqual(result.value, 7)
        self.assertEqual(client._complete.call_count, 2)
        corrected = client._complete.call_args[0][0]
        self.assertIn("did not match the required JSON schema", corrected[-1].content)

    def test_a_second_mismatch_is_raised(self):
        from pydantic import ValidationError

        from ronin.llm.messages import TextMessage

        client = self._client()
        answer = self._model()
        client._complete = Mock(
            side_effect=ValidationError.from_exception_data("Answer", [])
        )

        with self.assertRaises(ValidationError):
            client.generate_response([TextMessage(content="hi")], answer)

        self.assertEqual(client._complete.call_count, 2)


class PlannerBoundaryTests(unittest.TestCase):
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
        from ronin.testing.models import ExpectedOutcome, GeneratedTestCase
        from ronin.testing.qa_planner import QAPlanner

        test_case = GeneratedTestCase(
            test_id="TS_001",
            scenario_name="Sign in",
            classification="happy_path",
            steps_instructions=["Enter the email", "Tap Login"],
            expected_outcomes=[ExpectedOutcome(statement=SENTINEL_OUTCOME)],
        )
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
        """The call site, not the prompt. QAAgent must pass setup, not the case."""
        import inspect

        from ronin.testing.qa_agent import QAAgent

        source = inspect.getsource(QAAgent.execute_scenario)
        construction = source[source.index("QAPlanner("):]
        construction = construction[: construction.index(")")]

        self.assertIn("setup=", construction)
        self.assertNotIn("test_case=", construction)
        self.assertNotIn("security_rules=", construction)

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
