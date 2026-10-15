import json
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


class VerdictTests(unittest.TestCase):
    def _agent(self):
        from ronin.testing.qa_agent import QAAgent

        return QAAgent.__new__(QAAgent)

    def _result(self, status="goal_reached", reason=None):
        return SimpleNamespace(status=status, failure_reason=reason)

    def _output(self, satisfied=True, evidence="Confirmation is visible"):
        return SimpleNamespace(
            is_scenario_feasible=True,
            infeasibility_reason=None,
            goal_satisfied=satisfied,
            goal_evidence=evidence,
        )

    def _verdict(self, **overrides):
        values = {
            "run_result": self._result(),
            "stop_reason": None,
            "all_verifications": [],
            "all_issues": [],
            "qa_outputs": [self._output()],
            "required_outcomes": [],
            "security_rules": [],
        }
        values.update(overrides)
        return self._agent()._determine_status(**values)

    def test_model_done_without_observable_evidence_is_not_a_pass(self):
        status, _reason = self._verdict(qa_outputs=[self._output(False, "")])

        self.assertEqual(status, "UNVERIFIED")

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


class ProviderNormalizationTests(unittest.TestCase):
    def test_compact_provider_shapes_are_normalized(self):
        from ronin.testing.qa_planner import QAStepResponse

        response = QAStepResponse.model_validate(
            {
                "current_screen": "Login",
                "observation": "Login form",
                "reasoning": "Focus email",
                "action": {
                    "type": "tap",
                    "coordinates": [10, 20],
                    "reason": "focus",
                },
                "rule_verifications": {"Login is visible": "passed"},
                "detected_issues": [
                    {"issue": "Debug banner", "severity": "low"}
                ],
            }
        )

        self.assertEqual(response.action.coordinates.x, 10)
        self.assertEqual(response.rule_verifications[0].status, "passed")
        self.assertEqual(response.detected_issues[0].issue_type, "qa_issue")
        self.assertEqual(response.detected_issues[0].description, "Debug banner")


class CapabilityTests(unittest.TestCase):
    def _case(self, text):
        from ronin.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Security",
            classification="security_compliance",
            steps_instructions=[text],
            expected_outcomes=[],
        )

    def test_network_inspection_is_rejected_before_execution(self):
        from ronin.testing.capabilities import unsupported_reason

        self.assertIn(
            "network traffic capture",
            unsupported_reason(self._case("Perform network inspection")),
        )

    def test_visual_scenario_remains_supported(self):
        from ronin.testing.capabilities import unsupported_reason

        self.assertIsNone(unsupported_reason(self._case("Verify the button is visible")))


class ActionSafetyTests(unittest.TestCase):
    def _control(self):
        return SimpleNamespace(
            screen_width=100,
            screen_height=200,
            driver=object(),
            platform=__import__(
                "ronin.agent.device.platform", fromlist=["Platform"]
            ).Platform.ANDROID,
            tap=Mock(),
        )

    def _tap(self, x, y):
        from ronin.agent.actuation.actions import Coordinates, TapAction

        return TapAction(coordinates=Coordinates(x=x, y=y), reason="tap")

    def test_out_of_bounds_action_is_rejected_before_dispatch(self):
        from ronin.agent.actuation.orchestrator import ActionOrchestrator

        control = self._control()
        with self.assertRaisesRegex(ValueError, "outside screen width"):
            ActionOrchestrator(control).execute(self._tap(100, 50))

        control.tap.assert_not_called()

    def test_device_failure_propagates(self):
        from ronin.agent.actuation.orchestrator import ActionOrchestrator

        control = self._control()
        control.tap.side_effect = RuntimeError("gesture failed")

        with self.assertRaisesRegex(RuntimeError, "gesture failed"):
            ActionOrchestrator(control).execute(self._tap(50, 50))


class ArtifactSafetyTests(unittest.TestCase):
    def _scenario(self):
        from ronin.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Checkout",
            classification="happy_path",
            steps_instructions=["Enter 'password123' in Password"],
            expected_outcomes=["Do not display '4111222233334444'"],
        )

    def test_progress_sanitizes_the_filename(self):
        from ronin.testing.models import UserJourney
        from ronin.testing.progress import ScenarioProgress
        from ronin.utils.secrets import SecretScrubber

        scenario = self._scenario()
        scrubber = SecretScrubber()
        journey = UserJourney(
            id="../../escape", name="Journey", description="test", security_rules=[]
        )
        with tempfile.TemporaryDirectory() as directory:
            progress = ScenarioProgress(
                Path(directory), journey, [scenario], scrubber=scrubber
            )
            progress.write()
            payload = progress.path.read_text()

            self.assertEqual(progress.path.parent, Path(directory))
            self.assertIn("TS_1", payload)
            self.assertFalse(progress.path.with_suffix(".json.tmp").exists())

    def test_progress_redacts_known_values(self):
        from ronin.testing.models import UserJourney
        from ronin.testing.progress import ScenarioProgress
        from ronin.utils.secrets import SecretScrubber

        scenario = self._scenario()
        scrubber = SecretScrubber()
        scrubber.learn_secrets(
            {"password": "password123", "card_number": "4111222233334444"}
        )
        journey = UserJourney(id="j", name="Journey", description="test", security_rules=[])
        with tempfile.TemporaryDirectory() as directory:
            progress = ScenarioProgress(
                Path(directory), journey, [scenario], scrubber=scrubber
            )
            progress.write()
            payload = progress.path.read_text()

            self.assertNotIn("password123", payload)
            self.assertNotIn("4111222233334444", payload)

    def test_junit_distinguishes_errors_from_assertion_failures(self):
        from ronin.testing.models import JourneyReport, ScenarioReport
        from ronin.testing.runner import RoninTestRunner

        report = JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status="FAILED",
            scenarios=[
                ScenarioReport(
                    test_id="one",
                    scenario_name="Harness",
                    classification="happy_path",
                    status="ERROR",
                    failure_reason="connection refused",
                ),
                ScenarioReport(
                    test_id="two",
                    scenario_name="Evidence",
                    classification="happy_path",
                    status="UNVERIFIED",
                    failure_reason="missing evidence",
                ),
            ],
            failed_count=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = RoninTestRunner.__new__(RoninTestRunner)
            runner.assets_dir = Path(directory)
            runner._save_junit_report(report)
            path = next(Path(directory).glob("junit_*.xml"))
            root = ET.parse(path).getroot()

            self.assertEqual(root.attrib["errors"], "1")
            self.assertEqual(root.attrib["failures"], "1")
            self.assertEqual(len(root.findall(".//error")), 1)
            self.assertEqual(len(root.findall(".//failure")), 1)

    def test_report_write_failure_makes_the_run_fail(self):
        from ronin.testing.models import JourneyReport
        from ronin.testing.runner import RoninTestRunner

        report = JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status="PASSED",
            passed_count=1,
        )

        RoninTestRunner._mark_report_write_failure(
            report, "junit_report", OSError("disk full")
        )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.failed_count, 1)
        self.assertEqual(report.scenarios[-1].status, "ERROR")

    def _element(self, text, box=(2, 3, 8, 9)):
        return SimpleNamespace(
            text=text,
            bounding_box=SimpleNamespace(
                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
            ),
        )

    def test_declared_secret_regions_are_covered_before_saving(self):
        from PIL import Image

        from ronin.agent.perception.media_manager import MediaManager
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"card_number": "4111222233334444"})
        image = Image.new("RGB", (20, 20), "white")

        safe = MediaManager(scrubber).redact_artifact_image(
            image, [self._element("4111 2222 3333 4444")]
        )
        self.assertEqual(safe.getpixel((4, 5)), (255, 255, 255))

        safe = MediaManager(scrubber).redact_artifact_image(
            image, [self._element("4111222233334444")]
        )
        self.assertEqual(safe.getpixel((4, 5)), (0, 0, 0))
        self.assertEqual(image.getpixel((4, 5)), (255, 255, 255))

    def test_ordinary_labels_are_never_covered(self):
        from PIL import Image

        from ronin.agent.perception.media_manager import MediaManager
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "secret-password123"})
        image = Image.new("RGB", (20, 20), "white")

        safe = MediaManager(scrubber).redact_artifact_image(
            image, [self._element("Continue")]
        )

        self.assertEqual(safe.getpixel((4, 5)), (255, 255, 255))

    def test_unknown_cost_is_reported_as_unknown(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(Path(directory))
            tracker.add_call("scenario:TS_1", "provider/model", None, usage)
            path = tracker.save_report()
            payload = json.loads(path.read_text())

        self.assertEqual(payload["cost_unknown_calls"], 1)
        call = payload["components"]["scenario:TS_1"]["calls"][0]
        self.assertFalse(call["cost_known"])
        self.assertEqual(call["model"], "provider/model")


class LifecycleOwnershipTests(unittest.TestCase):
    @patch("ronin.agent.device.device_session.urllib.request.urlopen")
    def test_existing_appium_service_is_not_claimed(self, urlopen):
        from ronin.agent.device.device_session import DeviceSession

        response = MagicMock()
        response.__enter__.return_value.status = 200
        urlopen.return_value = response
        session = DeviceSession.__new__(DeviceSession)
        session._appium_url = "http://127.0.0.1:4723"
        session._owns_appium = False
        session._appium_service = MagicMock()

        session._start_appium()

        session._appium_service.start.assert_not_called()
        self.assertFalse(session._owns_appium)


class ScenarioIsolationTests(unittest.TestCase):
    def test_supported_scenario_resets_perception_before_running(self):
        from ronin.agent.device_agent import DeviceRunResult
        from ronin.agent.history import AgentHistory
        from ronin.testing.models import GeneratedTestCase
        from ronin.testing.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        agent._use_multimodal_vision = False
        agent._fast_llm = object()
        agent._device_agent = SimpleNamespace(
            max_steps=40,
            replay_ledger=None,
            scrubber=__import__(
                "ronin.utils.secrets", fromlist=["SecretScrubber"]
            ).SecretScrubber(),
            run=Mock(
                return_value=DeviceRunResult(
                    status="step_limit",
                    steps_executed=1,
                    failure_reason="limit",
                    history=AgentHistory(),
                )
            ),
        )
        perception = SimpleNamespace(reset=Mock())
        case = GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Visible button",
            classification="happy_path",
            steps_instructions=["Verify the Login button is visible"],
            expected_outcomes=[],
        )

        report = agent.execute_scenario(
            case, [], object(), perception, Path("unused"), session_trail=object()
        )

        perception.reset.assert_called_once_with()
        self.assertEqual(report.status, "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()


class LaunchPaintTests(unittest.TestCase):
    """A Flutter debug build draws nothing for a second or two after am start."""

    def _device(self, png):
        return SimpleNamespace(get_screenshot=Mock(return_value=png))

    def _png(self, pixels):
        import io

        from PIL import Image

        image = Image.new("L", (10, 100))
        image.putdata(pixels)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_an_undrawn_screen_is_blank(self):
        from ronin.agent.launch import screen_is_blank

        self.assertTrue(screen_is_blank(self._device(self._png([255] * 1000))))

    def test_a_drawn_screen_is_not_blank(self):
        from ronin.agent.launch import screen_is_blank

        pixels = [0 if index % 2 else 255 for index in range(1000)]
        self.assertFalse(screen_is_blank(self._device(self._png(pixels))))

    def test_an_unreadable_screen_does_not_block_the_launch(self):
        from ronin.agent.launch import await_first_paint, screen_is_blank

        device = SimpleNamespace(get_screenshot=Mock(side_effect=RuntimeError("no screen")))

        self.assertIsNone(screen_is_blank(device))
        await_first_paint(device)

    def _drawn(self, seed=0):
        return self._png(
            [255 if (index + seed) % 3 else 0 for index in range(1000)]
        )

    def test_waiting_stops_once_a_drawn_screen_repeats(self):
        from ronin.agent.launch import await_first_paint

        blank = self._png([255] * 1000)
        device = self._device(blank)
        device.get_screenshot = Mock(
            side_effect=[blank, blank, self._drawn(), self._drawn()]
        )

        await_first_paint(device)

        self.assertEqual(device.get_screenshot.call_count, 4)

    def test_a_screen_that_never_settles_is_not_waited_out(self):
        from ronin.agent.launch import await_first_paint

        device = self._device(self._drawn())
        device.get_screenshot = Mock(
            side_effect=[self._drawn(seed=index) for index in range(60)]
        )

        started = time.monotonic()
        await_first_paint(device)
        elapsed = time.monotonic() - started

        from ronin.defaults import LAUNCH_CONTENT_GRACE, LAUNCH_PAINT_TIMEOUT

        self.assertLess(elapsed, LAUNCH_PAINT_TIMEOUT)
        self.assertGreaterEqual(elapsed, LAUNCH_CONTENT_GRACE)

    def test_a_splash_is_not_mistaken_for_the_first_screen(self):
        from ronin.agent.launch import await_first_paint

        splash, real = self._drawn(seed=0), self._drawn(seed=1)
        device = self._device(splash)
        device.get_screenshot = Mock(side_effect=[splash, real, real])

        await_first_paint(device)

        self.assertEqual(device.get_screenshot.call_count, 3)


class FirstLookTests(unittest.TestCase):
    """Verdicts read off a screen that may still be painting are withdrawn once."""

    def _response(self, crash=False, feasible=True):
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
        return SimpleNamespace(
            detected_issues=issues,
            is_scenario_feasible=feasible,
            action=SimpleNamespace(type="done", reason="stop"),
        )

    def test_a_crash_on_the_first_look_is_withdrawn(self):
        from ronin.testing.qa_agent import _defer_first_look

        response = self._response(crash=True)

        self.assertTrue(_defer_first_look(response, first_look=True))
        self.assertEqual(response.detected_issues, [])
        self.assertEqual(response.action.type, "wait")

    def test_a_crash_seen_later_stands(self):
        from ronin.testing.qa_agent import _defer_first_look

        response = self._response(crash=True)

        self.assertFalse(_defer_first_look(response, first_look=False))
        self.assertEqual(len(response.detected_issues), 1)

    def test_infeasibility_on_the_first_look_is_withdrawn(self):
        from ronin.testing.qa_agent import _defer_first_look

        response = self._response(feasible=False)

        self.assertTrue(_defer_first_look(response, first_look=True))
        self.assertTrue(response.is_scenario_feasible)
        self.assertEqual(response.action.type, "wait")

    def test_infeasibility_seen_later_stands(self):
        from ronin.testing.qa_agent import _defer_first_look

        response = self._response(feasible=False)

        self.assertFalse(_defer_first_look(response, first_look=False))
        self.assertFalse(response.is_scenario_feasible)

    def test_a_healthy_first_look_is_left_alone(self):
        from ronin.testing.qa_agent import _defer_first_look

        response = self._response()

        self.assertFalse(_defer_first_look(response, first_look=True))
        self.assertEqual(response.action.type, "done")

    def test_withdrawing_a_verdict_replaces_the_done_that_came_with_it(self):
        """Clearing the verdict alone changes the label; the loop still acts on `done`."""
        from ronin.agent.actuation.actions import ActionType
        from ronin.testing.qa_agent import _defer_first_look

        response = self._response(crash=True)

        _defer_first_look(response, first_look=True)

        self.assertEqual(response.action.type, ActionType.WAIT)
        self.assertGreater(response.action.duration, 0)


class JourneyEnvTests(unittest.TestCase):
    """A journey declares its values and says which are secret."""

    def _journey(self, data):
        from ronin.testing.cli import RoninCLI

        return RoninCLI.__new__(RoninCLI)._build_journey_object(data)

    def test_placeholders_are_replaced_in_the_journey_text(self):
        journey = self._journey(
            {
                "env": {"email": "qa@example.com"},
                "journey": "Type {{email}} in the Email field",
            }
        )

        self.assertEqual(journey.description, "Type qa@example.com in the Email field")

    def test_only_named_secrets_are_collected(self):
        journey = self._journey(
            {
                "env": {"email": "qa@example.com", "password": "hunter2"},
                "secrets": ["password"],
                "journey": "Sign in",
            }
        )

        self.assertEqual(journey.secrets, {"password": "hunter2"})

    def test_a_secret_naming_no_declared_value_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._journey({"env": {"email": "a@b.com"}, "secrets": ["password"]})

        self.assertIn("password", str(caught.exception))

    def test_values_can_come_from_the_environment(self):
        import os

        from ronin.testing.journey_env import resolve

        os.environ["RONIN_ENV_TEST"] = "from-the-shell"
        try:
            _, secrets = resolve(
                {"env": {"password": "${RONIN_ENV_TEST}"}, "secrets": ["password"]}
            )
        finally:
            del os.environ["RONIN_ENV_TEST"]

        self.assertEqual(secrets, {"password": "from-the-shell"})

    def test_security_rules_see_the_same_values(self):
        journey = self._journey(
            {
                "env": {"card": "4111222233334444"},
                "journey": "Pay",
                "security_rules": ["{{card}} must never be shown in full"],
            }
        )

        self.assertEqual(
            journey.security_rules, ["4111222233334444 must never be shown in full"]
        )

    def test_a_journey_without_an_env_block_still_loads(self):
        journey = self._journey({"journey": "Sign in", "security_rules": ["No leaks"]})

        self.assertEqual(journey.secrets, {})
        self.assertEqual(journey.description, "Sign in")


class PlannerAgreementTests(unittest.TestCase):
    """QAPlanner subclasses DevicePlanner; their action rules drifted apart once."""

    def _qa_prompt(self):
        from ronin.testing.qa_planner import QAPlanner

        planner = QAPlanner.__new__(QAPlanner)
        planner.use_multimodal_vision = False
        planner._test_case = SimpleNamespace(expected_outcomes=["Order confirmed"])
        planner._security_rules = ["Card is masked"]
        return planner._build_qa_prompt(
            goal_lines=["Sign in"],
            gui_elements=None,
            history="",
            trail=None,
            drift_warning=None,
            loop_nudge=None,
        )

    def _device_prompt(self):
        from ronin.agent.device_planner import DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        planner.use_multimodal_vision = False
        return planner._build_prompt(
            goal_lines=["Sign in"],
            gui_elements=None,
            history="",
            trail=None,
            drift_warning=None,
            loop_nudge=None,
        )

    def test_both_planners_carry_the_same_action_rules(self):
        from ronin.agent.device_planner import DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        shared = planner._action_vocabulary() + planner._action_constraints()

        self.assertIn(shared, self._device_prompt())
        self.assertIn(shared, self._qa_prompt())

    def test_neither_planner_asks_for_a_focusing_tap_before_typing(self):
        for prompt in (self._device_prompt(), self._qa_prompt()):
            self.assertIn("do not spend a separate", prompt)
            self.assertNotIn("Step A: tap", prompt)


class InlineSecretTests(unittest.TestCase):
    """A value can say it is secret where it is declared."""

    def _resolve(self, data):
        from ronin.testing.journey_env import resolve

        return resolve(data)

    def test_a_value_can_mark_itself_secret(self):
        values, secrets = self._resolve(
            {
                "env": {
                    "email": "qa@example.com",
                    "password": {"value": "hunter2", "secret": True},
                }
            }
        )

        self.assertEqual(values["password"], "hunter2")
        self.assertEqual(secrets, {"password": "hunter2"})

    def test_an_inline_value_defaults_to_ordinary(self):
        _, secrets = self._resolve({"env": {"note": {"value": "visible"}}})

        self.assertEqual(secrets, {})

    def test_inline_and_listed_secrets_combine(self):
        _, secrets = self._resolve(
            {
                "env": {
                    "password": {"value": "hunter2", "secret": True},
                    "cvv": "321",
                },
                "secrets": ["cvv"],
            }
        )

        self.assertEqual(secrets, {"cvv": "321", "password": "hunter2"})

    def test_a_mapping_without_a_value_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._resolve({"env": {"password": {"secret": True}}})

        self.assertIn("value", str(caught.exception))

    def test_a_misspelled_key_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._resolve({"env": {"password": {"value": "x", "secrets": True}}})

        self.assertIn("secrets", str(caught.exception))


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
                SimpleNamespace(
                    is_scenario_feasible=True,
                    infeasibility_reason=None,
                    goal_satisfied=True,
                    goal_evidence="Confirmation is visible",
                )
            ],
            "required_outcomes": [],
            "security_rules": [],
        }
        values.update(overrides)
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


class SecurityCoverageTests(unittest.TestCase):
    """Scoping rules per scenario leaves a hole; the suite has to close it."""

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


class RunnerWiringTests(unittest.TestCase):
    """A name used only inside `finally` is never exercised by importing the module."""

    def test_every_name_the_runner_calls_is_bound(self):
        import ronin.testing.runner as runner

        for name in ("unevidenced_security_rules", "aggregate_journey_status", "QAAgent"):
            self.assertTrue(hasattr(runner, name), f"{name} is not importable")


class SecurityScopingTests(VerificationAggregationTests):
    """The planner reports every rule on every screen; being mentioned proves nothing."""

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


class ScanDumpSafetyTests(unittest.TestCase):
    """A scan dump is a file, so detected text in it outlives the run."""

    def _element(self, text):
        from ronin.agent.perception.image_descriptor import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="text",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
        )

    def test_declared_secrets_are_masked_in_the_dump(self):
        from ronin.agent.perception.scan_dump import element_to_dict
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "secret-password123"})

        dumped = element_to_dict(self._element("secret-password123"), 0, scrubber)

        self.assertEqual(dumped["text"], "{{password}}")

    def test_ordinary_detected_text_survives(self):
        from ronin.agent.perception.scan_dump import element_to_dict
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "secret-password123"})

        dumped = element_to_dict(self._element("Continue"), 0, scrubber)

        self.assertEqual(dumped["text"], "Continue")


class MissingEnvTests(unittest.TestCase):
    """A blank password is a configuration error, not a test of blank passwords."""

    def test_an_unset_variable_fails_the_journey(self):
        import os

        from ronin.testing.journey_env import resolve

        os.environ.pop("RONIN_DEFINITELY_UNSET", None)

        with self.assertRaises(ValueError) as caught:
            resolve({"env": {"password": "${RONIN_DEFINITELY_UNSET}"}})

        self.assertIn("RONIN_DEFINITELY_UNSET", str(caught.exception))


class CapabilityMatchingTests(unittest.TestCase):
    """Seeing a contrasting element is observable; measuring a ratio is not."""

    def _reason(self, *instructions):
        from ronin.testing.capabilities import unsupported_reason

        return unsupported_reason(
            SimpleNamespace(steps_instructions=list(instructions), expected_outcomes=[])
        )

    def test_a_visible_high_contrast_element_is_supported(self):
        self.assertIsNone(self._reason("The high-contrast error banner is visible"))

    def test_a_measured_contrast_ratio_is_unsupported(self):
        self.assertIn("contrast", self._reason("The contrast ratio is at least 4.5:1"))

    def test_a_wcag_assertion_is_unsupported(self):
        self.assertIn("contrast", self._reason("Text meets WCAG AA"))
