import io
import json
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


from ronin.errors import ObservationUnavailable


def _solid_png(fill: str, size=(200, 400)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, fill).save(buffer, format="PNG")
    return buffer.getvalue()


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

    def test_the_report_records_what_produced_it(self):
        """A run that behaves differently must be attributable to what changed."""
        from ronin.testing.models import JourneyReport, RunEnvironment
        from ronin.testing.reports import save_markdown_report

        report = JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status="PASSED",
            passed_count=1,
            environment=RunEnvironment(
                ronin_version="0.1.0",
                model="openrouter/some/model",
                fast_model="openrouter/some/fast",
                platform="android",
                grounding="marks",
                detector="orasul/deki-yolo@287f12d1",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_markdown_report(report, Path(directory))
            payload = path.read_text()

        self.assertIn("openrouter/some/model", payload)
        self.assertIn("openrouter/some/fast", payload)
        self.assertIn("orasul/deki-yolo@287f12d1", payload)
        self.assertIn("marks", payload)

    def test_a_report_without_an_environment_still_writes(self):
        from ronin.testing.models import JourneyReport
        from ronin.testing.reports import save_markdown_report

        report = JourneyReport(
            journey_id="j", journey_name="Journey", description="d", status="PASSED"
        )

        with tempfile.TemporaryDirectory() as directory:
            payload = save_markdown_report(report, Path(directory)).read_text()

        self.assertIn("Journey", payload)
        self.assertNotIn("**Detector**", payload)

    def test_the_runner_records_the_pinned_detector(self):
        from ronin.agent.device.platform import Platform
        from ronin.agent.perception.image_descriptor import YOLO_REVISION
        from ronin.testing.runner import RoninTestRunner

        runner = RoninTestRunner(
            device_name="emulator-5554",
            app_path="app.apk",
            platform=Platform.ANDROID,
            model="openrouter/some/model",
            grounding="marks",
        )

        environment = runner._environment()

        self.assertEqual(environment.platform, "android")
        self.assertIn(YOLO_REVISION, environment.detector)

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

        device = SimpleNamespace(
            get_screenshot=Mock(side_effect=ObservationUnavailable("no screen"))
        )

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
        from ronin.agent.perception.elements import BoundingBox, GUIElement

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


class ConfigFileTests(unittest.TestCase):
    """A setting that does nothing must not look like one that was applied."""

    def _write(self, text):
        path = Path(tempfile.mkdtemp()) / "ronin.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_misspelled_setting_is_named_and_the_rest_still_load(self):
        from ronin.config import read_file

        path = self._write("platfrom: ios\nmodel: a/model\n")

        with self.assertLogs("ronin.config", level="WARNING") as captured:
            settings = read_file(path)

        self.assertEqual(settings, {"model": "a/model"})
        self.assertIn("platfrom", captured.output[0])

    def test_a_file_of_known_settings_warns_about_nothing(self):
        from ronin.config import read_file

        path = self._write("platform: ios\nmax_steps: 7\n")

        with self.assertNoLogs("ronin.config", level="WARNING"):
            settings = read_file(path)

        self.assertEqual(settings, {"platform": "ios", "max_steps": 7})


class StalenessTests(unittest.TestCase):
    """Seconds pass between reading the screen and tapping it."""

    def _png(self, fill, patch=None):
        import io

        from PIL import Image

        image = Image.new("RGB", (200, 400), fill)
        if patch:
            box, colour = patch
            for x in range(box[0], box[2]):
                for y in range(box[1], box[3]):
                    image.putpixel((x, y), colour)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _with_patch(self, png, box, colour):
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                image.putpixel((x, y), colour)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _element(self, box=(10, 10, 60, 60)):
        from ronin.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Button",
            type="yolo_detection",
            confidence=0.9,
            text="Continue",
            bounding_box=BoundingBox(
                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
            ),
        )

    def _tap(self, x, y):
        from ronin.agent.actuation.actions import TapAction, Coordinates

        return TapAction(coordinates=Coordinates(x=x, y=y), reason="tap it")

    def test_an_unchanged_target_is_not_stale(self):
        from ronin.agent.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))

        self.assertFalse(
            target_moved(before, before, self._tap(30, 30), [self._element()])
        )

    def test_a_target_that_moved_is_stale(self):
        from ronin.agent.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._png("white")

        self.assertTrue(
            target_moved(before, after, self._tap(30, 30), [self._element()])
        )

    def test_change_elsewhere_on_screen_is_ignored(self):
        """A banner landing in the corner must not throw away a good plan."""
        from ronin.agent.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._with_patch(after, (100, 300, 190, 390), (255, 0, 0))

        self.assertFalse(
            target_moved(before, after, self._tap(30, 30), [self._element()])
        )

    def test_a_point_outside_every_element_still_gets_checked(self):
        from ronin.agent.staleness import target_moved

        before = self._png("white", ((100, 300, 190, 390), (0, 0, 0)))
        after = self._png("white")

        self.assertTrue(target_moved(before, after, self._tap(145, 345), []))

    def test_an_action_with_no_target_is_never_stale(self):
        from ronin.agent.actuation.actions import KeyEventAction
        from ronin.agent.staleness import target_moved

        before = self._png("white", ((10, 10, 60, 60), (0, 0, 0)))
        after = self._png("black")

        self.assertFalse(
            target_moved(before, after, KeyEventAction(key=4, reason="back"), [])
        )

    def test_an_unreadable_screenshot_does_not_strand_the_run(self):
        from ronin.agent.staleness import target_moved

        self.assertFalse(
            target_moved(b"not a png", b"nor this", self._tap(30, 30), [])
        )


class ExitCodeTests(unittest.TestCase):
    """Only the entry point ends the process; everything below answers with a code."""

    def test_a_usage_problem_is_reported_once_and_exits_two(self):
        from ronin.cli.main import main
        from ronin.errors import UsageError

        with (
            patch("ronin.cli.main.load_config", side_effect=UsageError("no app binary")),
            patch("sys.stderr", new=io.StringIO()) as captured,
        ):
            code = main(["doctor"])

        self.assertEqual(code, 2)
        self.assertIn("no app binary", captured.getvalue())

    def test_an_unloadable_suite_answers_two_without_exiting(self):
        from ronin.testing.cli import RoninCLI

        cli = RoninCLI()

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.yaml"
            code = cli.run([str(missing)])

        self.assertEqual(code, 2)

    def test_no_journey_file_is_a_usage_error_not_an_exit(self):
        from ronin.errors import UsageError
        from ronin.testing.cli import RoninCLI

        cli = RoninCLI()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("ronin.testing.cli.Path.cwd", return_value=Path(directory)),
            self.assertRaises(UsageError),
        ):
            cli._resolve_journey_file(None)

    def test_a_setting_ronin_config_does_not_know_is_not_looked_up_there(self):
        """grounding and mark_detail come from a flag or the journey, or default."""
        from ronin.agent.device_planner import GROUNDING_COORDINATES
        from ronin.agent.perception.marks import DETAIL_NUMBERS
        from ronin.testing.cli import RoninCLI
        from ronin.testing.models import JourneyReport

        cli = RoninCLI()
        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )

        with (
            patch.object(
                RoninCLI,
                "load_journeys",
                return_value=[SimpleNamespace(id="j", name="J")],
            ),
            patch.object(RoninCLI, "_resolve_journey_file", return_value=Path("j.yaml")),
            patch("ronin.testing.cli.resolve_app_path", return_value=Path("app.apk")),
            patch("ronin.testing.cli.RoninTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            cli.run(["j.yaml"])

        self.assertEqual(
            runner_class.call_args.kwargs["grounding"], GROUNDING_COORDINATES
        )
        self.assertEqual(runner_class.call_args.kwargs["mark_detail"], DETAIL_NUMBERS)

    def test_the_suite_result_decides_the_code(self):
        from ronin.testing.cli import RoninCLI
        from ronin.testing.models import JourneyReport

        def report(status):
            return JourneyReport(
                journey_id="j", journey_name="J", description="d", status=status
            )

        for status, expected in (("PASSED", 0), ("FAILED", 1)):
            cli = RoninCLI()
            with (
                patch.object(
                RoninCLI,
                "load_journeys",
                return_value=[SimpleNamespace(id="j", name="J")],
            ),
                patch.object(RoninCLI, "_resolve_journey_file", return_value=Path("j.yaml")),
                patch("ronin.testing.cli.resolve_app_path", return_value=Path("app.apk")),
                patch("ronin.testing.cli.RoninTestRunner") as runner_class,
            ):
                runner_class.return_value.run_journey.return_value = report(status)
                self.assertEqual(cli.run(["j.yaml"]), expected)


class DeviceFailureTests(unittest.TestCase):
    """A failed device instruction has to say what it was, and keep its cause."""

    def _failing_adb(self):
        from ronin.utils.subprocess_runner import SubprocessRunError

        return SubprocessRunError("adb failed", 1, "", "device offline")

    def test_a_failed_command_names_the_intent_and_keeps_the_cause(self):
        from ronin.agent.device.adb_control import AdbControl
        from ronin.errors import DeviceUnavailable
        from ronin.utils.subprocess_runner import SubprocessRunError

        cause = self._failing_adb()
        with (
            patch("ronin.agent.device.adb_control.run_command", side_effect=cause),
            self.assertRaises(DeviceUnavailable) as raised,
        ):
            AdbControl.set_airplane_mode("emulator-5554", True)

        self.assertIn("set airplane mode", str(raised.exception))
        self.assertIn("airplane-mode enable", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, SubprocessRunError)
        self.assertEqual(raised.exception.__cause__.stderr, "device offline")

    def test_a_nested_call_reports_the_instruction_that_failed(self):
        """Not the outermost wrapper: the step that could not be carried out."""
        from ronin.agent.device.adb_control import AdbControl
        from ronin.errors import DeviceUnavailable

        with (
            patch(
                "ronin.agent.device.adb_control.run_command",
                side_effect=self._failing_adb(),
            ),
            self.assertRaises(DeviceUnavailable) as raised,
        ):
            AdbControl.enable_feedback("emulator-5554", True, True)

        self.assertIn("toggle pointer location", str(raised.exception))

    def test_an_unreadable_screen_is_reported_as_such(self):
        from ronin.agent.device.device_control import DeviceControl
        from ronin.errors import ObservationUnavailable

        control = DeviceControl.__new__(DeviceControl)
        control.device_name = "emulator-5554"
        control.driver = SimpleNamespace(
            get_screenshot_as_png=Mock(side_effect=OSError("connection reset"))
        )

        with self.assertRaises(ObservationUnavailable) as raised:
            control.get_screenshot()

        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_a_report_that_cannot_be_written_says_so(self):
        from ronin.errors import ArtifactWriteFailed
        from ronin.testing.models import JourneyReport
        from ronin.testing.reports import save_markdown_report

        report = JourneyReport(
            journey_id="j", journey_name="Journey", description="d", status="PASSED"
        )

        with self.assertRaises(ArtifactWriteFailed):
            save_markdown_report(report, Path("/nonexistent-directory-for-ronin"))


class VisionWeightsTests(unittest.TestCase):
    """The weights decide what the agent can see, so which weights must be fixed."""

    def test_the_detector_is_pinned_to_a_commit(self):
        from ronin.agent.perception.image_descriptor import YOLO_REVISION

        self.assertRegex(YOLO_REVISION, r"^[0-9a-f]{40}$")

    def test_setup_prefetches_exactly_what_a_run_loads(self):
        """Otherwise setup warms one set of weights and the run downloads another."""
        from ronin.agent.perception.image_descriptor import (
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


class MarkTests(unittest.TestCase):
    """The number drawn on the screenshot and the number in the prompt must agree."""

    def _el(self, name, text, box):
        from ronin.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element=name,
            type="yolo_detection",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(
                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
            ),
        )

    def _screen(self):
        return [
            self._el("Button", "Continue", (480, 2180, 600, 2220)),
            self._el("Text", "Credit / Debit Card", (380, 660, 500, 700)),
            self._el("View", "", (500, 2190, 560, 2210)),
        ]

    def test_numbers_run_in_reading_order(self):
        from ronin.agent.perception.marks import build

        numbered = [(m.index, m.element.text) for m in build(self._screen())]

        self.assertEqual(numbered[0], (1, "Credit / Debit Card"))
        self.assertEqual(numbered[-1][1], "Continue")

    def test_repeat_detections_get_one_number(self):
        from ronin.agent.perception.marks import build

        doubled = self._screen() + [
            self._el("Text", "Credit / Debit Card", (381, 661, 501, 701))
        ]

        self.assertEqual(len(build(doubled)), len(build(self._screen())))

    def test_the_offered_list_carries_no_coordinates(self):
        from ronin.agent.perception.marks import DETAIL_TEXT, build, describe

        listing = describe(build(self._screen()), height=2400, detail=DETAIL_TEXT)

        self.assertIn("'Continue'", listing)
        self.assertNotIn("2201", listing)
        self.assertNotIn("center", listing)
        self.assertIn("lower", listing)

    def test_detected_text_is_left_out_by_default(self):
        """The model is looking at the screen those words are printed on."""
        from ronin.agent.perception.marks import build, describe

        listing = describe(build(self._screen()), height=2400)

        self.assertNotIn("Continue", listing)
        self.assertIn("1 |", listing)
        self.assertIn("lower", listing)

    def test_the_numbers_are_always_listed(self):
        """They bound what the model may answer, whatever else is left out."""
        from ronin.agent.perception.marks import DETAIL_NUMBERS, build, describe

        marks = build(self._screen())
        listing = describe(marks, height=2400, detail=DETAIL_NUMBERS)

        for mark in marks:
            self.assertIn(f"{mark.index} |", listing)

    def test_the_picture_can_be_sent_alone(self):
        from ronin.agent.perception.marks import DETAIL_NONE, build, describe

        self.assertEqual(describe(build(self._screen()), 2400, DETAIL_NONE), "")

    def test_a_number_resolves_to_its_element(self):
        from ronin.agent.perception.marks import build, find

        marks = build(self._screen())
        continue_mark = next(m for m in marks if m.element.text == "Continue")

        self.assertEqual(find(marks, continue_mark.index).center, (540, 2200))
        self.assertIsNone(find(marks, 99))

    def test_marks_mode_never_offers_coordinates_to_the_model(self):
        from ronin.agent.device_planner import GROUNDING_MARKS, DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        planner.grounding = GROUNDING_MARKS
        planner.use_multimodal_vision = False

        vocabulary = planner._action_vocabulary()

        self.assertIn("`mark` (int)", vocabulary)
        self.assertIn("target_not_listed", vocabulary)
        self.assertNotIn('"x": int', vocabulary)


class TargetDecisionTests(unittest.TestCase):
    """Nothing guesses: an unusable target stops the step and is recorded."""

    def _snapshot(self, marks=(), png=None):
        return SimpleNamespace(
            marks=list(marks),
            gui_elements=[],
            raw_img_bytes=_solid_png("white") if png is None else png,
            current_phash="p",
        )

    def _device(self, png=None):
        """Re-reads the same screen the snapshot was taken from, so nothing is stale."""
        return SimpleNamespace(
            get_screenshot=Mock(
                return_value=_solid_png("white") if png is None else png
            )
        )

    def _marks(self):
        from ronin.agent.perception.elements import BoundingBox, GUIElement
        from ronin.agent.perception.marks import Mark

        element = GUIElement(
            element="Button",
            type="yolo_detection",
            confidence=0.9,
            text="Continue",
            bounding_box=BoundingBox(x_min=10, y_min=20, x_max=30, y_max=40),
        )
        return [Mark(index=1, element=element)]

    def test_a_valid_number_becomes_the_point_to_touch(self):
        """The number is swapped for its coordinate twin — every mode ends in a point."""
        from ronin.agent.actuation.actions import TapAction, TapMarkAction
        from ronin.agent.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=1, reason="continue"),
            self._snapshot(self._marks()),
            self._device(),
        )

        self.assertTrue(decision.ok)
        self.assertIsInstance(decision.action, TapAction)
        self.assertEqual(
            (decision.action.coordinates.x, decision.action.coordinates.y), (20, 30)
        )

    def test_a_number_that_names_nothing_stops_the_step(self):
        from ronin.agent.actuation.actions import TapMarkAction
        from ronin.agent.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=47, reason="continue"),
            self._snapshot(self._marks()),
            self._device(),
        )

        self.assertFalse(decision.ok)
        self.assertIn("no element 47", decision.observation)
        self.assertIsNone(decision.stop_scenario)

    def test_an_undetected_target_ends_the_scenario_and_is_kept(self):
        from ronin.agent.actuation.actions import TargetNotListedAction
        from ronin.agent.targeting import evaluate

        decision = evaluate(
            TargetNotListedAction(description="blue button at the bottom", reason="r"),
            self._snapshot(self._marks()),
            self._device(),
        )

        self.assertFalse(decision.ok)
        self.assertTrue(decision.unmatched)
        self.assertIn("blue button at the bottom", decision.stop_scenario)

    def test_coordinates_still_work_when_marks_are_not_in_use(self):
        from ronin.agent.actuation.actions import Coordinates, TapAction
        from ronin.agent.targeting import evaluate

        action = TapAction(coordinates=Coordinates(x=5, y=6), reason="tap")

        self.assertTrue(evaluate(action, self._snapshot(), self._device()).ok)

    def test_an_untargeted_action_does_not_re_read_the_screen(self):
        """A key event is aimed at no point, so there is nothing to re-check."""
        from ronin.agent.actuation.actions import KeyEventAction
        from ronin.agent.targeting import evaluate

        device = self._device()

        decision = evaluate(
            KeyEventAction(key=4, reason="back"), self._snapshot(), device
        )

        self.assertTrue(decision.ok)
        device.get_screenshot.assert_not_called()

    def test_a_target_that_moved_stops_the_step(self):
        """The number resolved to a point, but the screen moved under it."""
        from ronin.agent.actuation.actions import TapMarkAction
        from ronin.agent.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=1, reason="continue"),
            self._snapshot(self._marks()),
            self._device(png=_solid_png("black")),
        )

        self.assertFalse(decision.ok)
        self.assertIn("screen changed", decision.observation)

    def test_a_screen_that_cannot_be_read_again_stops_the_step(self):
        """No current screen is no evidence that the target survived."""
        from ronin.agent.actuation.actions import TapMarkAction
        from ronin.agent.targeting import evaluate

        device = SimpleNamespace(
            get_screenshot=Mock(side_effect=ObservationUnavailable("session gone"))
        )

        decision = evaluate(
            TapMarkAction(mark=1, reason="continue"),
            self._snapshot(self._marks()),
            device,
        )

        self.assertFalse(decision.ok)
        self.assertIn("could not be read", decision.observation)

    def test_an_empty_screenshot_stops_the_step(self):
        from ronin.agent.actuation.actions import TapMarkAction
        from ronin.agent.targeting import evaluate

        decision = evaluate(
            TapMarkAction(mark=1, reason="continue"),
            self._snapshot(self._marks()),
            self._device(png=b""),
        )

        self.assertFalse(decision.ok)
        self.assertIn("could not be read", decision.observation)

    def test_resolving_a_number_leaves_the_proposed_action_untouched(self):
        """A proposed action is parsed once, never edited into a resolved one."""
        from ronin.agent.actuation.actions import TapMarkAction
        from ronin.agent.targeting import evaluate

        proposed = TapMarkAction(mark=1, reason="continue")

        decision = evaluate(proposed, self._snapshot(self._marks()), self._device())

        self.assertIsNot(decision.action, proposed)
        self.assertFalse(hasattr(proposed, "coordinates"))
        self.assertEqual(proposed.mark, 1)


class MarkLegibilityTests(unittest.TestCase):
    """A number the model cannot read is worse than no number at all."""

    def test_the_badge_font_scales_with_the_screenshot(self):
        from ronin.agent.perception.marks import _font

        small, large = _font(800), _font(2400)

        self.assertGreater(getattr(large, "size", 0), getattr(small, "size", 0))
        self.assertGreaterEqual(getattr(large, "size", 0), 30)

    def test_a_missing_font_file_does_not_crash_rendering(self):
        from PIL import Image

        from ronin.agent.perception import marks

        with patch.object(marks, "FONT_CANDIDATES", ("no-such-font.ttf",)):
            rendered = marks.render(Image.new("RGB", (100, 200), "white"), [])

        self.assertEqual(rendered.size, (100, 200))


class KeyboardMarkTests(unittest.TestCase):
    """An open keyboard is most of the screen; numbering it buries the app."""

    def _el(self, text, y):
        from ronin.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="yolo_detection",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(x_min=10, y_min=y, x_max=60, y_max=y + 40),
        )

    def _screen(self):
        return [
            self._el("Email Address", 800),
            self._el("Login", 1300),
            *[self._el(letter, 1800) for letter in "qwertyuiop"],
            self._el("?!", 2200),
        ]

    def test_keys_are_not_numbered_while_the_keyboard_is_open(self):
        from ronin.agent.perception.marks import build, without_keyboard_keys

        kept = without_keyboard_keys(self._screen(), keyboard_open=True, screen_height=2400)
        texts = [m.element.text for m in build(kept)]

        self.assertEqual(texts, ["Email Address", "Login"])

    def test_everything_is_numbered_when_the_keyboard_is_closed(self):
        from ronin.agent.perception.marks import build, without_keyboard_keys

        kept = without_keyboard_keys(self._screen(), keyboard_open=False, screen_height=2400)

        self.assertEqual(len(build(kept)), len(self._screen()))

    def test_a_long_label_low_on_screen_is_not_mistaken_for_a_key(self):
        from ronin.agent.perception.marks import without_keyboard_keys

        elements = [self._el("Place Order", 2200)]

        self.assertEqual(
            len(without_keyboard_keys(elements, keyboard_open=True, screen_height=2400)), 1
        )


class UnresolvedMarkTests(unittest.TestCase):
    """A number naming nothing must be reportable, not fatal."""

    def _action(self):
        from ronin.agent.actuation.actions import TapMarkAction

        return TapMarkAction(mark=47, reason="continue")

    def test_an_unresolved_mark_can_be_described(self):
        from ronin.agent.device_agent import DeviceAgent

        described = DeviceAgent.__new__(DeviceAgent)._describe_action(self._action())

        self.assertIn("element 47", described)

    def test_an_unresolved_mark_can_be_hashed_for_loop_detection(self):
        from ronin.agent.actuation.actions import TapMarkAction
        from ronin.agent.loop_detector import RollingLoopDetector

        same = RollingLoopDetector.hash_action(self._action())
        other = RollingLoopDetector.hash_action(TapMarkAction(mark=48, reason="continue"))

        self.assertEqual(same, RollingLoopDetector.hash_action(self._action()))
        self.assertNotEqual(same, other)

    def test_a_not_listed_action_can_be_described_and_hashed(self):
        from ronin.agent.actuation.actions import TargetNotListedAction
        from ronin.agent.device_agent import DeviceAgent
        from ronin.agent.loop_detector import RollingLoopDetector

        action = TargetNotListedAction(description="blue button", reason="r")

        self.assertIn("target_not_listed", DeviceAgent.__new__(DeviceAgent)._describe_action(action))
        self.assertTrue(RollingLoopDetector.hash_action(action))


class NestCollapseTests(unittest.TestCase):
    """A field is a container holding an icon and a value — one control, not three."""

    def _el(self, kind, text, box):
        from ronin.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element=kind,
            type="yolo_detection",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(
                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
            ),
        )

    def _field(self):
        return [
            self._el("View", "", (60, 800, 1020, 970)),  # the field container
            self._el("ImageView", "", (80, 860, 140, 920)),  # the icon
            self._el("Text", "qa@example.com", (160, 860, 700, 910)),  # the value
        ]

    def test_a_field_becomes_one_mark(self):
        from ronin.agent.perception.marks import build

        marks = build(self._field(), screen_area=1080 * 2400)

        self.assertEqual(len(marks), 1)

    def test_the_collapsed_mark_keeps_the_name(self):
        from ronin.agent.perception.marks import build

        self.assertEqual(
            build(self._field(), screen_area=1080 * 2400)[0].element.text,
            "qa@example.com",
        )

    def test_the_collapsed_mark_is_touched_where_its_label_is(self):
        from ronin.agent.perception.marks import build

        mark = build(self._field(), screen_area=1080 * 2400)[0]

        self.assertEqual(mark.center, (430, 885))

    def test_a_layout_region_keeps_its_controls_separate(self):
        from ronin.agent.perception.marks import build

        page = [self._el("View", "", (0, 0, 1080, 2400)), *self._field()]

        self.assertGreater(len(build(page, screen_area=1080 * 2400)), 1)


class KeyboardBandTests(unittest.TestCase):
    """The strip with no numbers has to say why it has no numbers."""

    def _key(self, text, y):
        from ronin.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="yolo_detection",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(x_min=10, y_min=y, x_max=60, y_max=y + 40),
        )

    def test_the_band_spans_the_keys(self):
        from ronin.agent.perception.marks import keyboard_band

        band = keyboard_band([self._key("q", 1800), self._key("m", 2200)], 1080)

        self.assertEqual(band, (0, 1800, 1080, 2240))

    def test_there_is_no_band_without_a_keyboard(self):
        from ronin.agent.perception.marks import keyboard_band

        self.assertIsNone(keyboard_band([], 1080))

    def test_a_wide_key_inside_the_band_is_absorbed(self):
        """"?123" and "space" are too long for the label rule but sit among the keys."""
        from ronin.agent.perception.marks import split_keyboard

        elements = [
            self._key("q", 1700),
            self._key("?123", 1950),
            self._key("m", 2200),
        ]

        kept, keys = split_keyboard(elements, keyboard_open=True, screen_height=2400)

        self.assertEqual(kept, [])
        self.assertEqual(len(keys), 3)

    def test_an_app_control_above_the_keyboard_survives(self):
        from ronin.agent.perception.marks import split_keyboard

        elements = [self._key("Place Order", 1400), self._key("q", 1700), self._key("m", 2200)]

        kept, keys = split_keyboard(elements, keyboard_open=True, screen_height=2400)

        self.assertEqual([e.text for e in kept], ["Place Order"])
        self.assertEqual(len(keys), 2)


class GroundingContractTests(unittest.TestCase):
    """Each mode accepts one way of naming a target, and refuses the other."""

    def _snapshot(self):
        return SimpleNamespace(
            marks=[], gui_elements=[], raw_img_bytes=b"", current_phash="p"
        )

    def _device(self):
        return SimpleNamespace(get_screenshot=Mock(return_value=b""))

    def _verdict(self, action, grounding):
        from ronin.agent.targeting import evaluate

        return evaluate(action, self._snapshot(), self._device(), grounding)

    def test_marks_mode_refuses_raw_coordinates(self):
        """Obeying them would reinstate the invented coordinate numbering prevents."""
        from ronin.agent.actuation.actions import Coordinates, TapAction
        from ronin.agent.device_planner import GROUNDING_MARKS

        decision = self._verdict(
            TapAction(coordinates=Coordinates(x=535, y=1222), reason="r"),
            GROUNDING_MARKS,
        )

        self.assertFalse(decision.ok)
        self.assertIn("mark", decision.observation)

    def test_each_mode_makes_the_other_mode_unrepresentable(self):
        """Prose plus a runtime check is not enough — the model obeys the schema."""
        from pydantic import ValidationError

        from ronin.agent.actuation.actions import TapAction, TapMarkAction

        with self.assertRaises(ValidationError):
            TapAction(reason="r")  # coordinates are required
        with self.assertRaises(ValidationError):
            TapMarkAction(reason="r")  # a mark is required
        self.assertNotIn("coordinates", TapMarkAction.model_fields)
        self.assertNotIn("mark", TapAction.model_fields)

    def test_the_marks_schema_offers_no_coordinates_at_all(self):
        from ronin.agent.device_planner import MarksStepResponse

        schema = MarksStepResponse.model_json_schema()
        tap = next(d for n, d in schema["$defs"].items() if n.startswith("Tap"))

        self.assertEqual(sorted(tap["properties"]), ["mark", "reason", "type"])
        self.assertIn("mark", tap["required"])

    def test_an_untargeted_action_needs_neither(self):
        from ronin.agent.actuation.actions import KeyEventAction
        from ronin.agent.device_planner import GROUNDING_MARKS

        self.assertTrue(self._verdict(KeyEventAction(key=4, reason="back"), GROUNDING_MARKS).ok)

    def test_the_heading_matches_the_mode(self):
        from ronin.agent.device_planner import (
            GROUNDING_COORDINATES,
            GROUNDING_MARKS,
            DevicePlanner,
        )

        planner = DevicePlanner.__new__(DevicePlanner)
        planner.grounding = GROUNDING_COORDINATES
        self.assertNotIn("numbered", planner._target_heading())
        planner.grounding = GROUNDING_MARKS
        self.assertIn("numbered", planner._target_heading())

    def test_the_orchestrator_explains_why_not_listed_cannot_run(self):
        from ronin.agent.actuation.actions import TargetNotListedAction
        from ronin.agent.actuation.orchestrator import ActionOrchestrator

        orchestrator = ActionOrchestrator.__new__(ActionOrchestrator)
        orchestrator._scrubber = None

        with self.assertRaises(ValueError) as caught:
            orchestrator._run_action(
                TargetNotListedAction(description="blue button", reason="r")
            )

        self.assertIn("nothing to", str(caught.exception))

    def test_an_unmatched_screen_is_saved_redacted(self):
        from ronin.agent.targeting import save_unmatched_screen

        media = Mock()
        snapshot = SimpleNamespace(raw_img_bytes=b"png", gui_elements=["el"])

        save_unmatched_screen(media, Path("/tmp/x.png"), snapshot)

        media.save_redacted_bytes.assert_called_once_with(b"png", ["el"], Path("/tmp/x.png"))


class ListRowTests(NestCollapseTests):
    """A row of separate controls is not one control, however neatly it is boxed."""

    def _card(self):
        return [
            self._el("View", "", (40, 600, 1040, 700)),  # the card
            self._el("Text", "Burger Palace", (160, 620, 500, 650)),
            self._el("Text", "Fast Food 1.2 km", (160, 655, 500, 685)),
            self._el("Text", "Menu", (820, 630, 960, 680)),  # the only live part
        ]

    def test_a_card_keeps_its_button_reachable(self):
        from ronin.agent.perception.marks import build

        texts = [m.element.text for m in build(self._card(), screen_area=1080 * 2400)]

        self.assertIn("Menu", texts)
        self.assertIn("Burger Palace", texts)

    def test_a_single_value_wrapper_still_collapses(self):
        from ronin.agent.perception.marks import build

        self.assertEqual(len(build(self._field(), screen_area=1080 * 2400)), 1)

    def test_a_container_with_its_own_label_is_left_alone(self):
        from ronin.agent.perception.marks import build

        labelled = [
            self._el("View", "Total", (40, 600, 1040, 700)),
            self._el("Text", "£35.00", (820, 630, 960, 680)),
        ]

        self.assertEqual(len(build(labelled, screen_area=1080 * 2400)), 2)
