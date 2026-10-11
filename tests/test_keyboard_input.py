import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ronin.agent.device.device_control import DeviceControl
from ronin.agent.device.platform import Platform
from ronin.agent.device_agent import (
    AgentHistory,
    DeviceAgent,
    RollingLoopDetector,
    _RunContext,
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
            raw_img_bytes=b"image",
        )
        context = _RunContext(
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


class DeviceTypingTests(unittest.TestCase):
    def _control(self, rect):
        element = SimpleNamespace(id="field", rect=rect, send_keys=Mock())
        driver = SimpleNamespace(
            switch_to=SimpleNamespace(active_element=element),
        )
        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.ANDROID
        control.driver = driver
        control.tap = Mock()
        control.screen_width = 1080
        control.screen_height = 2400
        return control, element

    @patch("ronin.agent.device.device_control.time.sleep")
    def test_type_reuses_already_focused_field(self, _sleep):
        control, element = self._control(
            {"x": 63, "y": 820, "width": 954, "height": 147}
        )

        control.type("test@email.com", 540, 893)

        control.tap.assert_not_called()
        element.send_keys.assert_called_once_with("test@email.com")

    @patch("ronin.agent.device.device_control.time.sleep")
    def test_type_rejects_stale_coordinates(self, _sleep):
        control, element = self._control(
            {"x": 63, "y": 820, "width": 954, "height": 147}
        )

        with self.assertRaisesRegex(RuntimeError, "did not focus anything there"):
            control.type("test@email.com", 537, 1260)

        control.tap.assert_called_once_with(537, 1260)
        element.send_keys.assert_not_called()

    @patch("ronin.agent.device.device_control.time.sleep")
    def test_type_rejects_a_tap_that_focused_a_different_field(self, _sleep):
        """Any text field opens the keyboard, so a wrong field looks like a reflow."""
        before = SimpleNamespace(id="nothing", rect={"x": 0, "y": 0, "width": 0, "height": 0})
        email = SimpleNamespace(
            id="email",
            rect={"x": 63, "y": 820, "width": 954, "height": 147},
            send_keys=Mock(),
        )
        switch_to = SimpleNamespace(active_element=before)
        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.ANDROID
        control.driver = SimpleNamespace(switch_to=switch_to)
        control.screen_width, control.screen_height = 1080, 2400

        def tap(_x, _y):
            switch_to.active_element = email

        control.tap = Mock(side_effect=tap)

        with self.assertRaisesRegex(RuntimeError, "did not focus anything there"):
            control.type("password123", 537, 1444)

        email.send_keys.assert_not_called()

    @patch("ronin.agent.device.device_control.time.sleep")
    def test_clear_propagates_stale_coordinate_failure(self, _sleep):
        control, _element = self._control(
            {"x": 63, "y": 820, "width": 954, "height": 147}
        )

        with self.assertRaisesRegex(RuntimeError, "did not focus anything there"):
            control.clear(537, 1260)

        control.tap.assert_called_once_with(537, 1260)


if __name__ == "__main__":
    unittest.main()


class SettleTests(unittest.TestCase):
    def _agent(self, hashes):
        agent = DeviceAgent.__new__(DeviceAgent)
        agent.settle_time = 2.0
        agent._quick_phash = Mock(side_effect=hashes)
        return agent

    @patch("ronin.agent.device_agent.time.sleep")
    def test_settle_returns_as_soon_as_the_screen_repeats(self, _sleep):
        agent = self._agent(["a", "a", "should-not-be-read"])

        self.assertEqual(agent._settle(object()), "a")
        self.assertEqual(agent._quick_phash.call_count, 2)

    @patch("ronin.agent.device_agent.time.sleep")
    def test_settle_gives_up_at_the_settle_budget(self, _sleep):
        agent = self._agent(["a", "b", "c", "d", "e", "f", "g"])

        agent._settle(object())

        # settle_time 2.0 / 0.35 poll = 5 polls after the first reading
        self.assertEqual(agent._quick_phash.call_count, 6)


class EarlyStopTests(unittest.TestCase):
    def _agent(self, stop_when_satisfied):
        agent = DeviceAgent.__new__(DeviceAgent)
        agent.stop_when_satisfied = stop_when_satisfied
        agent._flush_trail = Mock()
        return agent

    def _response(self, **kwargs):
        return SimpleNamespace(action=SimpleNamespace(type="tap"), **kwargs)

    def _context(self):
        return _RunContext(
            history=AgentHistory(), loop_detector=RollingLoopDetector(), steps_executed=3
        )

    def test_stops_when_the_goal_is_already_visible(self):
        agent = self._agent(True)
        response = self._response(goal_satisfied=True, goal_evidence="restaurant list shown")

        result = agent._check_done_action(response, None, "phash", self._context())

        self.assertIsNotNone(result)
        self.assertEqual(result.status, "goal_reached")

    def test_keeps_going_for_journeys(self):
        agent = self._agent(False)
        response = self._response(goal_satisfied=True, goal_evidence="restaurant list shown")

        self.assertIsNone(agent._check_done_action(response, None, "phash", self._context()))


class WarmSessionTests(unittest.TestCase):
    def _control(self):
        control = DeviceControl.__new__(DeviceControl)
        control.device_name = "Pixel_8"
        return control

    def _options(self):
        captured = {}
        return SimpleNamespace(set_capability=lambda k, v: captured.__setitem__(k, v)), captured

    @patch("ronin.agent.device.device_control.AdbControl")
    def test_skips_setup_when_the_helper_apps_are_present(self, adb):
        adb.resolve_emulator_serial.return_value = "emulator-5554"
        adb.installed_packages.return_value = {
            "io.appium.uiautomator2.server",
            "io.appium.uiautomator2.server.test",
            "io.appium.settings",
        }
        options, captured = self._options()

        self._control()._skip_redundant_setup(options)

        self.assertTrue(captured["appium:skipServerInstallation"])
        self.assertTrue(captured["appium:skipDeviceInitialization"])

    @patch("ronin.agent.device.device_control.AdbControl")
    def test_does_a_full_setup_when_the_server_is_missing(self, adb):
        adb.resolve_emulator_serial.return_value = "emulator-5554"
        adb.installed_packages.return_value = {"io.appium.settings"}
        options, captured = self._options()

        self._control()._skip_redundant_setup(options)

        self.assertEqual(captured, {})


class ElementFormattingTests(unittest.TestCase):
    """The restaurant list that caused a one-row mis-tap in a real run."""

    def _planner(self):
        from ronin.agent.device_planner import DevicePlanner

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


class ScenarioProgressTests(unittest.TestCase):
    def _progress(self, directory):
        from ronin.testing.models import GeneratedTestCase, UserJourney
        from ronin.testing.progress import ScenarioProgress

        journey = UserJourney(id="j1", name="Journey", description="d", security_rules=[])
        scenarios = [
            GeneratedTestCase(
                test_id=f"TS_00{n}",
                scenario_name=f"Scenario {n}",
                classification="happy_path",
                steps_instructions=["open app"],
                expected_outcomes=["ok"],
            )
            for n in (1, 2)
        ]
        return ScenarioProgress(Path(directory), journey, scenarios)

    def test_plan_is_written_before_anything_runs(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            progress = self._progress(tmp)
            progress.write()

            data = json.loads(progress.path.read_text())
            self.assertEqual(data["total"], 2)
            self.assertEqual(data["counts"], {"pending": 2})

    def test_status_updates_as_scenarios_finish(self):
        import json
        import tempfile

        from ronin.testing.models import ScenarioReport

        with tempfile.TemporaryDirectory() as tmp:
            progress = self._progress(tmp)
            progress.start("TS_001")
            progress.record(
                "TS_001",
                ScenarioReport(
                    test_id="TS_001",
                    scenario_name="Scenario 1",
                    classification="happy_path",
                    status="FAILED",
                    duration=12.34,
                    steps_executed=9,
                    failure_reason="wrong screen",
                ),
            )

            data = json.loads(progress.path.read_text())
            first = data["scenarios"][0]
            self.assertEqual(first["status"], "FAILED")
            self.assertEqual(first["steps_executed"], 9)
            self.assertEqual(first["duration_seconds"], 12.3)
            self.assertEqual(data["counts"], {"FAILED": 1, "pending": 1})


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


class ReplayTests(unittest.TestCase):
    def _agent(self, ledger, phashes):
        from ronin.agent.device_agent import DeviceAgent

        from ronin.utils.secrets import SecretScrubber

        agent = DeviceAgent.__new__(DeviceAgent)
        agent.scrubber = SecretScrubber()
        agent.replay_ledger = ledger
        agent._quick_phash = Mock(side_effect=phashes)
        agent._settle = Mock(side_effect=phashes[1::2])
        return agent

    def _tap(self):
        from ronin.agent.actuation.actions import ActionType, Coordinates, TapAction

        return TapAction(type=ActionType.TAP, coordinates=Coordinates(x=537, y=1664), reason="r")

    def _typing(self, text):
        from ronin.agent.actuation.actions import ActionType, Coordinates, TypeAction

        return TypeAction(
            type=ActionType.TYPE, coordinates=Coordinates(x=537, y=1080), text=text, reason="r"
        )

    def _ledger(self):
        from ronin.agent.replay import ReplayLedger

        ledger = ReplayLedger()
        ledger.record("0" * 64, self._tap(), "1" * 64, "tap Login", "Login")
        return ledger

    def _context(self):
        return _RunContext(history=AgentHistory(), loop_detector=RollingLoopDetector())

    @patch("ronin.agent.device_agent.ActionOrchestrator")
    def test_replays_a_known_transition(self, orchestrator):
        ledger = self._ledger()
        agent = self._agent(ledger, ["0" * 64, "1" * 64, "1" * 64])
        context = self._context()

        agent._replay_known_path(object(), ["tap the Login button"], context)

        orchestrator.return_value.execute.assert_called_once()
        self.assertEqual(len(context.history._entries), 1)

    @patch("ronin.agent.device_agent.ActionOrchestrator")
    def test_stops_when_the_screen_is_not_where_it_used_to_lead(self, orchestrator):
        ledger = self._ledger()
        agent = self._agent(ledger, ["0" * 64, "f" * 64])
        context = self._context()

        agent._replay_known_path(object(), ["tap the Login button"], context)

        orchestrator.return_value.execute.assert_called_once()
        self.assertEqual(len(context.history._entries), 0)

    @patch("ronin.agent.device_agent.ActionOrchestrator")
    def test_does_not_replay_text_the_scenario_never_mentions(self, orchestrator):
        from ronin.agent.replay import ReplayLedger

        ledger = ReplayLedger()
        ledger.record("0" * 64, self._typing("password123"), "1" * 64, "type password", "Password")
        agent = self._agent(ledger, ["0" * 64])

        agent._replay_known_path(object(), ["sign in with 'wrongpass'"], self._context())

        orchestrator.return_value.execute.assert_not_called()

    @patch("ronin.agent.device_agent.ActionOrchestrator")
    def test_gestures_are_never_replayed(self, orchestrator):
        """Destructive scenarios are built from gestures; they must not be inherited."""
        from ronin.agent.actuation.actions import ActionType, Coordinates, LongPressAction
        from ronin.agent.replay import ReplayLedger

        ledger = ReplayLedger()
        ledger.record(
            "0" * 64,
            LongPressAction(
                type=ActionType.LONG_PRESS,
                coordinates=Coordinates(x=1, y=2),
                duration=1000,
                reason="r",
            ),
            "1" * 64,
            "long press",
        )

        self.assertEqual(len(ledger), 0)

    @patch("ronin.agent.device_agent.ActionOrchestrator")
    def test_unknown_screen_plans_as_usual(self, orchestrator):
        agent = self._agent(self._ledger(), ["e" * 64])

        agent._replay_known_path(object(), ["tap the Login button"], self._context())

        orchestrator.return_value.execute.assert_not_called()

    @patch("ronin.agent.device_agent.ActionOrchestrator")
    def test_does_not_replay_a_tap_this_scenario_never_asked_for(self, orchestrator):
        """A 'Place Order' tap learned elsewhere must not fire in another scenario."""
        from ronin.agent.replay import ReplayLedger

        ledger = ReplayLedger()
        ledger.record("0" * 64, self._tap(), "1" * 64, "tap Place Order", "Place Order")
        agent = self._agent(ledger, ["0" * 64])

        agent._replay_known_path(object(), ["verify the cart subtotal"], self._context())

        orchestrator.return_value.execute.assert_not_called()

    @patch("ronin.agent.device_agent.ActionOrchestrator")
    def test_only_back_is_replayable_among_key_events(self, orchestrator):
        from ronin.agent.actuation.actions import ActionType, KeyEventAction
        from ronin.agent.replay import ReplayLedger

        ledger = ReplayLedger()
        ledger.record(
            "0" * 64,
            KeyEventAction(type=ActionType.KEY_EVENT, key=3, reason="home"),
            "1" * 64,
            "key event",
        )
        agent = self._agent(ledger, ["0" * 64])

        agent._replay_known_path(object(), ["anything"], self._context())

        orchestrator.return_value.execute.assert_not_called()


class ProgressAbortTests(unittest.TestCase):
    def test_a_crash_leaves_nothing_marked_running(self):
        import json
        import tempfile

        from ronin.testing.models import GeneratedTestCase, UserJourney
        from ronin.testing.progress import ScenarioProgress

        journey = UserJourney(id="j1", name="J", description="d", security_rules=[])
        scenarios = [
            GeneratedTestCase(
                test_id=f"TS_00{n}",
                scenario_name=f"S{n}",
                classification="happy_path",
                steps_instructions=["x"],
                expected_outcomes=[],
            )
            for n in (1, 2)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            progress = ScenarioProgress(Path(tmp), journey, scenarios)
            progress.start("TS_001")
            progress.abort("interrupted")

            data = json.loads(progress.path.read_text())
            self.assertEqual(data["counts"], {"ABORTED": 2})
            self.assertTrue(all(s["failure_reason"] == "interrupted" for s in data["scenarios"]))


class SecretScrubbingTests(unittest.TestCase):
    def _scrubber(self):
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_from_instructions(
            ["Sign in with 'test@example.com' and 'password123'"]
        )
        return scrubber

    def test_values_are_removed_from_free_form_reasoning(self):
        scrubbed = self._scrubber().scrub("Type 'password123' into the Password field")

        self.assertNotIn("password123", scrubbed)
        self.assertIn("Password field", scrubbed)

    def test_values_typed_at_runtime_are_learned(self):
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn("hunter2")

        self.assertNotIn("hunter2", scrubber.scrub("I will enter hunter2 now"))

    def test_ordinary_text_is_untouched(self):
        self.assertEqual(
            self._scrubber().scrub("Tap the Login button"), "Tap the Login button"
        )
