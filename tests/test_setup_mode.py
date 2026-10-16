"""The state a scenario needs, established rather than assumed."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class SetupModeTests(unittest.TestCase):
    """Required state is part of what a test is.

    Left unstated it becomes a sentence in a prompt, and a prompt cannot
    enforce anything: one run asserted the app always opens on a login screen,
    restarted without clearing data, and skipped the scenario on finding the
    screen the previous one had navigated to.
    """

    def test_the_default_is_the_one_that_makes_scenarios_independent(self):
        from ronin.agent.device.launch import SetupMode
        from ronin.testing.models import GeneratedTestCase

        case = GeneratedTestCase(
            test_id="t",
            scenario_name="s",
            classification="happy_path",
            steps_instructions=["open the app"],
        )

        self.assertIs(case.setup, SetupMode.CLEAR_DATA)

    def test_what_the_planner_is_told_never_names_a_screen(self):
        """A named screen is a claim about one app, and a wrong one is obeyed."""
        from ronin.agent.device.launch import SetupMode

        for mode in SetupMode:
            described = mode.describe().lower()
            self.assertNotIn("login screen", described, mode)
            self.assertNotIn("home screen", described, mode)

    def test_each_mode_says_something_different(self):
        from ronin.agent.device.launch import SetupMode

        said = {mode.describe() for mode in SetupMode}

        self.assertEqual(len(said), len(list(SetupMode)))


class SetupExecutionTests(unittest.TestCase):
    """Each mode has to do what it says, or the contract is another assumption."""

    def _run(self, mode):
        from ronin.agent.device_agent import DeviceAgent

        agent = DeviceAgent.__new__(DeviceAgent)
        agent.replay_ledger = None
        agent._default_fast_planner = Mock()
        calls = []

        # The run is cut short once setup is done. Only the setup matters.
        with (
            patch("ronin.agent.device_agent.clear_app_data", lambda _d: calls.append("clear")),
            patch("ronin.agent.device_agent.restart_app", lambda _d: calls.append("restart")),
            patch("ronin.agent.device_agent.replay_known_path", side_effect=RuntimeError("stop")),
            contextlib.suppress(Exception),
        ):
                DeviceAgent.run(
                    agent,
                    goal="g",
                    device_control=SimpleNamespace(screen_height=2400),
                    perception=Mock(),
                    session_dir=Mock(),
                    setup=mode,
                )
        return calls

    def test_clear_data_wipes_then_restarts(self):
        from ronin.agent.device.launch import SetupMode

        self.assertEqual(self._run(SetupMode.CLEAR_DATA), ["clear", "restart"])

    def test_relaunch_restarts_without_wiping(self):
        from ronin.agent.device.launch import SetupMode

        self.assertEqual(self._run(SetupMode.RELAUNCH), ["restart"])

    def test_preserve_touches_nothing(self):
        from ronin.agent.device.launch import SetupMode

        self.assertEqual(self._run(SetupMode.PRESERVE), [])
