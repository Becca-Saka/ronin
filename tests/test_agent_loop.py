"""Settling, stopping, and replaying navigation already worked out."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ronin.agent.device_agent import (
    AgentHistory,
    DeviceAgent,
    RollingLoopDetector,
    _RunContext,
)


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


class ReplayTests(unittest.TestCase):
    def _replay(self, agent, goal, context):
        from ronin.agent.replay import replay_known_path

        return replay_known_path(
            ledger=agent.replay_ledger,
            device_control=object(),
            goal=goal,
            history=context.history,
            quick_phash=agent._quick_phash,
            settle=agent._settle,
            scrubber=agent.scrubber,
        )

    def _agent(self, ledger, phashes):

        from ronin.utils.secrets import SecretScrubber

        return SimpleNamespace(
            scrubber=SecretScrubber(),
            replay_ledger=ledger,
            _quick_phash=Mock(side_effect=phashes),
            _settle=Mock(side_effect=phashes[1::2]),
        )

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

    @patch("ronin.agent.replay.ActionOrchestrator")
    def test_replays_a_known_transition(self, orchestrator):
        ledger = self._ledger()
        agent = self._agent(ledger, ["0" * 64, "1" * 64, "1" * 64])
        context = self._context()

        self._replay(agent, ["tap the Login button"], context)

        orchestrator.return_value.execute.assert_called_once()
        self.assertEqual(len(context.history._entries), 1)

    @patch("ronin.agent.replay.ActionOrchestrator")
    def test_stops_when_the_screen_is_not_where_it_used_to_lead(self, orchestrator):
        ledger = self._ledger()
        agent = self._agent(ledger, ["0" * 64, "f" * 64])
        context = self._context()

        self._replay(agent, ["tap the Login button"], context)

        orchestrator.return_value.execute.assert_called_once()
        self.assertEqual(len(context.history._entries), 0)

    @patch("ronin.agent.replay.ActionOrchestrator")
    def test_does_not_replay_text_the_scenario_never_mentions(self, orchestrator):
        from ronin.agent.replay import ReplayLedger

        ledger = ReplayLedger()
        ledger.record("0" * 64, self._typing("password123"), "1" * 64, "type password", "Password")
        agent = self._agent(ledger, ["0" * 64])

        self._replay(agent, ["sign in with 'wrongpass'"], self._context())

        orchestrator.return_value.execute.assert_not_called()

    @patch("ronin.agent.replay.ActionOrchestrator")
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

    @patch("ronin.agent.replay.ActionOrchestrator")
    def test_unknown_screen_plans_as_usual(self, orchestrator):
        agent = self._agent(self._ledger(), ["e" * 64])

        self._replay(agent, ["tap the Login button"], self._context())

        orchestrator.return_value.execute.assert_not_called()

    @patch("ronin.agent.replay.ActionOrchestrator")
    def test_does_not_replay_a_tap_this_scenario_never_asked_for(self, orchestrator):
        """A 'Place Order' tap learned elsewhere must not fire in another scenario."""
        from ronin.agent.replay import ReplayLedger

        ledger = ReplayLedger()
        ledger.record("0" * 64, self._tap(), "1" * 64, "tap Place Order", "Place Order")
        agent = self._agent(ledger, ["0" * 64])

        self._replay(agent, ["verify the cart subtotal"], self._context())

        orchestrator.return_value.execute.assert_not_called()

    @patch("ronin.agent.replay.ActionOrchestrator")
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

        self._replay(agent, ["anything"], self._context())

        orchestrator.return_value.execute.assert_not_called()


