"""Opening up a screen so a blocked step becomes possible."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock


def _action(kind, **fields):
    """Done is a separate model, not an ActionType member."""
    from ronin.agent.actuation.actions import ActionType
    from ronin.agent.device_planner import DoneAction

    if kind == "done":
        return DoneAction(reason="because")
    return SimpleNamespace(type=ActionType(kind), reason="because", **fields)


class ForbiddenActionTests(unittest.TestCase):
    """Recovery opens the screen. It never performs the step it unblocked.

    Doing the step here would leave the main planner to do it again, and the
    two would have to be reconciled. Refusing outright removes the question.
    """

    def test_typing_is_refused(self):
        from ronin.agent.step.recovery import _permitted

        self.assertFalse(_permitted(_action("type", text="4111", mark=3)))

    def test_clearing_is_refused(self):
        from ronin.agent.step.recovery import _permitted

        self.assertFalse(_permitted(_action("clear", mark=3)))

    def test_giving_up_is_refused(self):
        """Recovery already has a way to give up: running out of moves."""
        from ronin.agent.step.recovery import _permitted

        self.assertFalse(_permitted(_action("target_not_listed", description="x")))

    def test_device_level_commands_are_refused(self):
        """Granting a permission changes the device, not what the screen shows."""
        from ronin.agent.step.recovery import _permitted

        self.assertFalse(_permitted(_action("capability", capability="permission")))

    def test_tapping_is_refused_because_a_tap_commits_something(self):
        """Nothing tells us what is under it, and the screen is a checkout page.

        The prompt saying "never submit" is a request. This is the guard.
        """
        from ronin.agent.step.recovery import _permitted

        self.assertFalse(_permitted(_action("tap", mark=3)))

    def test_only_back_is_an_allowed_key(self):
        """Enter submits a form and home leaves the app."""
        from ronin.agent.step.recovery import _permitted

        self.assertTrue(_permitted(_action("key_event", key=4)))
        for key in (66, 3, 67):
            self.assertFalse(_permitted(_action("key_event", key=key)), key)

    def test_the_moves_that_commit_nothing_are_allowed(self):
        from ronin.agent.step.recovery import _permitted

        self.assertTrue(_permitted(_action("swipe", start_x=1, start_y=2, end_x=3, end_y=4)))
        self.assertTrue(_permitted(_action("wait", duration=1.0)))


class RevealLoopTests(unittest.TestCase):
    """How many moves it spends, and when it hands back."""

    def _perception(self):
        snapshot = SimpleNamespace(
            gui_elements=[],
            marks=[],
            raw_img_bytes=b"png",
            planner_img_bytes=b"png",
            current_phash="screenA",
            timestamp="t",
            scan_info=None,
            prev_phash=None,
        )
        return snapshot

    def _run(self, responses, evaluate_ok=True):
        from unittest.mock import patch

        from ronin.agent.step import recovery

        planner = SimpleNamespace(plan_step=Mock(side_effect=responses))
        executed = []

        with (
            patch.object(recovery.observation, "capture", return_value=self._perception()),
            patch.object(recovery.observation, "number_targets"),
            patch.object(
                recovery.targeting,
                "evaluate",
                return_value=SimpleNamespace(
                    ok=evaluate_ok, action=None, observation="refused"
                ),
            ),
            patch.object(recovery, "ActionOrchestrator") as orchestrator,
        ):
            orchestrator.return_value.execute = lambda a: executed.append(a.type.value)
            note, _looks = recovery.reveal(
                target="the CVV field",
                planner=planner,
                device_control=SimpleNamespace(
                    showing_system_prompt=lambda: False,
                    is_still_in_sut=lambda: True,
                    is_keyboard_open=lambda: False,
                ),
                perception=SimpleNamespace(),
                grounding="marks",
                scrubber=None,
                screen_height=2400,
            )
        return note, executed, planner.plan_step

    def _response(self, action):
        return SimpleNamespace(action=action, reasoning="r")

    def test_it_stops_the_moment_the_target_is_reachable(self):
        """Carrying on after success is what spent the budget in a watched run."""
        note, executed, plan = self._run(
            [
                self._response(_action("key_event", key=4)),
                self._response(_action("done")),
                self._response(_action("tap", mark=1)),
            ]
        )

        self.assertIsNotNone(note)
        self.assertEqual(plan.call_count, 2)
        self.assertEqual(executed, ["key_event"])

    def test_an_already_reachable_target_costs_one_look(self):
        note, executed, plan = self._run([self._response(_action("done"))])

        self.assertIn("already reachable", note)
        self.assertEqual(executed, [])

    def test_it_gives_up_after_four_moves(self):
        """Written out, so widening the budget has to be deliberate.

        Reading REVEAL_STEPS here instead would move with it and assert nothing.
        """
        note, executed, plan = self._run(
            [self._response(_action("swipe", start_x=1, start_y=2, end_x=3, end_y=4))] * 20
        )

        self.assertIsNone(note)
        self.assertEqual(len(executed), 4, "four moves")
        self.assertEqual(plan.call_count, 5, "one look more than moves")

    def test_a_forbidden_choice_is_not_executed_and_does_not_end_it(self):
        note, executed, plan = self._run(
            [
                self._response(_action("type", text="4111", mark=3)),
                self._response(_action("done")),
            ]
        )

        self.assertEqual(executed, [])
        self.assertIsNotNone(note)

    def test_it_hands_back_if_it_has_left_the_app(self):
        """Back on a root screen backgrounds the app, and the rest is the launcher."""
        from unittest.mock import patch

        from ronin.agent.step import recovery

        planner = SimpleNamespace(plan_step=Mock())
        with patch.object(recovery.observation, "capture") as capture:
            note, _looks = recovery.reveal(
                target="the CVV field",
                planner=planner,
                device_control=SimpleNamespace(
                    showing_system_prompt=lambda: False,
                    is_still_in_sut=lambda: False,
                ),
                perception=SimpleNamespace(),
                grounding="marks",
                scrubber=None,
                screen_height=2400,
            )

        self.assertIsNone(note)
        capture.assert_not_called()
        planner.plan_step.assert_not_called()

    def test_it_stands_down_under_a_system_prompt(self):
        """Every move it has closes something, and closing a prompt denies it.

        The drift check already refuses to press back on one. Recovery runs
        after that check has passed the prompt through to the planner, so it
        has to refuse for itself.
        """
        from unittest.mock import patch

        from ronin.agent.step import recovery

        planner = SimpleNamespace(plan_step=Mock())
        with patch.object(recovery.observation, "capture") as capture:
            note, looks = recovery.reveal(
                target="the CVV field",
                planner=planner,
                device_control=SimpleNamespace(
                    showing_system_prompt=lambda: True, is_still_in_sut=lambda: True
                ),
                perception=SimpleNamespace(),
                grounding="marks",
                scrubber=None,
                screen_height=2400,
            )

        self.assertIsNone(note)
        capture.assert_not_called()
        planner.plan_step.assert_not_called()

    def test_a_final_move_that_works_is_still_recognised(self):
        """Observing only before acting would fail a scenario the last swipe saved."""
        swipe = self._response(_action("swipe", start_x=1, start_y=2, end_x=3, end_y=4))
        note, executed, plan = self._run(
            [swipe, swipe, swipe, swipe, self._response(_action("done"))]
        )

        self.assertIsNotNone(note)
        self.assertEqual(len(executed), 4)

    def test_the_note_says_what_was_done(self):
        """The main history gets one line, not the recovery's whole trail."""
        note, _executed, _plan = self._run(
            [
                self._response(_action("key_event", key=4)),
                self._response(_action("done")),
            ]
        )

        self.assertIn("key_event", note)


class KeyboardRevealTests(unittest.TestCase):
    """Looking behind the keyboard before deciding a screen is hopeless.

    Only what is on screen carries a number, and the keyboard covers the lower
    half. A field underneath it cannot be named, so the planner reaches for
    whatever is visible.
    """

    def _ctx(self):
        from ronin.agent.step.run_context import RunContext

        return RunContext(history=None, loop_detector=None)

    def _device(self, keyboard_open, closes=True):
        pressed = []

        def dismiss():
            pressed.append("dismiss")
            return closes

        return SimpleNamespace(
            is_keyboard_open=lambda: keyboard_open, dismiss_keyboard=dismiss
        ), pressed

    def _snapshot(self, phash="screenA"):
        return SimpleNamespace(current_phash=phash)

    def _reveal(self, device, snapshot, ctx):
        from ronin.agent.step.recovery import reveal_behind_keyboard

        return reveal_behind_keyboard(device, snapshot, ctx)

    def test_a_blocked_step_with_the_keyboard_up_closes_it(self):
        device, pressed = self._device(keyboard_open=True)

        self.assertTrue(self._reveal(device, self._snapshot(), self._ctx()))
        self.assertEqual(pressed, ["dismiss"])

    def test_nothing_happens_when_the_keyboard_is_already_down(self):
        """The screen is fully visible, so the target is genuinely absent."""
        device, pressed = self._device(keyboard_open=False)

        self.assertFalse(self._reveal(device, self._snapshot(), self._ctx()))
        self.assertEqual(pressed, [])

    def test_one_screen_gets_one_look_behind_the_keyboard(self):
        """A second failure on the same screen is not about the keyboard."""
        device, pressed = self._device(keyboard_open=True)
        ctx = self._ctx()

        self._reveal(device, self._snapshot(), ctx)

        self.assertFalse(self._reveal(device, self._snapshot(), ctx))
        self.assertEqual(pressed, ["dismiss"])

    def test_a_different_screen_gets_its_own_look(self):
        device, pressed = self._device(keyboard_open=True)
        ctx = self._ctx()

        self._reveal(device, self._snapshot("screenA"), ctx)
        self._reveal(device, self._snapshot("screenB"), ctx)

        self.assertEqual(pressed, ["dismiss", "dismiss"])

    def test_an_unknown_keyboard_state_is_not_treated_as_open(self):
        """is_keyboard_open returns None when no check could answer."""
        device, pressed = self._device(keyboard_open=None)

        self.assertFalse(self._reveal(device, self._snapshot(), self._ctx()))
        self.assertEqual(pressed, [])

    def test_a_close_that_fails_does_not_spend_the_screens_one_attempt(self):
        """iOS hide_keyboard can fail; the screen is no more visible for trying."""
        broken, broken_presses = self._device(keyboard_open=True, closes=False)
        ctx = self._ctx()

        self.assertFalse(self._reveal(broken, self._snapshot(), ctx))

        working, _ = self._device(keyboard_open=True)
        self.assertTrue(self._reveal(working, self._snapshot(), ctx))
        self.assertEqual(broken_presses, ["dismiss"])

    def test_an_unidentified_screen_is_not_recorded_as_dismissed(self):
        """Nothing can match it later, so it would consume every attempt."""
        device, pressed = self._device(keyboard_open=True)
        ctx = self._ctx()

        self.assertFalse(self._reveal(device, self._snapshot(phash=None), ctx))
        self.assertEqual(pressed, [])
        self.assertEqual(ctx.keyboard_dismissed, set())


class RecoveryBudgetTests(unittest.TestCase):
    """Recovery spends the scenario's money, so the scenario must see it."""

    def _ctx(self):
        from ronin.agent.history import AgentHistory
        from ronin.agent.step.run_context import RunContext

        ctx = RunContext(history=AgentHistory(), loop_detector=None)
        ctx.current_phash = "screenA"
        return ctx

    def _hand_over(self, ctx, note, looks):
        from unittest.mock import patch

        from ronin.agent.step import recovery

        with (
            patch.object(recovery, "reveal", return_value=(note, looks)),
            patch.object(recovery, "RevealPlanner"),
        ):
            return recovery.hand_over(
                target="the CVV field",
                llm=None,
                use_multimodal_vision=False,
                grounding="marks",
                scrubber=None,
                device_control=SimpleNamespace(),
                perception=SimpleNamespace(),
                ctx=ctx,
            )

    def test_the_looks_it_took_are_counted(self):
        ctx = self._ctx()

        self._hand_over(ctx, note="did a thing", looks=5)

        self.assertEqual(ctx.recovery_looks, 5)

    def test_a_recovery_that_failed_is_counted_too(self):
        """The model calls were made whether or not they helped."""
        ctx = self._ctx()

        self.assertFalse(self._hand_over(ctx, note=None, looks=5))
        self.assertEqual(ctx.recovery_looks, 5)

    def test_one_screen_is_rescued_once(self):
        ctx = self._ctx()
        self._hand_over(ctx, note="did a thing", looks=1)

        self.assertFalse(self._hand_over(ctx, note="again", looks=1))

    def test_a_scenario_is_rescued_at_most_three_times(self):
        """Capped per screen alone, a long journey rescues at every screen."""
        ctx = self._ctx()
        for n in range(3):
            ctx.current_phash = f"screen{n}"
            self.assertTrue(self._hand_over(ctx, note="did a thing", looks=1))

        ctx.current_phash = "screen4"
        self.assertFalse(self._hand_over(ctx, note="did a thing", looks=1))

    def test_the_run_s_own_step_count_is_left_alone(self):
        """History is numbered by loop step; mixing in recovery drifts it."""
        ctx = self._ctx()
        ctx.steps_executed = 7

        self._hand_over(ctx, note="did a thing", looks=5)

        self.assertEqual(ctx.steps_executed, 7)
        self.assertEqual(ctx.recovery_looks, 5)

    def test_a_recovery_that_throws_does_not_change_the_scenario_outcome(self):
        """Otherwise a failed rescue turns a stalled screen into an ERROR.

        The scenario should report the wall it hit, not the rescue that could
        not get it over.
        """
        from unittest.mock import patch

        from ronin.agent.step import recovery

        ctx = self._ctx()
        with (
            patch.object(recovery, "reveal", side_effect=RuntimeError("planner died")),
            patch.object(recovery, "RevealPlanner"),
        ):
            rescued = recovery.hand_over(
                target="the CVV field",
                llm=None,
                use_multimodal_vision=False,
                grounding="marks",
                scrubber=None,
                device_control=SimpleNamespace(),
                perception=SimpleNamespace(),
                ctx=ctx,
            )

        self.assertFalse(rescued)
