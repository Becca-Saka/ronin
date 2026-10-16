"""Giving up on a screen that will not take an action."""

import unittest


class StallTests(unittest.TestCase):
    """Steps that never reach the device cost a model call and change nothing."""

    def _stall(self):
        from ronin.agent.step.stall import Stall

        return Stall()

    def test_a_screen_gets_more_than_one_attempt(self):
        """One refusal is a thing to read and answer, not a reason to stop."""
        stall = self._stall()

        stall.record_blocked("screenA")
        self.assertFalse(stall.spent)

    def test_a_screen_runs_out_on_the_third_blocked_step(self):
        """The count is written out, so changing it has to be deliberate.

        Reading STALL_LIMIT here instead would move with it and assert nothing.
        """
        stall = self._stall()

        stall.record_blocked("screenA")
        self.assertFalse(stall.spent)
        stall.record_blocked("screenA")
        self.assertFalse(stall.spent)
        stall.record_blocked("screenA")
        self.assertTrue(stall.spent)

    def test_reaching_a_new_screen_starts_the_count_again(self):
        """Scrolling or dismissing a keyboard changes the screen, which is progress."""
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenA")

        stall.record_blocked("screenB")
        self.assertFalse(stall.spent)

    def test_an_action_that_reaches_the_device_clears_the_count(self):
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenA")

        stall.action_dispatched()

        stall.record_blocked("screenA")
        self.assertFalse(stall.spent)

    def test_an_unreadable_screen_never_accumulates(self):
        """Nothing can be compared with it, so it gets the lenient reading."""
        stall = self._stall()

        for _ in range(5):
            stall.record_blocked(None)

        self.assertFalse(stall.spent)

    def test_a_spent_screen_reports_itself_without_using_an_attempt(self):
        """The execution path asks after the fact; asking must not cost a try."""
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenA")

        self.assertFalse(stall.spent)
        stall.record_blocked("screenA")
        self.assertTrue(stall.spent)
        self.assertTrue(stall.spent)

    def test_a_screen_the_agent_came_back_to_is_judged_afresh(self):
        """Leaving and returning is progress, even back to the same pixels."""
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenB")

        stall.record_blocked("screenA")
        self.assertFalse(stall.spent)
