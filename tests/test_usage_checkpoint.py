"""What survives a run that does not reach its end."""

import json
import tempfile
import unittest
from pathlib import Path

from ronin.llm.usage import CallRecord


def _call(cost: float = 0.01) -> CallRecord:
    return CallRecord(
        model="openrouter/google/gemini-2.5-flash",
        served_model="gemini-2.5-flash",
        generation_id="gen-1",
        attempt=1,
        finish_reason="stop",
        latency_ms=100.0,
        cost=cost,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )


class UsageCheckpointTests(unittest.TestCase):
    """A crash after scenario three must not lose scenarios one and two."""

    def test_a_checkpoint_holds_what_has_been_spent_so_far(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            tracker.add_call("scenario:one", _call())
            written = tracker.checkpoint()

            self.assertEqual(written.name, "llm_usage_partial.json")
            saved = json.loads(written.read_text())
            self.assertEqual(saved["total_calls"], 1)

    def test_a_later_checkpoint_replaces_the_earlier_one(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            tracker.add_call("scenario:one", _call())
            tracker.checkpoint()
            tracker.add_call("scenario:two", _call())
            written = tracker.checkpoint()

            self.assertEqual(json.loads(written.read_text())["total_calls"], 2)
            self.assertEqual(
                len(list(Path(directory).glob("llm_usage_partial*.json"))), 1
            )

    def test_the_final_report_keeps_its_own_name(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            tracker.add_call("scenario:one", _call())
            tracker.checkpoint()
            final = tracker.save_report()

            self.assertTrue(final.name.startswith("llm_usage_2"))
            self.assertTrue((Path(directory) / "llm_usage_partial.json").exists())

    def test_a_checkpoint_that_cannot_be_written_does_not_stop_the_run(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        tracker = LLMUsageTracker(output_dir=Path("/dev/null/nowhere"))
        tracker.add_call("scenario:one", _call())

        self.assertIsNone(tracker.checkpoint())


class CallLogTests(unittest.TestCase):
    """A crash inside a scenario keeps every call it had already made."""

    def test_each_call_is_logged_as_it_happens(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            tracker.add_call("scenario:one", _call())
            tracker.add_call("scenario:one", _call())

            lines = tracker.log_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["component"], "scenario:one")

    def test_the_log_names_the_scenario_each_call_belongs_to(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            tracker.add_call("scenario:one", _call())
            tracker.add_call("scenario:two", _call())

            components = [
                json.loads(line)["component"]
                for line in tracker.log_path.read_text().splitlines()
            ]
            self.assertEqual(components, ["scenario:one", "scenario:two"])

    def test_the_log_survives_what_the_roll_up_would_lose(self):
        """The roll-up is written between scenarios; the log is written per call."""
        from ronin.testing.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory))
            tracker.add_call("scenario:one", _call())
            tracker.checkpoint()
            # Scenario two dies before its checkpoint.
            tracker.add_call("scenario:two", _call())

            rolled_up = json.loads((Path(directory) / "llm_usage_partial.json").read_text())
            self.assertEqual(rolled_up["total_calls"], 1)
            self.assertEqual(len(tracker.log_path.read_text().splitlines()), 2)

    def test_a_declared_secret_never_reaches_the_log(self):
        from ronin.testing.usage_tracker import LLMUsageTracker
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})
        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=Path(directory), scrubber=scrubber)
            call = _call()
            object.__setattr__(call, "raw", {"choices": [{"text": "typed hunter2"}]})
            tracker.add_call("scenario:one", call)

            written = tracker.log_path.read_text()
            self.assertNotIn("hunter2", written)

    def test_a_log_that_cannot_be_written_does_not_stop_the_run(self):
        from ronin.testing.usage_tracker import LLMUsageTracker

        tracker = LLMUsageTracker(output_dir=Path("/dev/null/nowhere"))
        tracker.add_call("scenario:one", _call())

        self.assertEqual(tracker.totals.total_calls, 1)


if __name__ == "__main__":
    unittest.main()
