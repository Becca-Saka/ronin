"""What a run spent, and what the file has to be able to answer later."""

import json
import tempfile
import unittest
from pathlib import Path


class UsageRecordTests(unittest.TestCase):
    """Every call is written down whole, so a later question still has an answer."""

    def _call(self, **overrides):
        from ronin.llm.usage import CallRecord

        values = {
            "model": "provider/model",
            "served_model": "provider/model-002",
            "generation_id": "gen-1",
            "attempt": 1,
            "finish_reason": "stop",
            "latency_ms": 1200.0,
            "cost": 0.004,
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        values.update(overrides)
        return CallRecord(**values)

    def _recorded(self, *calls):
        from ronin.testing.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(Path(directory))
            for call in calls:
                tracker.add_call("scenario:TS_1", call)
            return json.loads(tracker.save_report().read_text())

    def test_unknown_cost_is_reported_as_unknown(self):
        payload = self._recorded(self._call(cost=None))

        self.assertEqual(payload["cost_unknown_calls"], 1)
        call = payload["components"]["scenario:TS_1"]["calls"][0]
        self.assertFalse(call["cost_known"])
        self.assertEqual(call["model"], "provider/model")

    def test_a_truncated_answer_is_counted_not_just_recorded(self):
        """"length" means the answer was cut off, which reads as a bad answer."""
        payload = self._recorded(self._call(finish_reason="length"))

        self.assertEqual(payload["truncated_calls"], 1)

    def test_a_retry_is_counted_as_one(self):
        payload = self._recorded(self._call(), self._call(attempt=2))

        self.assertEqual(payload["total_calls"], 2)
        self.assertEqual(payload["retried_calls"], 1)

    def test_cached_and_reasoning_tokens_are_totalled(self):
        payload = self._recorded(self._call(cached_tokens=900, reasoning_tokens=40))

        self.assertEqual(payload["total_cached_tokens"], 900)
        self.assertEqual(payload["total_reasoning_tokens"], 40)

    def test_the_whole_response_is_kept(self):
        payload = self._recorded(self._call(raw={"id": "gen-1", "choices": []}))

        self.assertEqual(
            payload["components"]["scenario:TS_1"]["calls"][0]["raw"]["id"], "gen-1"
        )

    def test_declared_secrets_are_masked_inside_the_dumped_response(self):
        from ronin.testing.usage_tracker import LLMUsageTracker
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})
        call = self._call(raw={"choices": [{"text": "I typed hunter2 there"}]})

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(Path(directory), scrubber=scrubber)
            tracker.add_call("scenario:TS_1", call)
            written = tracker.save_report().read_text()

        self.assertNotIn("hunter2", written)
        self.assertIn("{{password}}", written)


class UsageTotalsTests(unittest.TestCase):
    """The totals are a fixed set of fields, so they are a type rather than a dict."""

    def _call(self, **overrides):
        from ronin.llm.usage import CallRecord

        values = {
            "model": "provider/model",
            "served_model": "provider/model",
            "generation_id": "gen-1",
            "attempt": 1,
            "finish_reason": "stop",
            "latency_ms": 100.0,
            "cost": 0.001,
        }
        values.update(overrides)
        return CallRecord(**values)

    def test_totals_accumulate_across_calls(self):
        from ronin.testing.usage_tracker import Totals

        totals = Totals()
        totals.add(self._call(prompt_tokens=10, cost=0.002))
        totals.add(self._call(prompt_tokens=5, cost=0.003))

        self.assertEqual(totals.total_calls, 2)
        self.assertEqual(totals.total_prompt_tokens, 15)
        self.assertAlmostEqual(totals.total_estimated_cost_usd, 0.005)

    def test_a_missing_cost_is_counted_not_treated_as_free(self):
        from ronin.testing.usage_tracker import Totals

        totals = Totals()
        totals.add(self._call(cost=None))

        self.assertEqual(totals.cost_unknown_calls, 1)
        self.assertEqual(totals.total_estimated_cost_usd, 0.0)

    def test_the_report_keeps_the_field_names_already_on_disk(self):
        from ronin.testing.usage_tracker import Totals
        from dataclasses import asdict

        written = asdict(Totals())

        for name in ("total_calls", "total_estimated_cost_usd", "components"):
            self.assertIn(name, written)


class DecisionRecordTests(unittest.TestCase):
    """A decision call reports less than a completion, and names itself differently."""

    def _response(self, request_id="req_01a1", raises=False):
        from types import SimpleNamespace

        class Response(SimpleNamespace):
            @property
            def request_id(self):
                if raises:
                    raise RuntimeError("The response did not include a request ID.")
                return request_id

        return Response(
            model="jev-1.14.0",
            usage=SimpleNamespace(input_tokens=3216, output_tokens=36),
        )

    def _record(self, response):
        from ronin.llm.usage import record_decision

        return record_decision(response, "typesafe/jev-1.14.0", 264.0, 0.000135)

    def test_the_call_is_named_by_the_id_the_service_gave_it(self):
        """A decision response carries no `id`, so reading one stored a blank."""
        self.assertEqual(self._record(self._response()).generation_id, "req_01a1")

    def test_a_response_that_names_no_call_is_still_recorded(self):
        """The SDK raises when the header is absent, and usage must not raise."""
        record = self._record(self._response(raises=True))

        self.assertEqual(record.generation_id, "")
        self.assertEqual(record.total_tokens, 3252)

    def test_the_tokens_a_decision_reports_are_what_is_counted(self):
        record = self._record(self._response())

        self.assertEqual(record.prompt_tokens, 3216)
        self.assertEqual(record.completion_tokens, 36)
        self.assertEqual(record.total_tokens, 3252)
