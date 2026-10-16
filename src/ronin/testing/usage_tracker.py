"""What every model call cost, and what it spent that on.

Totals the token counts, cache counts, latency and cost across a run, and keeps
one row per call carrying the whole response. Counts two kinds of call
separately: answers cut off at the token limit, and questions asked twice after
a schema mismatch.

A response quotes the values a journey declared secret, because the model
writes its reasoning in its own words. The scrubber masks them before anything
reaches disk.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from ronin.llm.usage import CallRecord
from ronin.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)

TRUNCATED = "length"


@dataclass
class Totals:
    """What a run spent, one call at a time.

    These field names go straight into the report. They match the names already
    written to telemetry on disk, so an older file stays readable.
    """

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    cost_unknown_calls: int = 0
    total_cached_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_latency_ms: float = 0.0

    truncated_calls: int = 0
    """Answers cut off at the token limit. They read as malformed, not as wrong."""

    retried_calls: int = 0
    """Questions asked twice, because the first answer did not fit the schema."""

    components: Dict[str, Any] = field(default_factory=dict)

    def add(self, call: CallRecord) -> None:
        self.total_calls += 1
        self.total_prompt_tokens += call.prompt_tokens
        self.total_completion_tokens += call.completion_tokens
        self.total_tokens += call.total_tokens
        self.total_cached_tokens += call.cached_tokens
        self.total_reasoning_tokens += call.reasoning_tokens
        self.total_latency_ms += call.latency_ms or 0.0

        if call.cost is None:
            self.cost_unknown_calls += 1
        else:
            self.total_estimated_cost_usd += call.cost

        if call.finish_reason == TRUNCATED:
            self.truncated_calls += 1
        if call.attempt > 1:
            self.retried_calls += 1


class LLMUsageTracker:
    """Manages and serializes LLM token and cost telemetry during testing."""

    def __init__(
        self, output_dir: Path = Path("assets"), scrubber: SecretScrubber | None = None
    ):
        self.output_dir = output_dir
        self._scrubber = scrubber
        self.totals = Totals()

    def add_call(self, component_name: str, call: CallRecord) -> None:
        """Callback to record a single LLM call for a specific component."""
        self.totals.add(call)
        if call.finish_reason == TRUNCATED:
            logger.warning(
                "Answer truncated at the token limit (%s); it will read as malformed "
                "rather than wrong, so raise the limit before rewording the schema.",
                call.generation_id,
            )

        component = self.totals.components.setdefault(
            component_name, {"models": [], "calls": []}
        )
        if call.model not in component["models"]:
            component["models"].append(call.model)
        component["calls"].append(asdict(call) | {"cost_known": call.cost is not None})

    def _scrubbed(self, value):
        """Masks declared secrets anywhere in the records, before anything writes them."""
        if self._scrubber is None:
            return value
        if isinstance(value, str):
            return self._scrubber.scrub(value)
        if isinstance(value, dict):
            return {key: self._scrubbed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._scrubbed(item) for item in value]
        return value

    def save_report(self) -> Path:
        """Serializes usage records to a JSON file in the assets directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"llm_usage_{timestamp}.json"

        temporary = filepath.with_suffix(filepath.suffix + ".tmp")
        payload = self._scrubbed(asdict(self.totals))
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(filepath)
        logger.info("LLM usage telemetry successfully saved to %s", filepath)
        return filepath
