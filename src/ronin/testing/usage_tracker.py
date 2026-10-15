import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMUsageTracker:
    """Manages and serializes LLM token and cost telemetry during testing."""

    def __init__(self, output_dir: Path = Path("assets")):
        self.output_dir = output_dir
        self.usage_records: Dict[str, Any] = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_estimated_cost_usd": 0.0,
            "cost_unknown_calls": 0,
            "components": {}
        }

    def add_call(
        self, component_name: str, model: str, cost: Optional[float], usage: Any
    ):
        """Callback to record a single LLM call for a specific component."""
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        self.usage_records["total_calls"] += 1
        self.usage_records["total_prompt_tokens"] += prompt_tokens
        self.usage_records["total_completion_tokens"] += completion_tokens
        self.usage_records["total_tokens"] += total_tokens
        if cost is None:
            self.usage_records["cost_unknown_calls"] += 1
        else:
            self.usage_records["total_estimated_cost_usd"] += cost

        if component_name not in self.usage_records["components"]:
            self.usage_records["components"][component_name] = {
                "models": [],
                "calls": []
            }
        models = self.usage_records["components"][component_name]["models"]
        if model not in models:
            models.append(model)

        self.usage_records["components"][component_name]["calls"].append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "cost_known": cost is not None,
        })

    def save_report(self) -> Path:
        """Serializes usage records to a JSON file in the assets directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"llm_usage_{timestamp}.json"
        
        temporary = filepath.with_suffix(filepath.suffix + ".tmp")
        temporary.write_text(json.dumps(self.usage_records, indent=2), encoding="utf-8")
        temporary.replace(filepath)
        logger.info("LLM usage telemetry successfully saved to %s", filepath)
        return filepath
