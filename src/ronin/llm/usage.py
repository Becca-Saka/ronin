"""One completion, recorded.

Reads a provider response into a CallRecord. The token counts, cache counts and
cost come from the usage block. The finish reason, latency, served model and
generation id sit outside it. The record also keeps the whole response, which
costs about a kilobyte and does not repeat the prompt.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CallRecord:
    """One completion, in the form the tracker stores."""

    model: str
    """The model asked for."""

    served_model: str
    """The model the provider actually used, which a gateway may choose."""

    generation_id: str
    """The provider's handle for this call. Traces it in their dashboard."""

    attempt: int
    """1 unless a schema mismatch forced the question to be asked again."""

    finish_reason: str
    """"length" means the answer ran out of room. It then reads as a bad answer."""

    latency_ms: float
    cost: Optional[float]

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    prompt_image_tokens: Optional[int] = None
    prompt_text_tokens: Optional[int] = None

    cost_details: dict = field(default_factory=dict)

    raw: dict = field(default_factory=dict)
    """The response as the provider sent it, whole."""


def _detail(block: dict, name: str, default=0):
    value = (block or {}).get(name)
    return default if value is None else value


def record(response, model: str, attempt: int) -> CallRecord:
    """Reads a provider response into a CallRecord."""
    usage = response.usage.model_dump()
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    hidden = response._hidden_params or {}

    return CallRecord(
        model=model,
        served_model=response.model,
        generation_id=response.id,
        attempt=attempt,
        finish_reason=response.choices[0].finish_reason,
        latency_ms=hidden.get("_response_ms"),
        cost=usage.get("cost"),
        prompt_tokens=_detail(usage, "prompt_tokens"),
        completion_tokens=_detail(usage, "completion_tokens"),
        total_tokens=_detail(usage, "total_tokens"),
        cached_tokens=_detail(prompt_details, "cached_tokens"),
        cache_write_tokens=_detail(prompt_details, "cache_write_tokens"),
        reasoning_tokens=_detail(completion_details, "reasoning_tokens"),
        prompt_image_tokens=_detail(prompt_details, "image_tokens", None),
        prompt_text_tokens=_detail(prompt_details, "text_tokens", None),
        cost_details=usage.get("cost_details") or {},
        raw=response.model_dump(),
    )
