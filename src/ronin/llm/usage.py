"""What one completion is worth remembering about itself.

The provider reports far more than a token count, and most of it answers a
question we would otherwise guess at: whether a prompt was cached, how much of
it was the screenshot, whether an answer was cut off rather than finished, and
which model actually served a request that named a family.

Collected at the call so the parts that are not on the usage block — the
finish reason, the latency, the generation id — are still reachable.

The whole response is kept alongside the fields picked out of it. A completion
is about a kilobyte and does not echo the prompt back, so there is no reason to
decide now which parts a later question will want.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CallRecord:
    """One completion, as it should be written down."""

    model: str
    """The model asked for."""

    served_model: str
    """The model the provider actually used, which a gateway may choose."""

    generation_id: str
    """The provider's handle for this call, for tracing it in their dashboard."""

    attempt: int
    """1 unless a schema mismatch forced the question to be asked again."""

    finish_reason: str
    """"length" means the answer was truncated, which looks like a bad answer."""

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
    """The response as the provider sent it, so nothing has to be guessed later."""


def _detail(block: dict, name: str, default=0):
    value = (block or {}).get(name)
    return default if value is None else value


def record(response, model: str, attempt: int) -> CallRecord:
    """Reads a completion into the record we keep of it."""
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
