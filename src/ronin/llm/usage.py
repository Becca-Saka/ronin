"""One model call, recorded, whichever route it took.

Reads a provider response into a CallRecord. The token counts, cache counts and
cost come from the usage block. The finish reason, latency, served model and
generation id sit outside it. The record also keeps the whole response, which
costs about a kilobyte and does not repeat the prompt.

Completions and decisions have separate readers and one record, so a run's cost
stays one number.
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
    """"length" means the answer ran out of room. It then reads as a bad answer.

    Empty on a decision: a typed answer cannot run out of room.
    """

    latency_ms: Optional[float]
    """How long the call took, or None where the provider did not report it.

    Optional because a route that reports no timing leaves this unknown, and
    a zero would read as an instant answer. The tracker sums it as zero.
    """

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


def _count(block: dict, name: str) -> int:
    """A token count, as zero where the provider reported none.

    Zero is the honest reading for a count: a block that does not break out
    reasoning tokens is reporting that none were spent on them.
    """
    value = (block or {}).get(name)
    return value if isinstance(value, int) else 0


def _optional_count(block: dict, name: str) -> Optional[int]:
    """A token count a provider may not break out at all.

    None rather than zero, because an image-token count nothing reported is
    not a call that sent no image.
    """
    value = (block or {}).get(name)
    return value if isinstance(value, int) else None


def _milliseconds(block: dict, name: str) -> Optional[float]:
    """A duration the provider may not report, as a number or None."""
    value = (block or {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


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
        latency_ms=_milliseconds(hidden, "_response_ms"),
        cost=usage.get("cost"),
        prompt_tokens=_count(usage, "prompt_tokens"),
        completion_tokens=_count(usage, "completion_tokens"),
        total_tokens=_count(usage, "total_tokens"),
        cached_tokens=_count(prompt_details, "cached_tokens"),
        cache_write_tokens=_count(prompt_details, "cache_write_tokens"),
        reasoning_tokens=_count(completion_details, "reasoning_tokens"),
        prompt_image_tokens=_optional_count(prompt_details, "image_tokens"),
        prompt_text_tokens=_optional_count(prompt_details, "text_tokens"),
        cost_details=usage.get("cost_details") or {},
        raw=response.model_dump(),
    )


def _request_id(response) -> str:
    """The id the decision service gave this call, or nothing when it gave none.

    Asked for by its own name, because a decision response carries no `id` and
    reading one stored a blank for every call. It is the only handle on a
    single call, so it is what a question to the vendor has to quote.

    Guarded because the SDK raises when the header is absent, and telemetry
    must not be the thing that ends a grounding call.
    """
    try:
        return response.request_id or ""
    except Exception:  # noqa: BLE001 - vendor boundary
        return ""


def record_decision(
    response, model: str, latency_ms: float, cost: Optional[float]
) -> CallRecord:
    """Reads a decision response into the same record a completion produces.

    A decision reports token counts and nothing else, so latency is measured by
    the caller and cost comes from the model's price. The rest keeps its
    default.
    """
    usage = response.usage
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0

    return CallRecord(
        model=model,
        served_model=getattr(response, "model", model),
        generation_id=_request_id(response),
        attempt=1,
        finish_reason="",
        latency_ms=latency_ms,
        cost=cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        raw=response.model_dump() if hasattr(response, "model_dump") else {},
    )
