"""Refusing a run that cannot have what its settings ask for.

A missing key, a missing extra and a grounding mode that needs one are all
answerable before a device is opened. Asked here, together, so that two
missing things cost one message rather than two boot cycles.
"""

from typing import Optional

from ronin.agent.grounding import GROUNDING_COORDINATES, GROUNDING_MARKS
from ronin.defaults import DEFAULT_GROUNDING
from ronin.errors import UsageError
from ronin.llm.credentials import missing_credentials
from ronin.utils.vision_deps import INSTALL_HINT as VISION_HINT
from ronin.utils.vision_deps import missing_vision_packages

DETECTED_GROUNDING = frozenset({GROUNDING_COORDINATES, GROUNDING_MARKS})
"""Modes that choose a target from detected elements, and so need a detector."""

CREDENTIAL_HINT = "Add them to a .env file in this directory, or export them."


def missing_for_grounding(grounding: str) -> list[str]:
    """What a grounding mode needs and does not have, named once with its fix."""
    if grounding not in DETECTED_GROUNDING:
        return []
    absent = missing_vision_packages()
    if not absent:
        return []
    return [
        f"grounding={grounding} reads detected elements, which needs "
        f"{', '.join(absent)}.\n" + VISION_HINT.replace("\n", "\n  ")
    ]


def missing_for_models(*models: Optional[str]) -> list[str]:
    """The credentials these models need and do not have."""
    wanted = {
        name: model for model in models for name in missing_credentials(model)
    }
    if not wanted:
        return []
    named = ", ".join(f"{name} (for {model})" for name, model in wanted.items())
    return [f"Missing credentials: {named}.\n  {CREDENTIAL_HINT}"]


def require_run(
    model: Optional[str],
    fast_model: Optional[str] = None,
    grounding: str = DEFAULT_GROUNDING,
) -> None:
    """Refuses to start when a run cannot have what its settings ask for.

    Every missing thing is named at once. Fixing them one error at a time
    costs a device boot for each.
    """
    missing = missing_for_models(model, fast_model) + missing_for_grounding(grounding)
    if missing:
        raise UsageError("This run cannot start:\n\n" + "\n\n".join(missing))
