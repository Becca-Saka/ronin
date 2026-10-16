"""Turning a chosen target into something safe to touch, or refusing to.

In marks mode the planner answers with a number rather than a point. That makes
the answer checkable: a number naming nothing is caught here, before the device
is touched, instead of surfacing three steps later as a tap on empty space.

Nothing in this module guesses. A target that cannot be resolved stops the step
and is recorded, because a detection gap we count is worth more than one we
paper over.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ronin.agent.actuation.actions import TARGETED_ACTIONS, ActionType, Coordinates
from ronin.agent.history import ActionEntry, ActionResult
from ronin.agent.device_planner import GROUNDING_MARKS
from ronin.agent.perception import marks
from ronin.agent.staleness import has_target, target_moved

logger = logging.getLogger(__name__)


def record_unexecuted(history, step: int, action_desc: str, observation: str) -> None:
    """Tells the planner why nothing happened, so it re-plans instead of repeating."""
    history.append(
        ActionEntry(
            step_num=step + 1,
            action_desc=action_desc,
            result=ActionResult(success=False, observation=observation),
        )
    )


def resolve_mark(action, element_marks, grounding: str = "") -> tuple:
    """Turns a chosen number into the point to touch.

    Returns (reason the step cannot proceed, action to run instead). A marks
    action becomes its coordinate form here, because every mode ends in a point;
    the number only decides who worked that point out.

    In marks mode the number is the only accepted way to name a target. A
    response carrying coordinates instead is refused rather than obeyed —
    accepting it would reinstate the invented coordinate the numbering exists
    to prevent, and the model has no way for us to tell a read-off coordinate
    from a guessed one.
    """
    if action.type not in TARGETED_ACTIONS:
        return None, None

    index = getattr(action, "mark", None)
    if grounding == GROUNDING_MARKS:
        if index is None:
            return (
                "Not executed — this action needs `mark`, the number of the element "
                "on the screenshot. Coordinates are not accepted. If the element you "
                "need has no number, use target_not_listed."
            ), None
    elif index is None:
        if getattr(action, "coordinates", None) is None:
            return "Not executed — this action needs `coordinates` of the target.", None
        return None, None

    mark = marks.find(element_marks, index)
    if mark is None:
        available = len(element_marks)
        logger.warning("Element %s was chosen but only 1-%d exist.", index, available)
        return (
            f"Not executed — there is no element {index}. The numbered elements "
            f"are 1 to {available}. Choose one of those."
        ), None

    x, y = mark.center
    # Every marks variant carries this, set by actions._marks_variant.
    resolved = type(action).coordinate_form(
        coordinates=Coordinates(x=x, y=y), **action.model_dump(exclude={"mark", "type"})
    )
    logger.info("Element %s resolves to (%d, %d).", index, x, y)
    return None, resolved


def is_target_not_listed(action) -> bool:
    return getattr(action, "type", None) == ActionType.TARGET_NOT_LISTED


def save_unmatched_screen(media_manager, path: Path, snapshot) -> None:
    """Keeps the screen a target was missing from, so the gap can be studied later.

    Through the media manager, not straight to disk: this file outlives the run
    like any other artifact, so declared secrets on screen get covered first.
    """
    try:
        media_manager.save_redacted_bytes(
            snapshot.raw_img_bytes, snapshot.gui_elements, path
        )
    except Exception as exc:
        logger.warning("Could not save the unmatched screen: %s", exc)


SCREEN_MOVED = (
    "Not executed — the screen changed while this action was being chosen, so "
    "its target was already out of date. Read the current elements and choose "
    "again."
)

SCREEN_UNREADABLE = (
    "Not executed — the screen could not be read again to confirm the target is "
    "still there. Look at the screen again before choosing."
)


def stale_target_reason(device_control, snapshot, action) -> Optional[str]:
    """Why the target can no longer be trusted, or None while it still can be.

    Refusing costs a step and re-perceives, which is cheaper than touching
    whatever has taken the target's place. A screen that cannot be read at all
    is no evidence the target survived, so it counts too.

    The catch is broad because a lost session surfaces in several shapes, and
    every one of them leaves the target unverified.
    """
    if not has_target(action):
        return None

    try:
        fresh = device_control.get_screenshot()
    except Exception as exc:
        logger.warning("Could not re-read the screen before acting: %s", exc)
        return SCREEN_UNREADABLE
    if not fresh:
        logger.warning("The screen re-read before acting came back empty.")
        return SCREEN_UNREADABLE
    if target_moved(snapshot.raw_img_bytes, fresh, action, snapshot.gui_elements):
        return SCREEN_MOVED
    return None


@dataclass(frozen=True)
class Decision:
    """Whether an action may touch the device, and what to say if not."""

    ok: bool = True
    observation: Optional[str] = None
    stop_scenario: Optional[str] = None
    unmatched: bool = False
    action: Optional[object] = None
    """The action to run instead, once a number has been turned into a point."""


USABLE = Decision()


def evaluate(action, snapshot, device_control, grounding: str = "") -> Decision:
    """Everything that can stop an action before the device is touched."""
    if is_target_not_listed(action):
        logger.warning(
            "Target not among the %d detected elements: %s",
            len(snapshot.marks),
            action.description,
        )
        return Decision(
            ok=False,
            observation=f"Target not detected: {action.description}",
            stop_scenario=(
                f"Target was not among the detected elements: {action.description}"
            ),
            unmatched=True,
        )

    unresolved, resolved = resolve_mark(action, snapshot.marks, grounding)
    if unresolved:
        return Decision(ok=False, observation=unresolved)
    action = resolved or action

    stale = stale_target_reason(device_control, snapshot, action)
    if stale:
        return Decision(ok=False, observation=stale)

    return Decision(action=action)
