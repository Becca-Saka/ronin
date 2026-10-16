"""Turning a chosen target into something safe to touch, or refusing to.

In marks mode the planner answers with a number rather than a point, which
makes the answer checkable. A number naming nothing is caught here, before
anything touches the device. It no longer surfaces three steps later, as a tap
on empty space.

Nothing in this module guesses. A target that cannot be resolved stops the step,
and is recorded. A detection gap we count is worth more than one we hide.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ronin.agent.actuation.actions import (
    TARGETED_ACTIONS,
    Action,
    TEXT_FIELD_ACTIONS,
    ActionType,
    Coordinates,
)
from ronin.agent.history import ActionEntry, ActionResult
from ronin.agent.device_planner import GROUNDING_MARKS
from ronin.agent.perception import marks
from ronin.agent.step.staleness import has_target, target_moved
from ronin.errors import ObservationUnavailable

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
    action becomes its coordinate form here, because every mode ends in a point.
    the number only decides who worked that point out.

    In marks mode the number is the only accepted way to name a target. A
    response carrying coordinates instead is refused rather than obeyed.
    Accepting it would reinstate the invented coordinate that the numbering
    exists to prevent, and we cannot tell a read-off coordinate from a guessed
    one.
    """
    if action.type not in TARGETED_ACTIONS:
        return None, None

    index = getattr(action, "mark", None)
    if grounding == GROUNDING_MARKS:
        if index is None:
            return (
                "Not executed. This action needs `mark`. `mark` is the number on "
                "the screenshot. Do not send coordinates. If the element you need "
                "has no number, answer target_not_listed."
            ), None
    elif index is None:
        if getattr(action, "coordinates", None) is None:
            return "Not executed. This action needs `coordinates` of the target.", None
        return None, None

    mark = marks.find(element_marks, index)
    if mark is None:
        available = len(element_marks)
        logger.warning("Element %s was chosen but only 1-%d exist.", index, available)
        return (
            f"Not executed. There is no element {index}. The numbers on this "
            f"screen are 1 to {available}. Choose one of them."
        ), None

    if mark.refuses_text and action.type in TEXT_FIELD_ACTIONS:
        return (
            f"Not executed. Element {index} holds no text. Typing there did not "
            "move the cursor, twice. Choose a different number. If no number is "
            "the field you need, the field is off screen. Close the keyboard "
            "with key_event 4, or scroll. Then look again."
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
    """Why the target is no longer trustworthy, or None while it still is.

    A refusal costs one step and one more look at the screen. That is cheaper
    than touching whatever took the target's place. A screen Ronin cannot read
    is no evidence that the target survived, so it counts too.
    """
    if not has_target(action):
        return None

    try:
        fresh = device_control.get_screenshot()
    except ObservationUnavailable as exc:
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
    unmatched: bool = False
    """The planner asked for something no number names. The loop decides whether
    that ends the scenario, because only the loop knows if the screen changed
    since the last time it was asked."""

    target: Optional[str] = None
    """What the planner said it was looking for, when no number named it."""
    action: Optional[Action] = None
    """The action to run instead, once a number has been turned into a point."""


USABLE = Decision()


def refuse_mark(
    strikes: dict[str, list[marks.Box]],
    refused: dict[str, list[marks.Box]],
    phash: Optional[str],
    index: Optional[int],
    element_marks: Sequence[marks.Mark],
) -> None:
    """Records a failure to put text somewhere, and bans the place on the second.

    Focus that did not move proves the tap reached nothing focusable. It does
    not prove the element holds no text: a tap into the padding of a real field
    does the same. Banning on one failure can make a genuine password field
    untypable for the rest of the screen, so the first failure only leaves a
    mark against the place.

    Kept by place rather than by number. Numbers are handed out in reading
    order over whatever the detector returned, so they do not survive a
    detection appearing or disappearing between two looks at one screen.
    """
    mark = marks.find(element_marks, index) if index is not None else None
    if phash is None or mark is None:
        return

    box = marks.box_of(mark.element)
    first = strikes.setdefault(phash, [])
    if not any(marks.is_same_element(mark.element, seen) for seen in first):
        first.append(box)
        logger.info("Element %d took no text. Trying it once more.", index)
        return

    refused.setdefault(phash, []).append(box)
    logger.info("Element %d took no text twice. Its place will be skipped.", index)


def stalled_reason(decision) -> str:
    """Why a screen that keeps refusing ends the scenario.

    Named from what the last refusal was, so the report says which wall the
    run hit rather than only that it stopped.
    """
    if decision.target:
        return (
            "Target was not among the detected elements, on a screen already "
            f"searched for it: {decision.target}"
        )
    return (
        "No action reached the screen after repeated attempts. Last refusal: "
        f"{decision.observation}"
    )


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
            observation=(
                f"Not executed. No number names this: {action.description}. Only "
                "what is on screen gets a number. A target below the fold has no "
                "number. A target behind the keyboard has no number. Scroll it "
                "into view, or close the keyboard with key_event 4. Then look "
                "again. Answer target_not_listed again if it is still absent."
            ),
            unmatched=True,
            target=action.description,
        )

    unresolved, resolved = resolve_mark(action, snapshot.marks, grounding)
    if unresolved:
        return Decision(ok=False, observation=unresolved)
    action = resolved or action

    stale = stale_target_reason(device_control, snapshot, action)
    if stale:
        return Decision(ok=False, observation=stale)

    return Decision(action=action)
