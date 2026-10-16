"""Changing a screen until a blocked step becomes possible.

The main planner works through a scenario. When a step cannot be performed —
the field is behind the keyboard, under an overlay, or below the fold — asking
that planner again produces the same answer, because it is answering the same
question against the same screen.

So a second, narrower planner takes over. It is given one job: make the thing
reachable. It never performs the step itself, which is what keeps the handback
simple. Nothing it does can be mistaken for scenario progress, so the main
planner resumes exactly where it left off and no reconciliation is needed.

It is bounded. A recovery that cannot find the target in a few moves has not
understood the screen, and more moves spend a scenario's budget on guessing.
"""

import logging
from typing import List, Optional

from ronin.agent.actuation.actions import ActionType
from ronin.agent.actuation.orchestrator import ActionOrchestrator
from ronin.agent.device_planner import DevicePlanner, DoneAction
from ronin.agent.history import ActionEntry, ActionResult
from ronin.agent.step import observation, targeting

logger = logging.getLogger(__name__)

REVEAL_STEPS = 4
"""Moves a recovery may make before the screen is declared hopeless.

It gets one look more than it has moves, so the last move is judged like every
other one. Without it a final scroll could expose the target and the scenario
would still be failed for never having seen it.
"""

BACK = 4

PERMITTED = frozenset({ActionType.SWIPE, ActionType.WAIT, ActionType.KEY_EVENT})
"""Everything recovery may do, listed rather than excluded.

Recovery runs on its own, with nobody reading the screen it is working. So it
is held to moves that cannot commit anything: scrolling moves the viewport,
waiting does nothing, and back closes what is on top.

Tapping is not here. A tap is only as safe as the thing under it, and nothing
tells us what that is — the screen is pixels, and the control we would be
reaching for sits on a checkout page. "Never submit" in a prompt is a request,
not a guard. The cost is that recovery cannot press Dismiss on an overlay.
Back closes most of them anyway.
"""

MAX_RECOVERIES = 3
"""Rescues one scenario may have, across all of its screens.

Capped per screen alone, a long journey could be rescued at every one of them
and spend many times the budget it was given.
"""

SAFE_KEYS = frozenset({BACK})
"""Keys that close something. Enter submits and home leaves the app.

Back is safe only over the app's own screens. Over a system permission prompt
it answers "deny", so recovery stands down entirely while one is up rather
than reasoning about which key is safe under it.
"""


class RevealPlanner(DevicePlanner):
    """Chooses moves that change what the screen offers, and nothing else."""

    def _action_vocabulary(self) -> str:
        return """
### Allowed Action Types
Each action must have a `type` field set to one of:
- "swipe": Scroll the screen. Fields: `type`, `start_x`, `start_y`, `end_x`, `end_y`, `reason`.
- "key_event": Close what is on top. Fields: `type`, `key` (must be 4), `reason`.
- "wait": Let the screen settle. Fields: `type`, `duration` (float seconds), `reason`.
- "done": The target is on screen and ready. Fields: `type`, `reason`.

There is no tap in this list. You cannot press anything. A press commits an
action, and nobody is watching this screen.
"""

    def _action_constraints(self) -> str:
        return """
CRITICAL CONSTRAINTS:
1. Do not carry out the step. Change what is on screen. Someone else carries
   out the step after you.
2. Answer "done" as soon as the target is on screen and ready. Do not confirm
   it. Do not tidy up. Do not continue.
3. To show something below the fold, swipe up. Start low on the screen. End
   high on the screen.
4. The soft keyboard hides the lower half of the screen. Close it with
   key_event 4.
5. A dialog also hides the screen. Close it with key_event 4.
6. Answer "done" if the target is already on screen. The screen may have
   changed since the step was blocked.
"""


def _permitted(action) -> bool:
    """Whether recovery may run this, judged on what it can commit."""
    if action.type not in PERMITTED:
        return False
    if action.type == ActionType.KEY_EVENT:
        return action.key in SAFE_KEYS
    return True


def reveal(
    target: str,
    planner: RevealPlanner,
    device_control,
    perception,
    grounding: str,
    scrubber,
    screen_height: int,
) -> tuple[Optional[str], int]:
    """Works the screen until `target` is reachable.

    Returns what it did and how many looks it took. None for the first when
    the target was not reached, which leaves the caller's own decision about
    the scenario untouched. The count is the caller's to charge: these are
    model calls the scenario paid for and its budget has to see them.
    """
    looks = 0
    goal = (
        f"A step cannot run. Its target is not available: {target}. Change what "
        "is on screen to make the target available."
    )
    history: List[str] = []

    for move in range(REVEAL_STEPS + 1):
        # Every move recovery has closes something, and closing a permission
        # prompt answers it. The planner is told to answer these itself.
        if device_control.showing_system_prompt():
            logger.info("A system prompt is up. Recovery has nothing safe to do.")
            return None, looks

        # Back closes what is on top, and on a root screen what is on top is
        # the app. Carrying on would work the launcher for the moves left.
        if not device_control.is_still_in_sut():
            logger.info("Recovery left the app. Handing back for drift recovery.")
            return None, looks

        looks += 1
        snapshot = observation.capture(perception, None, None)
        observation.number_targets(
            snapshot,
            _Numbering(screen_height, device_control.is_keyboard_open()),
            grounding,
        )

        response = planner.plan_step(
            goal=goal,
            gui_elements=snapshot.gui_elements,
            history="\n".join(history),
            current_image_bytes=snapshot.planner_img_bytes,
            marks=snapshot.marks,
            screen_height=screen_height,
        )
        action = response.action

        if isinstance(action, DoneAction):
            logger.info("Recovery reports the target is reachable: %s", response.reasoning)
            found = _note(history) if history else "Recovery found the target already reachable."
            return found, looks

        if move == REVEAL_STEPS:
            break

        if not _permitted(action):
            logger.warning("Recovery proposed %s. It may not do that.", action.type)
            history.append(f"{action.type.value} was refused: recovery may not do that.")
            continue

        decision = targeting.evaluate(action, snapshot, device_control, grounding)
        if not decision.ok:
            history.append(f"{action.type.value} was refused: {decision.observation}")
            continue

        runnable = decision.action if decision.action is not None else action
        try:
            ActionOrchestrator(device_control, scrubber).execute(runnable)
            history.append(f"{action.type.value}: {action.reason}")
        except Exception as exc:  # noqa: BLE001 - device boundary
            logger.warning("Recovery step failed: %s", exc)
            history.append(f"{action.type.value} failed: {exc}")

    logger.info("Recovery spent %d moves without reaching the target.", REVEAL_STEPS)
    return None, looks


def hand_over(
    target: str,
    llm,
    use_multimodal_vision: bool,
    grounding: str,
    scrubber,
    device_control,
    perception,
    ctx,
) -> bool:
    """Gives a spent screen to a planner whose only job is to open it up.

    Once per screen. A recovery that did not work will not work twice, and the
    scenario has a budget to protect.
    """
    screen = ctx.current_phash
    if screen is None or screen in ctx.recovered:
        return False
    if len(ctx.recovered) >= MAX_RECOVERIES:
        logger.info("A scenario may be rescued %d times. That is spent.", MAX_RECOVERIES)
        return False
    ctx.recovered.add(screen)

    try:
        note, looks = reveal(
            target=target,
            planner=RevealPlanner(llm, use_multimodal_vision, grounding),
            device_control=device_control,
            perception=perception,
            grounding=grounding,
            scrubber=scrubber,
            screen_height=ctx.screen_height or 0,
        )
    except Exception as exc:  # noqa: BLE001 - a rescue must not make things worse
        # Without this the scenario reports an execution error instead of the
        # stalled screen it actually hit, and the reason it stalled is lost.
        logger.warning("Recovery itself failed: %s", exc)
        return False
    # Counted whether or not it worked: the calls were made either way, and a
    # run that hides them looks cheaper than it was.
    ctx.recovery_looks += looks

    if note is None:
        return False

    ctx.history.append(
        ActionEntry(
            step_num=ctx.steps_executed,
            action_desc="recovery",
            result=ActionResult(success=True, observation=note),
        )
    )
    ctx.stall.action_dispatched()
    return True


def _note(history: List[str]) -> str:
    return "Recovery reached the target by: " + ". ".join(history)


class _Numbering:
    """The slice of run context that numbering reads.

    The keyboard state comes along so its keys stay unnumbered, exactly as in
    the main loop. Eighty numbered keys would bury the handful of controls
    recovery is looking at.

    Recovery keeps no refusals of its own: it does not type, so it learns
    nothing about which elements hold text.
    """

    def __init__(self, screen_height: int, keyboard_open):
        self.keyboard_open = keyboard_open
        self.screen_height = screen_height
        self.refused_marks: dict = {}


def reveal_behind_keyboard(device_control, snapshot, ctx) -> bool:
    """Pulls the keyboard down once per screen when a step cannot proceed.

    The keyboard hides the lower half, and only what is on screen gets a
    number, so a target the planner cannot name is often underneath it.
    Done rather than asked for, because a planner told to scroll does not
    reliably do it.

    Once per screen. If the keyboard returns and the step fails again,
    something other than the keyboard is wrong.
    """
    screen = snapshot.current_phash
    if not device_control.is_keyboard_open() or screen is None:
        return False
    if screen in ctx.keyboard_dismissed:
        return False

    logger.info("Step blocked with the keyboard open. Closing it to see the rest.")
    if not device_control.dismiss_keyboard():
        return False
    ctx.keyboard_dismissed.add(screen)
    return True
