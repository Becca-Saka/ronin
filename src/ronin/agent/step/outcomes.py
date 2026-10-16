"""How a run ends, and the one thing to do on the way out.

A run stops for one of five reasons:

- the goal was reached
- a hook said stop
- the steps ran out
- the screen stopped changing
- the plan stopped changing

Each ending records the step count and the history, and flushes the trail. They
are built here rather than at each exit.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from ronin.agent.device_planner import DeviceStepResponse, DoneAction
from ronin.agent.history import AgentHistory
from ronin.agent.step.observation import PerceptionSnapshot
from ronin.agent.trail import SessionTrail

logger = logging.getLogger(__name__)

StepHookFn = Callable[[DeviceStepResponse, PerceptionSnapshot], Optional[bool]]
"""Called after planning and before acting, with the plan and its screen.

The screen comes too. A hook that sees only the plan has to take the planner's
word for what the screen holds.
"""


@dataclass
class DeviceRunResult:
    """Outcome of a DeviceAgent.run() call."""

    status: Literal[
        "goal_reached", "step_limit", "loop_detected", "hook_stop", "exception"
    ]
    steps_executed: int
    failure_reason: Optional[str]
    history: AgentHistory


def flush_trail(
    session_trail: Optional[SessionTrail],
    current_phash: Optional[str],
) -> None:
    if session_trail and current_phash:
        session_trail.flush_transition(current_phash)


def loop_detected(
    ctx,
    session_trail: Optional[SessionTrail],
    current_phash: Optional[str],
    failure_reason: str,
) -> DeviceRunResult:
    flush_trail(session_trail, current_phash)
    return DeviceRunResult(
        status="loop_detected",
        steps_executed=ctx.steps_executed,
        failure_reason=failure_reason,
        history=ctx.history,
    )


def step_limit(
    limit: int,
    ctx,
    session_trail: Optional[SessionTrail],
) -> DeviceRunResult:
    logger.warning("Reached max step limit (%d).", limit)
    flush_trail(session_trail, ctx.current_phash)
    return DeviceRunResult(
        status="step_limit",
        steps_executed=ctx.steps_executed,
        failure_reason=f"Reached maximum step limit ({limit}) before goal was completed.",
        history=ctx.history,
    )


def page_stagnation(
    ctx,
    session_trail: Optional[SessionTrail],
    current_phash: Optional[str],
) -> Optional[DeviceRunResult]:
    if not ctx.loop_detector.terminated():
        return None

    logger.error("Page stagnation limit reached. Terminating.")
    return loop_detected(
        ctx,
        session_trail,
        current_phash,
        "Agent stuck: screen did not change for too many consecutive steps.",
    )


def action_repetition(
    ctx,
    session_trail: Optional[SessionTrail],
    current_phash: Optional[str],
) -> Optional[DeviceRunResult]:
    if not ctx.loop_detector.terminated():
        return None

    logger.error("Action repetition limit reached. Terminating.")
    return loop_detected(
        ctx,
        session_trail,
        current_phash,
        "Agent stuck in an execution loop: repeated the same action too many times.",
    )


def hook_stop(
    step_hook: StepHookFn,
    step_response: DeviceStepResponse,
    snapshot: PerceptionSnapshot,
    session_trail: Optional[SessionTrail],
    current_phash: Optional[str],
    ctx,
) -> Optional[DeviceRunResult]:
    if step_hook(step_response, snapshot) is not False:
        return None

    flush_trail(session_trail, current_phash)
    return DeviceRunResult(
        status="hook_stop",
        steps_executed=ctx.steps_executed,
        failure_reason="Execution stopped by step hook (crash, infeasibility, or critical issue).",
        history=ctx.history,
    )


def goal_reached(
    stop_when_satisfied: bool,
    step_response: DeviceStepResponse,
    session_trail: Optional[SessionTrail],
    current_phash: Optional[str],
    ctx,
) -> Optional[DeviceRunResult]:
    satisfied = stop_when_satisfied and getattr(
        step_response, "goal_satisfied", False
    )
    if not isinstance(step_response.action, DoneAction) and not satisfied:
        return None

    if isinstance(step_response.action, DoneAction):
        reason = step_response.action.reason
    else:
        reason = getattr(step_response, "goal_evidence", "") or "goal observed as complete"
    logger.info("Goal reached: %s", reason)
    flush_trail(session_trail, current_phash)
    return DeviceRunResult(
        status="goal_reached",
        steps_executed=ctx.steps_executed,
        failure_reason=None,
        history=ctx.history,
    )
