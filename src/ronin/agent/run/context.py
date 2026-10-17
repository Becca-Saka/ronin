"""What one run carries from step to step.

Each group belongs to a policy that decides when a screen is going nowhere, and
each is keyed by screen, because none of it means anything on a different one.
"""

from dataclasses import dataclass, field
from typing import Optional

from ronin.agent.memory.history import AgentHistory
from ronin.agent.grounding import marks
from ronin.agent.memory.loop_detector import RollingLoopDetector
from ronin.agent.memory.stall import Stall


@dataclass
class RunContext:
    history: AgentHistory
    loop_detector: RollingLoopDetector
    prev_phash: Optional[str] = None
    last_external_nav: bool = False
    drift_warning: Optional[str] = None
    steps_executed: int = 0
    current_phash: Optional[str] = None
    prev_element_count: Optional[int] = None
    keyboard_open: Optional[bool] = None
    last_action_type: Optional[str] = None
    screen_height: Optional[int] = None
    pending_loop_nudge: Optional[str] = None
    stall: Stall = field(default_factory=Stall)
    """Steps in a row that never reached the device, and where."""

    keyboard_dismissed: set[str] = field(default_factory=set)
    """Screens whose keyboard was already pulled down once to look behind it."""

    recovered: set[str] = field(default_factory=set)
    """Screens already handed to a recovery planner."""

    recovery_looks: int = 0
    """Model calls recovery made, counted apart from the run's own steps.

    Adding them to steps_executed would mix two counts: the loop numbers its
    history by step, and a scenario that was rescued would appear to have run
    steps it never ran.
    """

    text_strikes: dict[str, list[marks.Box]] = field(default_factory=dict)
    """Per screen, elements that took no text once. One is not yet proof.

    Focus that did not move means the tap reached nothing focusable. A picture
    of a field does that, and so does a tap into the padding of a real one. A
    second failure at the same place separates them.
    """

    refused_marks: dict[str, list[marks.Box]] = field(default_factory=dict)
    """Per screen, where the elements that took no text sit.

    Held by place, not by number: see marks.is_same_element.
    """
