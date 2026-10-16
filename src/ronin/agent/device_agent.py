"""
DeviceAgent — core ReAct loop for LLM-driven mobile device control.
No QA concerns. Accepts an optional planner override so the testing layer
can inject QAPlanner without re-implementing the loop.
"""

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import imagehash
from PIL import Image

from ronin.agent.actuation.actions import ActionType
from ronin.agent.actuation.orchestrator import ActionOrchestrator
from ronin.agent.device.device_session import DeviceSession
from ronin.agent.device.platform import Platform
from ronin.agent.device_planner import (
    GROUNDING_COORDINATES,
    GROUNDING_MARKS,
    DevicePlanner,
    DeviceStepResponse,
    DoneAction,
)
from ronin.agent.history import ActionEntry, ActionResult, AgentHistory
from ronin.agent.launch import restart_app
from ronin.defaults import SETTLE_POLL_INTERVAL, SETTLE_STABLE_READINGS
from ronin.agent.loop_detector import RollingLoopDetector
from ronin.agent.replay import ReplayLedger, replay_known_path, target_label
from ronin.agent import targeting
from ronin.utils.secrets import SecretScrubber
from ronin.agent.perception import marks
from ronin.agent.perception.marks import DETAIL_NUMBERS
from ronin.agent.perception.media_manager import MediaManager
from ronin.agent.perception.perception import ScreenPerception
from ronin.agent.perception.scan_dump import (
    ScanInfo,
    build_screen_scan_record,
    write_screen_scan_dump,
)
from ronin.agent.trail import SessionTrail
from ronin.llm.llm_client import LLMClient
from ronin.utils.redactor import redact_text

logger = logging.getLogger(__name__)

StepHookFn = Callable[[DeviceStepResponse], Optional[bool]]
"""
Called after each planning step, before action execution.
Return False to request early loop termination (e.g. crash detected, infeasible scenario).
Return None or True to continue normally.
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


@dataclass
class _PerceptionSnapshot:
    gui_elements: list
    marks: list
    raw_img_bytes: bytes
    planner_img_bytes: bytes
    annotated_img: Image.Image
    current_phash: Optional[str]
    timestamp: str
    scan_info: Optional[ScanInfo] = None
    prev_phash: Optional[str] = None


@dataclass
class _RunContext:
    history: AgentHistory
    loop_detector: RollingLoopDetector
    visual_history: list[bytes] = field(default_factory=list)
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


class DeviceAgent:
    # Class-level so an agent built without __init__ still answers sensibly.
    grounding = GROUNDING_COORDINATES

    """
    LLM-driven ReAct agent for mobile device control.

    Manages the core loop: perceive → plan → execute → observe.
    The testing layer wraps this via QAAgent, injecting a QAPlanner and
    collecting QA outputs through the step_hook callback.
    """

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
        max_steps: int = 40,
        settle_time: float = 2.0,
        use_multimodal_vision: bool = False,
        visual_history_limit: int = 3,
        headless: bool = True,
        fast_model: str = None,
        planner: Optional[DevicePlanner] = None,
        stop_when_satisfied: bool = False,
        keep_device: bool = False,
        reuse_app: bool = False,
        replay_ledger: Optional[ReplayLedger] = None,
        grounding: str = GROUNDING_COORDINATES,
        mark_detail: str = DETAIL_NUMBERS,
    ):
        self.device_name = device_name
        self.app_path = app_path
        self.platform = platform
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.settle_time = settle_time
        self.visual_history_limit = visual_history_limit
        self.headless = headless
        self.use_multimodal_vision = use_multimodal_vision
        self.grounding = grounding
        self.mark_detail = mark_detail
        self.stop_when_satisfied = stop_when_satisfied
        self.keep_device = keep_device
        self.reuse_app = reuse_app
        self.replay_ledger = replay_ledger
        self.scrubber = SecretScrubber()

        self.llm = LLMClient(model=model)
        self.fast_llm = (
            LLMClient(model=fast_model)
            if fast_model and fast_model != model
            else self.llm
        )

        if planner is not None:
            self._default_planner = planner
            self._default_fast_planner = planner
        else:
            self._default_planner = DevicePlanner(
                self.llm, use_multimodal_vision, grounding, mark_detail
            )
            self._default_fast_planner = DevicePlanner(
                self.fast_llm, use_multimodal_vision, grounding, mark_detail
            )

        self.media_manager = MediaManager(self.scrubber)

    def run(
        self,
        goal: str | list[str],
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
        step_hook: Optional[StepHookFn] = None,
        planner_override: Optional[DevicePlanner] = None,
        max_steps: Optional[int] = None,
    ) -> DeviceRunResult:
        """
        Execute the ReAct loop until the goal is reached, the step limit is hit,
        a loop is detected, the step_hook requests a stop, or an exception occurs.

        planner_override replaces the default planner for this run only — used by
        QAAgent to inject a QAPlanner without re-instantiating the agent.
        """
        planner = planner_override or self._default_fast_planner
        session_dir.mkdir(parents=True, exist_ok=True)
        ctx = _RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            screen_height=getattr(device_control, "screen_height", None),
        )

        try:
            self._restart_app(device_control)
            replay_known_path(
                ledger=self.replay_ledger,
                device_control=device_control,
                goal=goal,
                history=ctx.history,
                quick_phash=self._quick_phash,
                settle=self._settle,
                scrubber=self.scrubber,
            )
            self._step_limit = max_steps or self.max_steps
            for step in range(self._step_limit):
                result = self._run_step(
                    step=step,
                    goal=goal,
                    planner=planner,
                    device_control=device_control,
                    perception=perception,
                    session_dir=session_dir,
                    session_trail=session_trail,
                    step_hook=step_hook,
                    ctx=ctx,
                )
                if result is not None:
                    return result

            return self._step_limit_result(ctx, session_trail)

        except Exception as exc:
            safe_error = self.scrubber.scrub(redact_text(str(exc)))
            logger.error(
                "Device agent run failed (%s): %s", type(exc).__name__, safe_error
            )
            return DeviceRunResult(
                status="exception",
                steps_executed=ctx.steps_executed,
                failure_reason=f"Execution exception: {safe_error}",
                history=ctx.history,
            )

    def _run_step(
        self,
        step: int,
        goal: str | list[str],
        planner: DevicePlanner,
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail],
        step_hook: Optional[StepHookFn],
        ctx: _RunContext,
    ) -> Optional[DeviceRunResult]:
        ctx.steps_executed += 1
        logger.info(
            "--- [DeviceAgent] Step %d / %d ---", step + 1, self.max_steps
        )

        ctx.drift_warning = self._check_context_drift(
            device_control, ctx.last_external_nav
        )

        snapshot = self._capture_perception(
            perception, session_trail, ctx.prev_phash
        )
        ctx.current_phash = snapshot.current_phash
        self._save_step_artifacts(session_dir, step, snapshot)

        needs_keyboard_check = (
            snapshot.scan_info and snapshot.scan_info.reason in ("navigation", "layout_shift")
        ) or ctx.last_action_type == ActionType.TYPE.value
        if needs_keyboard_check:
            detected = device_control.is_keyboard_open()
            if detected is not None:
                ctx.keyboard_open = detected

        reaction = self._build_reaction(snapshot.scan_info, ctx.prev_element_count, ctx.keyboard_open)
        ctx.history.update_last_reaction(reaction)
        ctx.prev_element_count = snapshot.scan_info.element_count if snapshot.scan_info else None

        page_nudge = ctx.loop_detector.check(None, snapshot.current_phash)
        stagnation_result = self._page_stagnation_result(
            ctx, session_trail, snapshot.current_phash
        )
        if stagnation_result is not None:
            return stagnation_result

        loop_nudge = "\n".join(
            filter(None, [ctx.pending_loop_nudge, page_nudge])
        ) or None
        ctx.pending_loop_nudge = None

        step_response = self._plan_step(
            planner=planner,
            goal=goal,
            snapshot=snapshot,
            ctx=ctx,
            session_trail=session_trail,
            loop_nudge=loop_nudge,
        )
        self._apply_plan_result(step_response, snapshot, session_trail, ctx)

        if step_hook:
            hook_result = self._check_step_hook(
                step_hook, step_response, session_trail, snapshot.current_phash, ctx
            )
            if hook_result is not None:
                return hook_result

        done_result = self._check_done_action(
            step_response, session_trail, snapshot.current_phash, ctx
        )
        if done_result is not None:
            return done_result

        action = step_response.action
        action_nudge = ctx.loop_detector.check(
            RollingLoopDetector.hash_action(action), None
        )
        loop_result = self._action_repetition_result(
            ctx, session_trail, snapshot.current_phash
        )
        if loop_result is not None:
            return loop_result

        ctx.pending_loop_nudge = action_nudge

        decision = targeting.evaluate(
            action, snapshot, device_control, self.grounding
        )
        if not decision.ok:
            targeting.record_unexecuted(
                ctx.history, step, self._describe_action(action), decision.observation
            )
            if decision.unmatched:
                targeting.save_unmatched_screen(
                    self.media_manager,
                    session_dir / f"not_listed_step{step + 1}_{snapshot.timestamp}.png",
                    snapshot,
                )
            if decision.stop_scenario:
                return self._loop_detected_result(
                    ctx, session_trail, snapshot.current_phash, decision.stop_scenario
                )
            return None

        action = decision.action or action
        self._save_action_screenshot(session_dir, step, snapshot, action)
        ctx.last_external_nav = self._execute_and_record(
            step=step,
            action=action,
            step_response=step_response,
            device_control=device_control,
            snapshot=snapshot,
            session_trail=session_trail,
            ctx=ctx,
        )

        ctx.prev_phash = snapshot.current_phash
        return None

    def _capture_perception(
        self,
        perception: ScreenPerception,
        session_trail: Optional[SessionTrail],
        prev_phash: Optional[str],
    ) -> _PerceptionSnapshot:
        gui_elements, raw_img, annotated_img = perception.perceive()
        current_phash = (
            str(perception.current_hash) if perception.current_hash else None
        )
        logger.info(
            "Perceived %d elements. pHash: %s", len(gui_elements), current_phash
        )

        if session_trail and current_phash:
            if prev_phash and current_phash != prev_phash:
                session_trail.flush_transition(current_phash)
            session_trail.record_screen(current_phash, "", gui_elements)

        buf = io.BytesIO()
        raw_img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        return _PerceptionSnapshot(
            gui_elements=gui_elements,
            marks=[],
            raw_img_bytes=raw_bytes,
            planner_img_bytes=raw_bytes,
            annotated_img=annotated_img,
            current_phash=current_phash,
            timestamp=time.strftime("%Y%m%d_%H%M%S"),
            scan_info=perception.last_scan_info,
            prev_phash=prev_phash,
        )

    def _save_step_artifacts(
        self,
        session_dir: Path,
        step: int,
        snapshot: _PerceptionSnapshot,
    ) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        base = f"step{step + 1}_{snapshot.timestamp}"
        annotated_path = session_dir / f"{base}.png"
        raw_path = session_dir / f"{base}_raw.png"
        scan_path = session_dir / f"{base}_scan.json"

        self.media_manager.save_redacted_pair(
            snapshot.annotated_img, snapshot.raw_img_bytes,
            snapshot.gui_elements, annotated_path, raw_path,
        )

        if snapshot.scan_info is None:
            return

        record = build_screen_scan_record(
            step=step + 1,
            timestamp=snapshot.timestamp,
            scan_info=snapshot.scan_info,
            gui_elements=snapshot.gui_elements,
            screenshot_paths={
                "raw": raw_path.name,
                "annotated": annotated_path.name,
                "scan": scan_path.name,
            },
            prev_phash=snapshot.prev_phash,
            scrubber=self.scrubber,
        )
        write_screen_scan_dump(scan_path, record)
        logger.debug("Perception scan dump saved to %s", scan_path)

    def _page_stagnation_result(
        self,
        ctx: _RunContext,
        session_trail: Optional[SessionTrail],
        current_phash: Optional[str],
    ) -> Optional[DeviceRunResult]:
        if not ctx.loop_detector.terminated():
            return None

        logger.error("Page stagnation limit reached. Terminating.")
        return self._loop_detected_result(
            ctx,
            session_trail,
            current_phash,
            "Agent stuck: screen did not change for too many consecutive steps.",
        )

    def _plan_step(
        self,
        planner: DevicePlanner,
        goal: str | list[str],
        snapshot: _PerceptionSnapshot,
        ctx: _RunContext,
        session_trail: Optional[SessionTrail],
        loop_nudge: Optional[str],
    ) -> DeviceStepResponse:
        # Keyboard appearance commonly resizes the app and moves every field.
        # Always plan from the current scan; pre-keyboard coordinates are stale.
        gui_elements = marks.annotate_keyboard_keys(
            snapshot.gui_elements, ctx.keyboard_open, ctx.screen_height
        )
        self._number_targets(snapshot, ctx)
        return planner.plan_step(
            goal=goal,
            gui_elements=gui_elements,
            history=ctx.history.format_for_prompt(),
            trail=session_trail.format_for_prompt() if session_trail else None,
            drift_warning=ctx.drift_warning,
            loop_nudge=loop_nudge,
            current_image_bytes=snapshot.planner_img_bytes,
            marks=snapshot.marks,
            screen_height=ctx.screen_height or 0,
            visual_history=list(ctx.visual_history),
            keyboard_open=ctx.keyboard_open,
        )

    def _number_targets(self, snapshot: _PerceptionSnapshot, ctx: _RunContext) -> None:
        """Numbers the elements and redraws the screenshot to match.

        Done here rather than at perception time because the keyboard state is
        only known now, and the keyboard's own keys must not be numbered — they
        would outnumber the app's controls several times over.
        """
        if self.grounding != GROUNDING_MARKS:
            return

        targets, keys = marks.split_keyboard(
            snapshot.gui_elements, ctx.keyboard_open, ctx.screen_height
        )
        raw_img = Image.open(io.BytesIO(snapshot.raw_img_bytes))
        width, height = raw_img.size
        snapshot.marks = marks.build(targets, screen_area=width * height)

        marked = io.BytesIO()
        marks.render(
            raw_img, snapshot.marks, band=marks.keyboard_band(keys, width)
        ).save(marked, format="PNG")
        snapshot.planner_img_bytes = marked.getvalue()

    def _apply_plan_result(
        self,
        step_response: DeviceStepResponse,
        snapshot: _PerceptionSnapshot,
        session_trail: Optional[SessionTrail],
        ctx: _RunContext,
    ) -> None:
        action_text = getattr(step_response.action, "text", None)
        self.scrubber.learn(action_text)

        logger.info("Screen: %s", step_response.current_screen)
        logger.info("Reasoning: %s", self.scrubber.scrub(step_response.reasoning))
        if step_response.evaluation_previous_goal:
            logger.info(
                "Previous step eval: %s",
                self.scrubber.scrub(step_response.evaluation_previous_goal),
            )

        ctx.history.update_last_evaluation(step_response.evaluation_previous_goal)

        if session_trail and snapshot.current_phash:
            session_trail.record_screen(
                snapshot.current_phash,
                step_response.current_screen,
                snapshot.gui_elements,
            )

    def _check_step_hook(
        self,
        step_hook: StepHookFn,
        step_response: DeviceStepResponse,
        session_trail: Optional[SessionTrail],
        current_phash: Optional[str],
        ctx: _RunContext,
    ) -> Optional[DeviceRunResult]:
        if step_hook(step_response) is not False:
            return None

        self._flush_trail(session_trail, current_phash)
        return DeviceRunResult(
            status="hook_stop",
            steps_executed=ctx.steps_executed,
            failure_reason="Execution stopped by step hook (crash, infeasibility, or critical issue).",
            history=ctx.history,
        )

    def _check_done_action(
        self,
        step_response: DeviceStepResponse,
        session_trail: Optional[SessionTrail],
        current_phash: Optional[str],
        ctx: _RunContext,
    ) -> Optional[DeviceRunResult]:
        satisfied = self.stop_when_satisfied and getattr(
            step_response, "goal_satisfied", False
        )
        if not isinstance(step_response.action, DoneAction) and not satisfied:
            return None

        if isinstance(step_response.action, DoneAction):
            reason = step_response.action.reason
        else:
            reason = getattr(step_response, "goal_evidence", "") or "goal observed as complete"
        logger.info("Goal reached: %s", reason)
        self._flush_trail(session_trail, current_phash)
        return DeviceRunResult(
            status="goal_reached",
            steps_executed=ctx.steps_executed,
            failure_reason=None,
            history=ctx.history,
        )

    def _action_repetition_result(
        self,
        ctx: _RunContext,
        session_trail: Optional[SessionTrail],
        current_phash: Optional[str],
    ) -> Optional[DeviceRunResult]:
        if not ctx.loop_detector.terminated():
            return None

        logger.error("Action repetition limit reached. Terminating.")
        return self._loop_detected_result(
            ctx,
            session_trail,
            current_phash,
            "Agent stuck in an execution loop: repeated the same action too many times.",
        )

    def _save_action_screenshot(
        self,
        session_dir: Path,
        step: int,
        snapshot: _PerceptionSnapshot,
        action,
    ) -> None:
        click_path = session_dir / f"action_step{step + 1}_{snapshot.timestamp}.png"
        self.media_manager.save_redacted_action_annotation(
            snapshot.annotated_img, snapshot.gui_elements, action, click_path
        )

    def _execute_and_record(
        self,
        step: int,
        action,
        step_response: DeviceStepResponse,
        device_control,
        snapshot: _PerceptionSnapshot,
        session_trail: Optional[SessionTrail],
        ctx: _RunContext,
    ) -> bool:
        error: Optional[str] = None
        last_external_nav = ctx.last_external_nav
        try:
            ActionOrchestrator(device_control, self.scrubber).execute(action)
            last_external_nav = step_response.external_navigation
        except Exception as exc:
            error = str(exc)
            logger.error("Action execution failed: %s", error)

        post_phash = self._settle(device_control)
        action_result = self._build_action_result(
            action, snapshot.current_phash, post_phash, error
        )
        action_desc = self._describe_action(action)

        if self.replay_ledger is not None and error is None:
            self.replay_ledger.record(
                snapshot.current_phash,
                action,
                post_phash,
                action_desc,
                target_label(action, snapshot.gui_elements),
            )
        ctx.history.append(
            ActionEntry(
                step_num=step + 1,
                action_desc=action_desc,
                result=action_result,
            )
        )

        if session_trail:
            session_trail.record_action(action_desc)

        ctx.visual_history.append(snapshot.raw_img_bytes)
        if len(ctx.visual_history) > self.visual_history_limit:
            ctx.visual_history.pop(0)

        ctx.last_action_type = getattr(action, "type", None)
        return last_external_nav

    def _flush_trail(
        self,
        session_trail: Optional[SessionTrail],
        current_phash: Optional[str],
    ) -> None:
        if session_trail and current_phash:
            session_trail.flush_transition(current_phash)

    def _loop_detected_result(
        self,
        ctx: _RunContext,
        session_trail: Optional[SessionTrail],
        current_phash: Optional[str],
        failure_reason: str,
    ) -> DeviceRunResult:
        self._flush_trail(session_trail, current_phash)
        return DeviceRunResult(
            status="loop_detected",
            steps_executed=ctx.steps_executed,
            failure_reason=failure_reason,
            history=ctx.history,
        )

    def _step_limit_result(
        self,
        ctx: _RunContext,
        session_trail: Optional[SessionTrail],
    ) -> DeviceRunResult:
        limit = getattr(self, "_step_limit", self.max_steps)
        logger.warning("Reached max step limit (%d).", limit)
        self._flush_trail(session_trail, ctx.current_phash)
        return DeviceRunResult(
            status="step_limit",
            steps_executed=ctx.steps_executed,
            failure_reason=f"Reached maximum step limit ({limit}) before goal was completed.",
            history=ctx.history,
        )

    def run_standalone(self, goal: str, record_session: bool = True) -> DeviceRunResult:
        """
        Convenience method for running a simple goal with a self-managed device session.
        Used by main.py and other non-testing callers.
        """
        session = DeviceSession(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            record_session=record_session,
            session_dir=self.assets_dir,
            keep_device=self.keep_device,
            reuse_app=self.reuse_app,
        )
        try:
            device_control = session.start(is_headless=self.headless)
            perception = ScreenPerception(session)
            result = self.run(
                goal=goal,
                device_control=device_control,
                perception=perception,
                session_dir=session.screenshots_dir,
            )
        finally:
            session.close()
        return result

    def _check_context_drift(
        self, device_control, last_external_nav: bool
    ) -> Optional[str]:
        if not device_control.is_still_in_sut():
            if not last_external_nav:
                logger.warning("Unexpected context drift. Attempting recovery...")
                device_control.recover_to_sut()
                time.sleep(self.settle_time)
                if not device_control.is_still_in_sut():
                    logger.error("Recovery failed.")
                    return (
                        "Warning: The device has drifted outside the target app (SUT) and recovery failed. "
                        "Please take an action to navigate back to the app, or terminate the scenario."
                    )
                logger.info("Recovery successful.")
                return None
            return (
                "Info: The device is currently outside the target app (SUT) as expected. "
                "Make sure to return to the target app once the external tasks are completed."
            )
        return None

    def _restart_app(self, device_control) -> None:
        restart_app(device_control)

    def _settle(self, device_control) -> Optional[str]:
        """Waits for the screen to stop changing, up to settle_time.

        A fixed sleep pays the worst case on every action; most screens are
        static again well before it elapses. Returns the settled pHash so the
        caller does not need a further screenshot.
        """
        deadline = max(1, int(self.settle_time / SETTLE_POLL_INTERVAL))
        phash = self._quick_phash(device_control)
        stable = 1
        for _ in range(deadline):
            time.sleep(SETTLE_POLL_INTERVAL)
            current = self._quick_phash(device_control)
            if current is not None and current == phash:
                stable += 1
                if stable >= SETTLE_STABLE_READINGS:
                    return current
            else:
                stable = 1
            phash = current
        return phash

    def _quick_phash(self, device_control) -> Optional[str]:
        """Lightweight pHash capture after action — no YOLO/OCR, just a screenshot."""
        try:
            png_bytes = device_control.get_screenshot()
            img = Image.open(io.BytesIO(png_bytes))
            return str(imagehash.phash(img, hash_size=16))
        except Exception:
            return None

    def _build_action_result(
        self,
        action,
        pre_phash: Optional[str],
        post_phash: Optional[str],
        error: Optional[str],
    ) -> ActionResult:
        if error:
            return ActionResult(
                success=False,
                observation=f"Action raised an error: {error[:200]}",
                error=error,
            )

        screen_changed = (
            (pre_phash != post_phash) if (pre_phash and post_phash) else None
        )
        t = action.type.value if hasattr(action.type, "value") else str(action.type)

        if screen_changed is None:
            return ActionResult(
                success=True, observation=f"{t} executed (screen state unknown)"
            )

        if screen_changed:
            return ActionResult(
                success=True,
                observation=f"{t} executed — screen changed (navigation or state update occurred)",
            )

        if t == ActionType.TYPE.value:
            coords = action.coordinates
            obs = (
                f"type executed at ({coords.x}, {coords.y}) — screen unchanged "
                f"(text may have been entered; masked fields cannot be visually confirmed)"
            )
        elif t in (ActionType.TAP.value, ActionType.DOUBLE_TAP.value):
            coords = action.coordinates
            obs = (
                f"{t} at ({coords.x}, {coords.y}) — screen unchanged "
                f"(tap may have missed, button may be obscured, or no navigation was triggered)"
            )
        elif t == ActionType.KEY_EVENT.value:
            obs = f"key_event {action.key} sent — screen unchanged"
        elif t == ActionType.SWIPE.value:
            obs = "swipe executed — screen unchanged (content may have scrolled without triggering navigation)"
        elif t == ActionType.CLEAR.value:
            obs = f"clear executed at ({action.coordinates.x}, {action.coordinates.y}) — screen unchanged"
        else:
            obs = f"{t} executed — screen unchanged"

        return ActionResult(success=True, observation=obs)

    def _build_reaction(
        self,
        scan_info: Optional[ScanInfo],
        prev_element_count: Optional[int],
        keyboard_open: Optional[bool] = None,
    ) -> str:
        """Describe what the screen actually did after the previous action, using perception metadata."""
        if scan_info is None or scan_info.reason == "first_pass":
            return ""

        if scan_info.reason == "stable":
            return "screen unchanged — coordinates remain valid"

        parts = []

        if scan_info.reason == "navigation":
            dist = scan_info.phash_distance
            parts.append(f"navigation detected (pHash Δ{dist})" if dist else "navigation detected")
        elif scan_info.reason == "layout_shift":
            match_pct = int(scan_info.match_rate * 100) if scan_info.match_rate is not None else "?"
            parts.append(f"layout shifted ({match_pct}% elements matched)")
        elif scan_info.reason == "ocr_refresh":
            parts.append("screen text updated")

        if prev_element_count is not None:
            delta = scan_info.element_count - prev_element_count
            if keyboard_open is not None and abs(delta) > 30:
                if keyboard_open:
                    parts.append(f"keyboard opened (+{delta} elements, {prev_element_count}→{scan_info.element_count})")
                else:
                    parts.append(f"keyboard dismissed ({delta} elements, {prev_element_count}→{scan_info.element_count})")
            elif delta != 0:
                parts.append(f"element count {prev_element_count}→{scan_info.element_count}")
        else:
            parts.append(f"{scan_info.element_count} elements")

        if scan_info.reason in ("navigation", "layout_shift"):
            parts.append("coordinates invalidated")

        return ", ".join(parts)

    def _describe_action(self, action) -> str:
        """Produce a concise, redacted human-readable action description for history."""
        from ronin.utils.redactor import mask_typed_text, redact_text

        t = action.type.value if hasattr(action.type, "value") else str(action.type)

        coords = getattr(action, "coordinates", None)
        # An unresolved mark has no point to report; name the number instead.
        where = (
            f"({coords.x}, {coords.y})"
            if coords is not None
            else f"element {getattr(action, 'mark', '?')}"
        )

        if t == ActionType.TYPE.value:
            text_repr = mask_typed_text(getattr(action, "text", ""))
            desc = f"type {text_repr} at {where}"
        elif t in (
            ActionType.TAP.value,
            ActionType.DOUBLE_TAP.value,
            ActionType.LONG_PRESS.value,
            ActionType.CLEAR.value,
        ):
            desc = f"{t} at {where}"
        elif t == ActionType.KEY_EVENT.value:
            desc = f"key_event {action.key}"
        elif t == ActionType.SWIPE.value:
            desc = f"swipe ({action.start_x},{action.start_y})→({action.end_x},{action.end_y})"
        elif t == ActionType.DRAG_AND_DROP.value:
            desc = f"drag ({action.source_x},{action.source_y})→({action.destination_x},{action.destination_y})"
        elif t == ActionType.CAPABILITY.value:
            desc = f"capability:{action.capability}"
        else:
            desc = t

        reason = getattr(action, "reason", "")
        if reason:
            desc += f" (reason: {redact_text(reason)})"
        return desc
