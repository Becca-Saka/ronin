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
from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.device_session import DeviceSession
from ronin.agent.device.platform import Platform
from ronin.agent.device_planner import (
    DevicePlanner,
    DeviceStepResponse,
    DoneAction,
)
from ronin.agent.history import ActionEntry, ActionResult, AgentHistory
from ronin.agent.loop_detector import RollingLoopDetector
from ronin.agent.perception.media_manager import MediaManager
from ronin.agent.perception.perception import ScreenPerception
from ronin.agent.perception.scan_dump import (
    ScanInfo,
    build_screen_scan_record,
    write_screen_scan_dump,
)
from ronin.agent.trail import SessionTrail
from ronin.llm.llm_client import LLMClient
from ronin.utils.redactor import redact_elements, redact_text

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
    raw_img_bytes: bytes
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
    keyboard_off_elements: Optional[list] = None


class DeviceAgent:
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
            self._default_planner = DevicePlanner(self.llm, use_multimodal_vision)
            self._default_fast_planner = DevicePlanner(
                self.fast_llm, use_multimodal_vision
            )

        self.media_manager = MediaManager()

    def run(
        self,
        goal: str | list[str],
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
        step_hook: Optional[StepHookFn] = None,
        planner_override: Optional[DevicePlanner] = None,
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

        self._restart_app(device_control)

        try:
            for step in range(self.max_steps):
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
            logger.exception("Exception during device agent run: %s", exc)
            return DeviceRunResult(
                status="exception",
                steps_executed=ctx.steps_executed,
                failure_reason=redact_text(f"Execution exception: {str(exc)}"),
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

        if ctx.keyboard_open is False or (ctx.keyboard_open is None and snapshot.scan_info and snapshot.scan_info.reason == "first_pass"):
            ctx.keyboard_off_elements = snapshot.gui_elements

        reaction = self._build_reaction(snapshot.scan_info, ctx.prev_element_count, ctx.keyboard_open)
        ctx.history.update_last_reaction(reaction)
        ctx.prev_element_count = snapshot.scan_info.element_count if snapshot.scan_info else None

        page_nudge = ctx.loop_detector.check(None, snapshot.current_phash)
        stagnation_result = self._page_stagnation_result(
            ctx, session_trail, snapshot.current_phash
        )
        if stagnation_result is not None:
            return stagnation_result

        step_response = self._plan_step(
            planner=planner,
            goal=goal,
            snapshot=snapshot,
            ctx=ctx,
            session_trail=session_trail,
            loop_nudge=page_nudge,
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

        combined_nudge = (
            "\n".join(filter(None, [page_nudge, action_nudge])) or None
        )

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

        if combined_nudge:
            ctx.loop_detector.check(None, None)  # preserve detector state; nudge not yet wired to next plan_step

        ctx.prev_phash = snapshot.current_phash
        return None

    def _capture_perception(
        self,
        perception: ScreenPerception,
        session_trail: Optional[SessionTrail],
        prev_phash: Optional[str],
    ) -> _PerceptionSnapshot:
        gui_elements, raw_img, annotated_img = perception.perceive()
        gui_elements = redact_elements(gui_elements)
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

        return _PerceptionSnapshot(
            gui_elements=gui_elements,
            raw_img_bytes=buf.getvalue(),
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

        snapshot.annotated_img.save(annotated_path)
        raw_path.write_bytes(snapshot.raw_img_bytes)

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
        if ctx.keyboard_open and ctx.keyboard_off_elements is not None:
            gui_elements = ctx.keyboard_off_elements
        else:
            gui_elements = self._annotate_keyboard_keys(
                snapshot.gui_elements, ctx.keyboard_open, ctx.screen_height
            )
        return planner.plan_step(
            goal=goal,
            gui_elements=gui_elements,
            history=ctx.history.format_for_prompt(),
            trail=session_trail.format_for_prompt() if session_trail else None,
            drift_warning=ctx.drift_warning,
            loop_nudge=loop_nudge,
            current_image_bytes=snapshot.raw_img_bytes,
            visual_history=list(ctx.visual_history),
            keyboard_open=ctx.keyboard_open,
        )

    def _apply_plan_result(
        self,
        step_response: DeviceStepResponse,
        snapshot: _PerceptionSnapshot,
        session_trail: Optional[SessionTrail],
        ctx: _RunContext,
    ) -> None:
        logger.info("Screen: %s", step_response.current_screen)
        logger.info("Reasoning: %s", step_response.reasoning)
        if step_response.evaluation_previous_goal:
            logger.info(
                "Previous step eval: %s", step_response.evaluation_previous_goal
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
        if not isinstance(step_response.action, DoneAction):
            return None

        logger.info("Goal reached: %s", step_response.action.reason)
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
        self.media_manager.save_action_annotation(
            annotated_img=snapshot.annotated_img,
            action_plan=action,
            filepath=click_path,
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
            ActionOrchestrator(device_control).execute(action)
            last_external_nav = step_response.external_navigation
        except Exception as exc:
            error = str(exc)
            logger.error("Action execution failed: %s", error)

        time.sleep(self.settle_time)

        post_phash = self._quick_phash(device_control)
        action_result = self._build_action_result(
            action, snapshot.current_phash, post_phash, error
        )
        action_desc = self._describe_action(action)
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
        logger.warning("Reached max step limit (%d).", self.max_steps)
        self._flush_trail(session_trail, ctx.current_phash)
        return DeviceRunResult(
            status="step_limit",
            steps_executed=ctx.steps_executed,
            failure_reason=f"Reached maximum step limit ({self.max_steps}) before goal was completed.",
            history=ctx.history,
        )

    def run_standalone(self, goal: str) -> DeviceRunResult:
        """
        Convenience method for running a simple goal with a self-managed device session.
        Used by main.py and other non-testing callers.
        """
        session = DeviceSession(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            record_session=True,
            session_dir=self.assets_dir,
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
        try:
            if not hasattr(device_control, "package_id"):
                return
            package_id = device_control.package_id
            logger.info("Restarting application: %s", package_id)

            if device_control.platform == Platform.ANDROID:
                adb_runner = getattr(device_control, "adb_control", None) or AdbControl
                activity = None
                try:
                    _, activity = adb_runner.get_apk_info(device_control.app_path)
                except Exception as apk_err:
                    logger.warning("Failed to get APK activity: %s", apk_err)

                if activity:
                    serial = getattr(
                        device_control, "serial", None
                    ) or adb_runner.resolve_emulator_serial(device_control.device_name)
                    cmd = ["adb"]
                    if serial:
                        cmd.extend(["-s", serial])
                    cmd.extend(
                        [
                            "shell",
                            "am",
                            "start",
                            "-n",
                            f"{package_id}/{activity}",
                            "--activity-clear-task",
                        ]
                    )
                    try:
                        adb_runner.run_adb_command(cmd)
                        time.sleep(2.5)
                        return
                    except Exception as launch_err:
                        logger.warning(
                            "ADB clear-task failed: %s. Falling back.", launch_err
                        )

            device_control.driver.terminate_app(package_id)
            time.sleep(1.5)
            device_control.driver.activate_app(package_id)
            time.sleep(2.5)
        except Exception as exc:
            logger.warning("Failed to restart app: %s", exc)

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

    def _annotate_keyboard_keys(
        self,
        elements: list,
        keyboard_open: Optional[bool],
        screen_height: Optional[int],
    ) -> list:
        """When the keyboard is open, prefix elements that are likely keyboard keys with [key].

        Keyboard keys are identified as elements in the lower 40% of the screen whose
        text is 1-2 characters long (e.g. 'q', 'w', 'shift', '123'). App UI elements
        in that region (longer labels like 'Login', 'Sign Up') are left unchanged.
        """
        if not keyboard_open or not screen_height:
            return elements

        keyboard_top = int(screen_height * 0.60)
        annotated = []
        for el in elements:
            center_y = (el.bounding_box.y_min + el.bounding_box.y_max) // 2
            text = (el.text or "").strip()
            is_key = center_y >= keyboard_top and len(text) <= 2
            if is_key:
                from copy import copy
                el = copy(el)
                el.text = f"[key] {text}" if text else "[key]"
            annotated.append(el)
        return annotated

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
        from ronin.utils.redactor import redact_text

        t = action.type.value if hasattr(action.type, "value") else str(action.type)

        if t == ActionType.TYPE.value:
            coords = action.coordinates
            text_repr = redact_text(action.text) if getattr(action, "text", None) else ""
            desc = f"type '{text_repr}' at ({coords.x}, {coords.y})"
        elif t in (
            ActionType.TAP.value,
            ActionType.DOUBLE_TAP.value,
            ActionType.LONG_PRESS.value,
            ActionType.CLEAR.value,
        ):
            coords = action.coordinates
            desc = f"{t} at ({coords.x}, {coords.y})"
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
