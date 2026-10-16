"""
DeviceAgent — core ReAct loop for LLM-driven mobile device control.
No QA concerns. Accepts an optional planner override so the testing layer
can inject QAPlanner without re-implementing the loop.
"""

import logging
from pathlib import Path
from typing import Optional


from ronin.agent.actuation.actions import ActionType
from ronin.agent.actuation.orchestrator import ActionOrchestrator
from ronin.agent.device.device_session import DeviceSession
from ronin.agent.device.platform import Platform
from ronin.agent.device_planner import (
    GROUNDING_COORDINATES,
    DevicePlanner,
    DeviceStepResponse,
)
from ronin.agent.history import ActionEntry, AgentHistory
from ronin.agent.device.launch import SetupMode, clear_app_data, restart_app
from ronin.agent.step import outcomes
from ronin.agent.step.outcomes import DeviceRunResult, StepHookFn
from ronin.agent.step import observation
from ronin.agent.step.observation import PerceptionSnapshot
from ronin.agent.step.artifacts import StepArtifacts
from ronin.agent.step import narration
from ronin.agent.step.loop_detector import RollingLoopDetector
from ronin.agent.step import recovery
from ronin.agent.step.run_context import RunContext
from ronin.agent.step.stall import SPENT
from ronin.agent.replay import ReplayLedger, replay_known_path, target_label
from ronin.agent.step import targeting
from ronin.utils.secrets import SecretScrubber
from ronin.agent.perception import marks
from ronin.agent.perception.media_manager import MediaManager
from ronin.agent.perception.perception import ScreenPerception
from ronin.agent.trail import SessionTrail
from ronin.errors import TargetNotInteractive
from ronin.llm.llm_client import LLMClient
from ronin.utils.redactor import redact_text

logger = logging.getLogger(__name__)

"""
Called after each planning step, before action execution.
Return False to request early loop termination (e.g. crash detected, infeasible scenario).
Return None or True to continue normally.
"""


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
        headless: bool = True,
        fast_model: str = None,
        planner: Optional[DevicePlanner] = None,
        stop_when_satisfied: bool = False,
        keep_device: bool = False,
        reuse_app: bool = False,
        replay_ledger: Optional[ReplayLedger] = None,
        grounding: str = GROUNDING_COORDINATES,
    ):
        self.device_name = device_name
        self.app_path = app_path
        self.platform = platform
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.settle_time = settle_time
        self.headless = headless
        self.use_multimodal_vision = use_multimodal_vision
        self.grounding = grounding
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
                self.llm, use_multimodal_vision, grounding
            )
            self._default_fast_planner = DevicePlanner(
                self.fast_llm, use_multimodal_vision, grounding
            )

        self.media_manager = MediaManager(self.scrubber)
        self.artifacts = StepArtifacts(self.media_manager, self.scrubber)

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
        setup: SetupMode = SetupMode.RELAUNCH,
    ) -> DeviceRunResult:
        """Runs the ReAct loop until something stops it.

        It stops on the goal, the step limit, a detected loop, a stop from
        step_hook, or an exception.

        planner_override replaces the default planner for this run only. QAAgent
        uses it to inject a QAPlanner without building the agent again.
        """
        planner = planner_override or self._default_fast_planner
        session_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            screen_height=getattr(device_control, "screen_height", None),
        )

        try:
            # Establish the state the caller asked for, rather than describing
            # a state to the planner and hoping the device agrees.
            if setup is SetupMode.CLEAR_DATA:
                clear_app_data(device_control)
            if setup is not SetupMode.PRESERVE:
                restart_app(device_control)
            replay_known_path(
                ledger=self.replay_ledger,
                device_control=device_control,
                goal=goal,
                history=ctx.history,
                quick_phash=observation.quick_phash,
                settle=lambda control: observation.settle(control, self.settle_time),
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
                    self._log_recovery_cost(ctx)
                    return result

            self._log_recovery_cost(ctx)
            return outcomes.step_limit(
                getattr(self, "_step_limit", self.max_steps), ctx, session_trail
            )

        except Exception as exc:
            safe_error = self.scrubber.scrub(redact_text(str(exc)))
            logger.error(
                "Device agent run failed (%s): %s", type(exc).__name__, safe_error
            )
            self._log_recovery_cost(ctx)
            return DeviceRunResult(
                status="exception",
                steps_executed=ctx.steps_executed,
                failure_reason=f"Execution exception: {safe_error}",
                history=ctx.history,
            )

    @staticmethod
    def _log_recovery_cost(ctx: RunContext) -> None:
        """Says what rescuing the run cost, which the step count does not show."""
        if ctx.recovery_looks:
            logger.info(
                "Recovery was used on %d screen(s), costing %d extra model call(s).",
                len(ctx.recovered),
                ctx.recovery_looks,
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
        ctx: RunContext,
    ) -> Optional[DeviceRunResult]:
        ctx.steps_executed += 1
        logger.info(
            "--- [DeviceAgent] Step %d / %d ---", step + 1, self.max_steps
        )

        ctx.drift_warning = observation.context_drift(
            device_control, ctx.last_external_nav, self.settle_time
        )

        snapshot = observation.capture(
            perception, session_trail, ctx.prev_phash
        )
        ctx.current_phash = snapshot.current_phash
        self.artifacts.save_step(session_dir, step, snapshot)

        needs_keyboard_check = (
            snapshot.scan_info and snapshot.scan_info.reason in ("navigation", "layout_shift")
        ) or ctx.last_action_type == ActionType.TYPE.value
        if needs_keyboard_check:
            detected = device_control.is_keyboard_open()
            if detected is not None:
                ctx.keyboard_open = detected

        reaction = narration.reaction(snapshot.scan_info, ctx.prev_element_count, ctx.keyboard_open)
        ctx.history.update_last_reaction(reaction)
        ctx.prev_element_count = snapshot.scan_info.element_count if snapshot.scan_info else None

        page_nudge = ctx.loop_detector.check(None, snapshot.current_phash)
        stagnation_result = outcomes.page_stagnation(
            ctx, session_trail, snapshot.current_phash
        )
        if stagnation_result is not None:
            return stagnation_result

        loop_nudge = "\n".join(
            filter(None, [ctx.pending_loop_nudge, page_nudge])
        ) or None
        ctx.pending_loop_nudge = None

        # Saved before the model is called, so a planning failure still leaves
        # the image the planner was given.
        observation.number_targets(snapshot, ctx, self.grounding)
        self.artifacts.save_planner_view(session_dir, step, snapshot)

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
            hook_result = outcomes.hook_stop(
                step_hook, step_response, snapshot, session_trail, snapshot.current_phash, ctx
            )
            if hook_result is not None:
                return hook_result

        done_result = outcomes.goal_reached(
            self.stop_when_satisfied,
            step_response,
            session_trail,
            snapshot.current_phash,
            ctx,
        )
        if done_result is not None:
            return done_result

        action = step_response.action
        action_nudge = ctx.loop_detector.check(
            RollingLoopDetector.hash_action(action), None
        )
        loop_result = outcomes.action_repetition(
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
                ctx.history, step, narration.describe(action), decision.observation
            )
            if decision.unmatched:
                targeting.save_unmatched_screen(
                    self.media_manager,
                    session_dir / f"not_listed_step{step + 1}_{snapshot.timestamp}.png",
                    snapshot,
                )
            if recovery.reveal_behind_keyboard(device_control, snapshot, ctx):
                return None
            ctx.stall.record_blocked(snapshot.current_phash)
            if ctx.stall.spent and not self._unblock(
                decision.target or narration.describe(action),
                device_control,
                perception,
                ctx,
            ):
                return outcomes.loop_detected(
                    ctx,
                    session_trail,
                    snapshot.current_phash,
                    targeting.stalled_reason(decision),
                )
            return None

        action = decision.action or action
        self.artifacts.save_action(session_dir, step, snapshot, action)
        ctx.last_external_nav = self._execute_and_record(
            step=step,
            action=action,
            step_response=step_response,
            device_control=device_control,
            snapshot=snapshot,
            session_trail=session_trail,
            ctx=ctx,
        )
        if ctx.stall.stuck and recovery.reveal_behind_keyboard(
            device_control, snapshot, ctx
        ):
            return None
        # An action that runs and fails stalls a screen as surely as one refused
        # before dispatch. Burning a mark pushes the planner to try a different
        # one, so a fresh target each time is still a screen going nowhere.
        if ctx.stall.spent and not self._unblock(
            narration.describe(action), device_control, perception, ctx
        ):
            return outcomes.loop_detected(
                ctx, session_trail, snapshot.current_phash, SPENT
            )

        ctx.prev_phash = snapshot.current_phash
        return None




    def _unblock(self, target: str, device_control, perception, ctx: RunContext) -> bool:
        """Supplies recovery with the parts of this agent it needs."""
        return recovery.hand_over(
            target=target,
            llm=self.fast_llm,
            use_multimodal_vision=self.use_multimodal_vision,
            grounding=self.grounding,
            scrubber=self.scrubber,
            device_control=device_control,
            perception=perception,
            ctx=ctx,
        )

    def _plan_step(
        self,
        planner: DevicePlanner,
        goal: str | list[str],
        snapshot: PerceptionSnapshot,
        ctx: RunContext,
        session_trail: Optional[SessionTrail],
        loop_nudge: Optional[str],
    ) -> DeviceStepResponse:
        # A keyboard commonly resizes the app and moves every field, so plan
        # from the current scan. Pre-keyboard coordinates are stale.
        gui_elements = marks.annotate_keyboard_keys(
            snapshot.gui_elements, ctx.keyboard_open, ctx.screen_height
        )
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
            keyboard_open=ctx.keyboard_open,
        )


    def _apply_plan_result(
        self,
        step_response: DeviceStepResponse,
        snapshot: PerceptionSnapshot,
        session_trail: Optional[SessionTrail],
        ctx: RunContext,
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





    def _execute_and_record(
        self,
        step: int,
        action,
        step_response: DeviceStepResponse,
        device_control,
        snapshot: PerceptionSnapshot,
        session_trail: Optional[SessionTrail],
        ctx: RunContext,
    ) -> bool:
        error: Optional[str] = None
        last_external_nav = ctx.last_external_nav
        try:
            ActionOrchestrator(device_control, self.scrubber).execute(action)
            last_external_nav = step_response.external_navigation
            ctx.stall.action_dispatched()
        except TargetNotInteractive as exc:
            error = str(exc)
            targeting.refuse_mark(
                ctx.text_strikes,
                ctx.refused_marks,
                snapshot.current_phash,
                step_response.action.model_dump().get("mark"),
                snapshot.marks,
            )
            logger.error("Action execution failed: %s", error)
        except Exception as exc:
            error = str(exc)
            logger.error("Action execution failed: %s", error)

        if error is not None:
            ctx.stall.record_blocked(snapshot.current_phash)

        post_phash = observation.settle(device_control, self.settle_time)
        action_result = narration.action_result(
            action, snapshot.current_phash, post_phash, error
        )
        action_desc = narration.describe(action)

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


        ctx.last_action_type = getattr(action, "type", None)
        return last_external_nav




    def run_standalone(self, goal: str, record_session: bool = True) -> DeviceRunResult:
        """Runs one goal against a device session this method owns."""
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
