"""
RoninAgent class orchestrating session management and mobile E2E test execution.
Implements a closed-loop ReAct execution loop using RoninPlanner.
"""

import io
import logging
import time
from pathlib import Path

from actuation.orchestrator import ActionOrchestrator
from device.adb_control import AdbControl
from device.device_session import DeviceSession
from device.platform import Platform
from llm.llm_client import LLMClient
from perception.media_manager import MediaManager
from perception.perception import ScreenPerception
from ronin_planner import RoninPlanner
from ronin_testing.models import (
    GeneratedTestCase,
    ScenarioReport,
    QAIssue,
    RuleVerification,
)
from utils.redactor import redact_elements, redact_text

logger = logging.getLogger(__name__)


class RoninAgent:
    """AI-driven mobile E2E testing agent.

    Coordinates device session, screen perception, and closed-loop ReAct planning.
    """

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
        max_steps: int = 15,
        settle_time: float = 2.0,
        use_multimodal_vision: bool = False,
        visual_history_limit: int = 3,
        headless: bool = True,
        fast_model: str = None,
    ):
        """Initializes the RoninAgent.

        Args:
            device_name: Name of the emulator/simulator to connect to.
            app_path: Path to the app binary under test.
            platform: Platform name (Platform.ANDROID or Platform.IOS).
            model: LiteLLM model identifier to use for planning.
            assets_dir: Directory to store annotations and recordings.
            max_steps: Maximum step iterations allowed in the ReAct loop.
            settle_time: Time to wait (in seconds) after actions to let the screen settle.
            use_multimodal_vision: Toggle to allow LLM coordinate override.
            visual_history_limit: Maximum number of previous screenshots to keep in history.
            fast_model: LiteLLM model identifier for simpler planning steps.
        """
        self.device_name = device_name
        self.headless = headless
        self.app_path = app_path
        self.platform = platform
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.settle_time = settle_time
        self.visual_history_limit = visual_history_limit

        self.llm = LLMClient(model=model)
        if fast_model and fast_model != model:
            self.fast_llm = LLMClient(model=fast_model)
        else:
            self.fast_llm = self.llm

        self.planner = RoninPlanner(
            self.llm, use_multimodal_vision=use_multimodal_vision
        )
        self.fast_planner = RoninPlanner(
            self.fast_llm, use_multimodal_vision=use_multimodal_vision
        )
        self.media_manager = MediaManager()

    def _check_context_drift(
        self, device_control, last_external_nav: bool
    ) -> str | None:
        """Checks if the device has drifted outside the target app and attempts recovery if unexpected."""
        if not device_control.is_still_in_sut():
            if not last_external_nav:
                logger.warning(
                    "Unexpected context drift detected. Attempting recovery..."
                )
                device_control.recover_to_sut()
                time.sleep(self.settle_time)

                # Re-verify after recovery attempt
                if not device_control.is_still_in_sut():
                    logger.error("Recovery failed. Drift warning active.")
                    return (
                        "Warning: The device has drifted outside the target app (SUT) and recovery failed. "
                        "Please take an action to navigate back to the app, or terminate the scenario."
                    )
                else:
                    logger.info("Recovery successful, returned to SUT.")
                    return None
            else:
                logger.info(
                    "Device is outside SUT, but this was expected (external_navigation=True)."
                )
                return (
                    "Info: The device is currently outside the target app (SUT) as expected. "
                    "Make sure to return to the target app once the external tasks are completed."
                )
        return None

    def _restart_app(self, device_control):
        """Resets the application to a clean start state using a process-safe relaunch on Android or standard restart on iOS."""
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
                    logger.warning("Failed to get APK activity info: %s", apk_err)

                if activity:
                    serial = getattr(
                        device_control, "serial", None
                    ) or adb_runner.resolve_emulator_serial(
                        device_control.device_name
                    )
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
                    logger.info("Running clear-task relaunch: %s", " ".join(cmd))
                    try:
                        adb_runner.run_adb_command(cmd)
                        time.sleep(2.5)
                        return
                    except Exception as launch_err:
                        logger.warning(
                            "Failed clear-task relaunch via ADB: %s. Falling back to terminate/activate.",
                            launch_err,
                        )

            # Fallback for Android (if ADB launch failed/not possible) or standard for iOS
            device_control.driver.terminate_app(package_id)
            time.sleep(1.5)
            device_control.driver.activate_app(package_id)
            time.sleep(2.5)
        except Exception as e:
            logger.warning("Failed to reset application state: %s", e)

    def execute(self, goal: str):
        """Runs the closed-loop ReAct execution loop to accomplish the target goal.

        Args:
            goal: The target task description/goal to run on the app.
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

            visual_history: list[bytes] = []
            text_history: list[str] = []
            last_external_nav = False
            drift_warning = None

            for step in range(self.max_steps):
                logger.info("--- ReAct Loop Step %s / %s ---", step + 1, self.max_steps)

                # 0. Context Drift Check
                drift_warning = self._check_context_drift(
                    device_control, last_external_nav
                )

                # 1. Screen Perception
                gui_elements, annotated_img = perception.perceive()
                gui_elements = redact_elements(gui_elements)

                clean_timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = (
                    session.screenshots_dir
                    / f"annotated_img_{clean_timestamp}_step{step + 1}.png"
                )
                annotated_img.save(filepath)
                logger.info("Saved step screenshot to %s", filepath)

                buf = io.BytesIO()
                annotated_img.save(buf, format="PNG")
                annotated_img_bytes = buf.getvalue()

                elements_summary = "\n".join(
                    f"{el.element:20} {el.text:20} {el.confidence:20} {str(el.bounding_box):20} {el.type:20}"
                    for el in gui_elements
                )
                logger.debug("Detected GUI Elements:\n%s", elements_summary)

                # 2. Plan the single next action (using fast planner)
                step_response = self.fast_planner.plan_next_step(
                    goal=goal,
                    gui_elements=gui_elements,
                    current_image_bytes=annotated_img_bytes,
                    visual_history=list(visual_history),
                    text_history=list(text_history),
                    drift_warning=drift_warning,
                )
                logger.info("Reasoning: %s", step_response.reasoning)

                # 3. Termination Check
                if step_response.is_done:
                    logger.info("Goal reached! Ending E2E execution.")
                    break

                if not step_response.action or not step_response.action.actions:
                    logger.warning(
                        "Planner returned not done, but did not specify any action. Terminating."
                    )
                    break

                # 4. Actuation
                action_to_run = step_response.action
                last_external_nav = getattr(action_to_run, "external_navigation", False)
                logger.info(
                    "Executing action plan: %s", redact_text(action_to_run.instruction)
                )

                # Saves a copy of the screenshot with the action highlighted
                click_filepath = (
                    session.screenshots_dir
                    / f"click_step{step + 1}_{clean_timestamp}.png"
                )
                self.media_manager.save_action_annotation(
                    annotated_img=annotated_img,
                    action_plan=action_to_run,
                    filepath=click_filepath,
                )

                ActionOrchestrator(device_control).orchestrate(action_to_run)

                # 5. History Update
                first_action = action_to_run.actions[0]
                action_desc = redact_text(
                    f"{first_action.type.value} (Reason: {first_action.reason})"
                )
                text_history.append(action_desc)

                visual_history.append(annotated_img_bytes)

                if len(visual_history) > self.visual_history_limit:
                    visual_history.pop(0)

                # 6. Settle Delay
                time.sleep(self.settle_time)
            else:
                logger.warning(
                    "Reached maximum step limit (%s). Terminating E2E run.",
                    self.max_steps,
                )

        finally:
            session.close()

    def execute_test_scenario(
        self,
        test_case: GeneratedTestCase,
        security_rules: list[str],
        device_control,
        perception,
        session_dir: Path,
    ) -> ScenarioReport:
        """Executes a single E2E test scenario using the ReAct loop and audits for layout/security issues."""
        start_time = time.time()
        visual_history: list[bytes] = []
        text_history: list[str] = []
        rule_verifications: list[RuleVerification] = []
        detected_issues: list[QAIssue] = []
        screenshots: list[str] = []
        status = "PASSED"
        failure_reason = None
        steps_executed = 0

        # Loop detection states
        previous_screen_sig = None
        previous_action_sig = None
        previous_action_desc = None
        consecutive_repeats = 0
        consecutive_waits = 0

        # Terminate and activate app to start from clean slate
        self._restart_app(device_control)

        try:
            last_external_nav = False
            drift_warning = None
            for step in range(self.max_steps):
                steps_executed += 1
                logger.info(
                    "--- [Scenario %s] Step %d / %d ---",
                    test_case.test_id,
                    step + 1,
                    self.max_steps,
                )

                # 0. Context Drift Check
                drift_warning = self._check_context_drift(
                    device_control, last_external_nav
                )

                # 1. Screen Perception
                gui_elements, annotated_img = perception.perceive()
                gui_elements = redact_elements(gui_elements)

                # Construct screen signature to detect repetition loops
                current_screen_sig = (
                    tuple(
                        sorted(
                            (
                                el.element,
                                el.text or "",
                                (el.bounding_box.x_min + el.bounding_box.x_max) // 2,
                                (el.bounding_box.y_min + el.bounding_box.y_max) // 2,
                            )
                            for el in gui_elements
                        )
                    )
                    if gui_elements
                    else ()
                )



                clean_timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = (
                    f"scen_{test_case.test_id}_step{step + 1}_{clean_timestamp}.png"
                )
                filepath = session_dir / filename
                annotated_img.save(filepath)
                screenshots.append(str(filepath))

                buf = io.BytesIO()
                annotated_img.save(buf, format="PNG")
                annotated_img_bytes = buf.getvalue()

                # 2. Plan step and audit via LLM Planner (using fast model)
                step_response = self.fast_planner.plan_next_step(
                    goal=test_case.steps_instructions,
                    expected_outcomes=test_case.expected_outcomes,
                    security_rules=security_rules,
                    gui_elements=gui_elements,
                    current_image_bytes=annotated_img_bytes,
                    visual_history=list(visual_history),
                    text_history=list(text_history),
                    drift_warning=drift_warning,
                )

                logger.info("Screen Name: %s", step_response.current_screen)
                logger.info("Reasoning: %s", step_response.reasoning)

                # Check relevance/feasibility of the test case
                if not getattr(step_response, "is_scenario_feasible", True):
                    logger.warning(
                        "Test scenario flagged as infeasible for current screen: %s",
                        getattr(
                            step_response, "infeasibility_reason", "No reason provided"
                        ),
                    )
                    status = "SKIPPED"
                    failure_reason = f"Test scenario is infeasible: {getattr(step_response, 'infeasibility_reason', 'No reason provided')}"
                    break

                # Update dynamic rule verifications
                for verify in step_response.rule_verifications:
                    if not verify.screen_name:
                        verify.screen_name = step_response.current_screen
                    rule_verifications.append(verify)

                # Update detected issues (assign screen name context)
                for issue in step_response.detected_issues:
                    issue.current_screen = step_response.current_screen
                    detected_issues.append(issue)

                # Check if there is a crash screen (hard failure that aborts execution)
                has_crash = any(
                    iss.issue_type == "crash_screen"
                    for iss in step_response.detected_issues
                )
                if has_crash:
                    logger.error("App crash screen detected! Terminating scenario.")
                    status = "FAILED"
                    failure_reason = "App crashed: crash screen detected."
                    break

                # Mark scenario failed on any rule failures or critical issues, but do not exit early
                has_critical_issue = any(
                    iss.severity == "critical" for iss in step_response.detected_issues
                )
                has_failed_rule = any(
                    v.status == "failed" for v in step_response.rule_verifications
                )
                if has_critical_issue or has_failed_rule:
                    status = "FAILED"
                    if not failure_reason:
                        if has_failed_rule:
                            failure_reason = f"Rule validation failed: {next(v.rule for v in step_response.rule_verifications if v.status == 'failed')}"
                        else:
                            failure_reason = f"Critical QA Issue: {next(iss.description for iss in step_response.detected_issues if iss.severity == 'critical')}"

                if step_response.is_done:
                    logger.info("Scenario objectives successfully finalized.")
                    break

                if not step_response.action or not step_response.action.actions:
                    if not step_response.is_done:
                        is_loading = (
                            "splash" in step_response.current_screen.lower()
                            or "loading" in step_response.current_screen.lower()
                            or not gui_elements
                        )
                        if is_loading and consecutive_waits < 3:
                            consecutive_waits += 1
                            logger.info(
                                "On splash/loading screen with no actions. Sleeping %s seconds and retrying (wait %d/3)...",
                                self.settle_time,
                                consecutive_waits,
                            )
                            time.sleep(self.settle_time)
                            steps_executed -= 1
                            continue
                        else:
                            logger.warning(
                                "Planner did not supply next action. Terminating scenario as FAILED."
                            )
                            status = "FAILED"
                            failure_reason = "Planner terminated early without supplying a next action before scenario objectives were met."
                            break
                    else:
                        logger.info("Scenario objectives successfully finalized.")
                        break

                # Loop detection: Compare proposed action signature with previous action signature
                first_action = step_response.action.actions[0]
                action_x = getattr(
                    getattr(first_action, "coordinates", None), "x", None
                )
                action_y = getattr(
                    getattr(first_action, "coordinates", None), "y", None
                )
                action_text = getattr(first_action, "text", None)
                action_key = getattr(first_action, "key", None)
                action_sig = (
                    first_action.type.value,
                    action_x,
                    action_y,
                    action_text,
                    action_key,
                    getattr(first_action, "source_x", None),
                    getattr(first_action, "source_y", None),
                    getattr(first_action, "destination_x", None),
                    getattr(first_action, "destination_y", None),
                    getattr(first_action, "start_x", None),
                    getattr(first_action, "start_y", None),
                    getattr(first_action, "end_x", None),
                    getattr(first_action, "end_y", None),
                    getattr(first_action, "duration", None),
                )

                if (
                    current_screen_sig == previous_screen_sig
                    and action_sig == previous_action_sig
                ):
                    consecutive_repeats += 1
                    logger.warning(
                        "Agent repeated same action on same screen state. Consecutive repeats: %d",
                        consecutive_repeats,
                    )
                else:
                    # Reset if action or screen changed
                    if current_screen_sig != previous_screen_sig:
                        consecutive_repeats = 0

                if consecutive_repeats >= 2:
                    logger.error(
                        "Agent stuck in an execution loop (repeated same action 3 times on same state). Terminating scenario."
                    )
                    status = "FAILED"
                    failure_reason = "Agent stuck in an execution loop: repeated action on screen state."
                    break

                # 3. Actuation
                action_to_run = step_response.action
                last_external_nav = getattr(action_to_run, "external_navigation", False)
                logger.info(
                    "Executing action: %s", redact_text(action_to_run.instruction)
                )

                click_filepath = (
                    session_dir
                    / f"click_{test_case.test_id}_step{step + 1}_{clean_timestamp}.png"
                )
                self.media_manager.save_action_annotation(
                    annotated_img=annotated_img,
                    action_plan=action_to_run,
                    filepath=click_filepath,
                )

                ActionOrchestrator(device_control).orchestrate(action_to_run)

                # 4. History Update
                action_desc = redact_text(
                    f"{first_action.type.value} (Reason: {first_action.reason})"
                )
                text_history.append(action_desc)

                visual_history.append(annotated_img_bytes)

                if len(visual_history) > self.visual_history_limit:
                    visual_history.pop(0)

                # Keep track of signatures for loop verification
                previous_screen_sig = current_screen_sig
                previous_action_sig = action_sig
                previous_action_desc = action_to_run.instruction
                consecutive_waits = 0

                # 5. Settle Delay
                time.sleep(self.settle_time)
            else:
                logger.warning(
                    "Reached maximum step limit (%d) for scenario.", self.max_steps
                )
                if status == "PASSED":
                    status = "FAILED"
                    failure_reason = (
                        "Reached maximum step limit before scenario completed."
                    )

        except Exception as e:
            logger.exception("Exception during test scenario execution: %s", e)
            status = "FAILED"
            failure_reason = redact_text(f"Execution crash: {str(e)}")

        duration = time.time() - start_time
        return ScenarioReport(
            test_id=test_case.test_id,
            scenario_name=test_case.scenario_name,
            classification=test_case.classification,
            status=status,
            duration=round(duration, 2),
            steps_executed=steps_executed,
            rule_verifications=rule_verifications,
            detected_issues=detected_issues,
            screenshots=screenshots,
            failure_reason=redact_text(failure_reason) if failure_reason else None,
        )
