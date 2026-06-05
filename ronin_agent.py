"""
RoninAgent class orchestrating session management and mobile E2E test execution.
Implements a closed-loop ReAct execution loop using RoninPlanner.
"""

import io
import logging
import time
from pathlib import Path

from actuation.orchestrator import ActionOrchestrator
from device.device_session import DeviceSession
from device.platform import Platform
from llm.llm_client import LLMClient
from perception.media_manager import MediaManager
from perception.perception import ScreenPerception
from ronin_planner import RoninPlanner

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
        self.planner = RoninPlanner(
            self.llm, use_multimodal_vision=use_multimodal_vision
        )
        self.media_manager = MediaManager()

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

            for step in range(self.max_steps):
                logger.info("--- ReAct Loop Step %s / %s ---", step + 1, self.max_steps)

                # 1. Screen Perception
                gui_elements, annotated_img = perception.perceive()

                self.assets_dir.mkdir(parents=True, exist_ok=True)
                clean_timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = self.assets_dir / f"annotated_img_{clean_timestamp}_step{step+1}.png"
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

                # 2. Plan the single next action
                step_response = self.planner.plan_next_step(
                    goal=goal,
                    gui_elements=gui_elements,
                    current_image_bytes=annotated_img_bytes,
                    visual_history=list(visual_history),
                    text_history=list(text_history),
                )
                logger.info("Reasoning: %s", step_response.reasoning)

                # 3. Termination Check
                if step_response.is_done:
                    logger.info("Goal reached! Ending E2E execution.")
                    break

                if not step_response.action or not step_response.action.actions:
                    logger.warning("Planner returned not done, but did not specify any action. Terminating.")
                    break

                # 4. Actuation
                action_to_run = step_response.action
                logger.info("Executing action plan: %s", action_to_run.instruction)

                # Saves a copy of the screenshot with the action highlighted
                click_filepath = self.assets_dir / f"click_step{step+1}_{clean_timestamp}.png"
                self.media_manager.save_action_annotation(
                    annotated_img=annotated_img,
                    action_plan=action_to_run,
                    filepath=click_filepath,
                )

                ActionOrchestrator(device_control).orchestrate(action_to_run)

                # 5. History Update
                first_action = action_to_run.actions[0]
                action_desc = f"{first_action.type.value} (Reason: {first_action.reason})"
                text_history.append(action_desc)

                visual_history.append(annotated_img_bytes)
                if len(visual_history) > self.visual_history_limit:
                    visual_history.pop(0)

                # 6. Settle Delay
                time.sleep(self.settle_time)
            else:
                logger.warning("Reached maximum step limit (%s). Terminating E2E run.", self.max_steps)

        finally:
            session.close()
