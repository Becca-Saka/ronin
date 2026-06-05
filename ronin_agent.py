"""
RoninAgent class orchestrating session management and mobile E2E test execution.
Delegates planning logic to the RoninPlanner.
"""

import io
import logging
import time
from pathlib import Path

from actuation.orchestrator import ActionOrchestrator
from device.device_session import DeviceSession
from device.platform import Platform
from llm.llm_client import LLMClient
from perception.perception import ScreenPerception
from ronin_planner import RoninPlanner

logger = logging.getLogger(__name__)


class RoninAgent:
    """AI-driven mobile E2E testing agent.

    Coordinates device session, screen perception, planning (via RoninPlanner), and actuation.
    """

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
    ):
        """Initializes the RoninAgent with device and model configurations.

        Args:
            device_name: Name of the emulator/simulator to connect to.
            app_path: Path to the app binary under test.
            platform: Platform name ('android' or 'ios').
            model: LiteLLM model identifier to use for planning.
            assets_dir: Directory to store annotations and recordings.
        """
        self.device_name = device_name
        self.app_path = app_path
        self.platform = platform
        self.assets_dir = assets_dir
        self.llm = LLMClient(model=model)
        self.planner = RoninPlanner(self.llm)

    def execute(self, goal: str):
        """Runs the E2E testing agent loop to accomplish the given goal.

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
            device_control = session.start(is_headless=True)
            perception = ScreenPerception(session)
            gui_elements, annotated_img = perception.perceive()

            self.assets_dir.mkdir(parents=True, exist_ok=True)
            clean_timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = self.assets_dir / f"annotated_img_{clean_timestamp}.png"
            annotated_img.save(filepath)
            logger.info("Saved annotated image to %s", filepath)

            elements_summary = "\n".join(
                f"{el.element:20} {el.text:20} {el.confidence:20} {str(el.bounding_box):20} {el.type:20}"
                for el in gui_elements
            )
            logger.info("Detected GUI Elements:\n%s", elements_summary)

            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            annotated_img_bytes = buf.getvalue()

            task_plan = self.planner.plan_task(goal, annotated_img_bytes)
            logger.info("Got task plan: %s", task_plan)

            actions = self.planner.plan_actions(
                task_plan, gui_elements, annotated_img_bytes
            )
            logger.info("Got actions: %s", actions)

            logger.info("Executing Actions...")
            ActionOrchestrator(device_control).orchestrate(actions)
            time.sleep(3)
            logger.info("Task completed successfully!")

        finally:
            session.close()
