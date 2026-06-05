import io
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from actuation.actions import ActionPlan
from actuation.orchestrator import ActionOrchestrator
from device.device_session import DeviceSession
from llm.llm_client import LLMClient
from llm.messages import ImageMessage, Message
from perception.perception import ScreenPerception

assets_dir = Path("assets")
goal = "I want to sign in to the app with email 'test@email.com'"


def main():
    device_name = "Pixel_8a"
    app_path = "example/testing_app/build/app/outputs/flutter-apk/app-debug.apk"
    session = DeviceSession(device_name, app_path)
    device_control = session.start()
    perception = ScreenPerception(session)
    gui_elements, annotated_img = perception.perceive()

    assets_dir.mkdir(parents=True, exist_ok=True)
    clean_timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = assets_dir / f"annotated_img_{clean_timestamp}.png"
    annotated_img.save(filepath)
    print(f"  ↳ saved → {filepath}")
    print(
        "{:20} {:20} {:20} {:20} {:20}".format(
            "Element",
            "Text",
            "Confidence",
            "Bounding box",
            "Type",
        )
    )
    print("-" * 80)
    for el in gui_elements:
        print(
            "{:20} {:20} {:20} {:20} {:20}".format(
                el.element, el.text, el.confidence, str(el.bounding_box), el.type
            )
        )

    def format_elements_for_prompt(elements) -> str:
        lines = []
        for el in elements:
            cx = (el.bounding_box.x_min + el.bounding_box.x_max) // 2
            cy = (el.bounding_box.y_min + el.bounding_box.y_max) // 2
            if el.text:
                lines.append(
                    f"- {el.element} | text: '{el.text}' | center: ({cx}, {cy})"
                )
            else:
                lines.append(f"- {el.element} | center: ({cx}, {cy})")
        return "\n".join(lines)

    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    annotated_img_bytes = buf.getvalue()

    class TaskAction(BaseModel):
        action: str = Field(..., description="The action to perform")
        step: int = Field(..., description="The step number")
        description: str = Field(..., description="The description of the action")
        reason: str = Field(..., description="The reason for the action")

    class TaskPlan(BaseModel):
        actions: list[TaskAction] = Field(
            ..., description="The sequence of actions to perform"
        )

    def send_llm_message(
        messages: list[Message],
        response_model: type[BaseModel],
    ) -> BaseModel:
        return LLMClient().generate_response(messages, response_model)

    print("Generating task plan...")
    task_prompt = f"""
    You are an action planner for a task user wants to perform on an app. Using the image attached, convert the user raw instruction into a precise sequence of actions needed to complete the task.

    ### User Task
    "{goal}"
    """
    task_plan: TaskPlan = send_llm_message(
        [ImageMessage(text=task_prompt, png_bytes=annotated_img_bytes)],
        TaskPlan,
    )
    print("Got task plan...")
    print(task_plan)
    actions = task_plan.actions
    prompt = f"""
    You are an action planner. Convert the user task into a precise sequence of UI actions.

    ### User Task
    {json.dumps([action.model_dump() for action in actions])}

    ### UI Elements (element | text | center coordinates)
    {format_elements_for_prompt(gui_elements)}

    ### Instructions
    1. Identify which elements are needed for the task using their center coordinates.
    2. Generate exactly the actions needed to complete the task — no more, no less.
    3. Do NOT repeat the same action consecutively unless the task genuinely requires it.
    """

    actions = send_llm_message(
        [ImageMessage(text=prompt, png_bytes=annotated_img_bytes)],
        ActionPlan,
    )
    print("Got actions...")
    print(actions)
    print("Executing Actions...")
    action_orchestrator = ActionOrchestrator(device_control)
    action_orchestrator.orchestrate(actions)
    time.sleep(3)
    print("Task completed successfully!")

    # --- cleanup ---
    session.close()


if __name__ == "__main__":
    main()
