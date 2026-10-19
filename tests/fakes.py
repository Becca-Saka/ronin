"""Builders for the values Ronin's boundaries hand to its logic.

Each one calls the real constructor. Rename or remove a field and the tests
fail to build, rather than keep passing against a shape that production no
longer uses.
"""

import io

from PIL import Image

from ronin.agent.step.observation import PerceptionSnapshot
from ronin.agent.perception.elements import BoundingBox, GUIElement
from ronin.agent.grounding.marks import Mark

STAMP = "2026-01-01T00:00:00"


def solid_png(fill: str = "white", size: tuple[int, int] = (200, 400)) -> bytes:
    """A screenshot of one colour, for comparing a screen against itself."""
    buffer = io.BytesIO()
    Image.new("RGB", size, fill).save(buffer, format="PNG")
    return buffer.getvalue()


def element(
    box: tuple[int, int, int, int] = (10, 10, 60, 60),
    name: str = "Button",
    text: str = "Continue",
    confidence: float = 0.9,
    kind: str = "yolo_detection",
) -> GUIElement:
    """One element, as the detector reports it."""
    return GUIElement(
        element=name,
        type=kind,
        confidence=confidence,
        text=text,
        bounding_box=BoundingBox(
            x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
        ),
    )


def numbered(*elements: GUIElement) -> list[Mark]:
    """The numbered targets offered to the planner, numbered in the order given."""
    return [Mark(index=index, element=item) for index, item in enumerate(elements, 1)]


def snapshot(marks=(), gui_elements=(), png=None, phash="p") -> PerceptionSnapshot:
    """What perception hands the step loop."""
    screen = solid_png() if png is None else png
    return PerceptionSnapshot(
        gui_elements=list(gui_elements),
        marks=list(marks),
        raw_img_bytes=screen,
        planner_img_bytes=screen,
        annotated_img=Image.new("RGB", (1, 1)),
        current_phash=phash,
        timestamp=STAMP,
    )


def qa_prompt(goal_lines=("Sign in",), setup=None):
    """The prompt a real QAPlanner builds, through its real constructor.

    Through the constructor, so a field it stopped taking cannot be assigned
    back in and go on looking present.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ronin.agent.device.launch import SetupMode
    from ronin.testing.qa_planner import QAPlanner

    llm = SimpleNamespace(generate_response=Mock())
    planner = QAPlanner(llm_client=llm, setup=setup or SetupMode.CLEAR_DATA)
    return planner._build_qa_prompt(
        goal_lines=list(goal_lines),
        gui_elements=None,
        history="",
        trail=None,
        drift_warning=None,
        loop_nudge=None,
    )


def walk(*instructions: str):
    """Prose actions as the typed steps a scenario carries.

    Built through the real step models, so an instruction a step would refuse
    is refused here too rather than reaching a test as a bare string.
    """
    from ronin.testing.steps import ActionStep

    return [ActionStep(instruction=text) for text in instructions]
