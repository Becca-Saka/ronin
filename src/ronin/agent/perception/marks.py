"""Numbering the detected elements so a target can be named rather than guessed.

The planner used to answer with coordinates, which meant an answer we could not
check: any pair of numbers is structurally valid, so an element the planner
could not find became a tap into empty space. Numbering the detections turns the
answer into a choice from a known set — an index that does not exist is caught
before the device is touched.

The screenshot labels and the prompt labels are generated here, from one pass,
so the two cannot drift apart.
"""

import logging
from copy import copy
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from ronin.agent.perception.elements import GUIElement

logger = logging.getLogger(__name__)

DUPLICATE_OVERLAP = 0.80
# How much of a box must sit inside another before it counts as part of it.
CONTAINED_FRACTION = 0.90
# A box bigger than this is a layout region, not a control, so it keeps its parts.
MAX_CONTAINER_FRACTION = 0.25
ROW_BAND = 25
BADGE_PAD = 3
BAND_COLOR = (36, 82, 168)
# Where the soft keyboard starts, and how short a label has to be to be a key.
KEYBOARD_TOP_FRACTION = 0.60
KEY_LABEL_LENGTH = 2


@dataclass(frozen=True)
class Mark:
    """One numbered element: what the model picks and what we tap.

    `point` can differ from the box centre. A collapsed control is drawn around
    its container but touched where its label is, which is the part that
    reliably responds — the centre of a wide list row often lands on padding.
    """

    index: int
    element: GUIElement
    point: Optional[tuple[int, int]] = None

    @property
    def center(self) -> tuple[int, int]:
        if self.point is not None:
            return self.point
        box = self.element.bounding_box
        return (box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2


def _overlap(a, b) -> float:
    left, top = max(a.x_min, b.x_min), max(a.y_min, b.y_min)
    right, bottom = min(a.x_max, b.x_max), min(a.y_max, b.y_max)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    areas = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (
        b.y_max - b.y_min
    )
    return intersection / float(areas - intersection or 1)


def _center(element) -> tuple[int, int]:
    box = element.bounding_box
    return (box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2


def split_keyboard(
    elements: List[GUIElement], keyboard_open: Optional[bool], screen_height: Optional[int]
) -> tuple[List[GUIElement], List[GUIElement]]:
    """Separates the soft keyboard's own keys from the app's controls.

    An open keyboard is most of what is on screen — 83 of 110 detections on a
    real sign-in screen — and numbering every letter would bury the handful of
    app controls under a wall of badges. The agent never needs them: text goes
    in through `type`, and Enter and Back are key events.

    YOLO has no class for a keyboard, so this is geometry: low on the screen
    with a label too short to be a sentence.
    """
    if not keyboard_open or not screen_height:
        return list(elements or []), []

    keyboard_top = int(screen_height * KEYBOARD_TOP_FRACTION)
    kept, keys = [], []
    for element in elements or []:
        box = element.bounding_box
        center_y = (box.y_min + box.y_max) // 2
        looks_like_key = (
            center_y >= keyboard_top
            and len((element.text or "").strip()) <= KEY_LABEL_LENGTH
        )
        (keys if looks_like_key else kept).append(element)

    # A second pass by position, because the label rule misses the wide keys —
    # "?123", "space", "return". Anything sitting within the keyboard's own
    # extent belongs to it whatever its label says.
    if keys:
        top = min(key.bounding_box.y_min for key in keys)
        bottom = max(key.bounding_box.y_max for key in keys)
        remaining = []
        for element in kept:
            center_y = (element.bounding_box.y_min + element.bounding_box.y_max) // 2
            (keys if top <= center_y <= bottom else remaining).append(element)
        kept = remaining
        logger.info("Keyboard is open — %d key(s) left unnumbered.", len(keys))
    return kept, keys


def without_keyboard_keys(
    elements: List[GUIElement], keyboard_open: Optional[bool], screen_height: Optional[int]
) -> List[GUIElement]:
    """The app's controls, with the soft keyboard's keys removed."""
    return split_keyboard(elements, keyboard_open, screen_height)[0]


def keyboard_band(keys: List[GUIElement], screen_width: int) -> Optional[tuple]:
    """The region the keyboard occupies, as one rectangle.

    Drawn so the bare strip where no numbers appear is explained rather than
    left as an unexplained hole: nothing is numbered here because the keyboard
    is covering it, and anything you are looking for behind it needs the
    keyboard dismissed first.
    """
    if not keys:
        return None
    boxes = [key.bounding_box for key in keys]
    return (
        0,
        min(box.y_min for box in boxes),
        screen_width,
        max(box.y_max for box in boxes),
    )


def annotate_keyboard_keys(
    elements: List[GUIElement], keyboard_open: Optional[bool], screen_height: Optional[int]
) -> List[GUIElement]:
    """Prefixes likely keyboard keys with [key] so the planner can tell them apart.

    Used where the keys are still offered as targets. Where they are numbered
    instead, see without_keyboard_keys — there they are not offered at all.
    """
    if not keyboard_open or not screen_height:
        return elements

    keyboard_top = int(screen_height * KEYBOARD_TOP_FRACTION)
    annotated = []
    for element in elements:
        center_y = (element.bounding_box.y_min + element.bounding_box.y_max) // 2
        text = (element.text or "").strip()
        if center_y >= keyboard_top and len(text) <= KEY_LABEL_LENGTH:
            element = copy(element)
            element.text = f"[key] {text}" if text else "[key]"
        annotated.append(element)
    return annotated


def _area(box) -> int:
    return max(1, (box.x_max - box.x_min) * (box.y_max - box.y_min))


def _contained_in(child, parent) -> bool:
    overlap_w = max(0, min(child.x_max, parent.x_max) - max(child.x_min, parent.x_min))
    overlap_h = max(0, min(child.y_max, parent.y_max) - max(child.y_min, parent.y_min))
    return (overlap_w * overlap_h) / _area(child) >= CONTAINED_FRACTION


def _is_one_control(parent: GUIElement, children: List[GUIElement]) -> bool:
    """Whether a box and its contents are one thing, or several things in a container.

    A field is a wrapper around a single value: no label of its own, one child
    that carries text. A list row is not — a restaurant card holds a name, a
    category, a distance and a Menu button, and folding those together loses
    the button, which is the only part that responds.
    """
    if (parent.text or "").strip():
        return False
    return sum(1 for child in children if (child.text or "").strip()) <= 1


def _collapse_nests(elements: List[GUIElement], screen_area: int) -> List[tuple]:
    """Folds a control's parts into the control, as one numbered thing.

    A text field is not one box on screen: it is a container holding an icon,
    the typed value, sometimes a label. All of them are boxes, so all of them
    are detected, and numbering each would offer four choices for one field —
    three of which land somewhere that may not respond.

    The container keeps the outline, since that is the thing being pointed at.
    The parts give it its name and the spot to touch, since an empty container
    is unnameable and its centre is often padding.
    """
    # Without a screen size the guard has to come from the elements themselves,
    # or a full-screen container would swallow every control on the page.
    if not screen_area:
        screen_area = max(
            (_area(el.bounding_box) for el in elements), default=0
        )
    limit = screen_area * MAX_CONTAINER_FRACTION
    ordered = sorted(elements, key=lambda el: _area(el.bounding_box), reverse=True)

    taken: set[int] = set()
    groups: List[tuple] = []
    for parent in ordered:
        if id(parent) in taken:
            continue
        if limit and _area(parent.bounding_box) > limit:
            continue
        children = [
            child
            for child in elements
            if child is not parent
            and id(child) not in taken
            and _contained_in(child.bounding_box, parent.bounding_box)
        ]
        if not children or not _is_one_control(parent, children):
            continue
        taken.add(id(parent))
        taken.update(id(child) for child in children)
        groups.append((parent, children))

    collapsed = []
    for element in elements:
        if id(element) in taken:
            continue
        collapsed.append((element, []))
    for parent, children in groups:
        collapsed.append((parent, children))
    return collapsed


def _named(parent: GUIElement, children: List[GUIElement]) -> GUIElement:
    """The container, wearing the names of its parts."""
    texts = [c.text.strip() for c in children if (c.text or "").strip()]
    if (parent.text or "").strip():
        texts.insert(0, parent.text.strip())
    if not texts:
        return parent
    named = copy(parent)
    named.text = " ".join(dict.fromkeys(texts))
    return named


def _touch_point(parent: GUIElement, children: List[GUIElement]) -> Optional[tuple]:
    for child in children:
        if (child.text or "").strip():
            return _center(child)
    return None


def build(elements: List[GUIElement], screen_area: int = 0) -> List[Mark]:
    """Numbers the elements in reading order, one number per control.

    Reading order rather than detection order so the numbers run down the screen
    the way a person would scan it, which makes a misread number land on a
    neighbour rather than somewhere unrelated.
    """
    kept: List[GUIElement] = []
    for element in elements or []:
        if any(
            element.text == other.text
            and _overlap(element.bounding_box, other.bounding_box) >= DUPLICATE_OVERLAP
            for other in kept
        ):
            continue
        kept.append(element)

    groups = _collapse_nests(kept, screen_area)
    groups.sort(key=lambda g: (_center(g[0])[1] // ROW_BAND, _center(g[0])[0]))
    return [
        Mark(
            index=position,
            element=_named(parent, children),
            point=_touch_point(parent, children),
        )
        for position, (parent, children) in enumerate(groups, 1)
    ]


def find(marks: List[Mark], index: int) -> Optional[Mark]:
    """The mark with this number, or None if the model named one that isn't there."""
    for mark in marks:
        if mark.index == index:
            return mark
    return None


def _region(element, height: int) -> str:
    _, cy = _center(element)
    if height <= 0:
        return "unknown"
    position = cy / height
    if position < 0.33:
        return "upper"
    if position < 0.66:
        return "middle"
    return "lower"


DETAIL_TEXT = "text"
DETAIL_NUMBERS = "numbers"
DETAIL_NONE = "none"


def describe(marks: List[Mark], height: int, detail: str = DETAIL_NUMBERS) -> str:
    """The list offered to the planner alongside the numbered screenshot.

    Never coordinates — that is the point. A coordinate in the prompt is one the
    model can echo back, and an echoed coordinate cannot be told from a guessed
    one. Vertical position survives as a coarse region, which is all the
    keyboard rules need.

    The detected text is off by default because the model is already looking at
    the screen the words are printed on, and our OCR reads English only: on a
    Chinese app it reported 推荐 as "m#" and 京东超市 as "55187", which is worse
    than saying nothing, since a confident wrong label gets reasoned about. The
    numbers still have to be listed — they bound what the model may answer.
    """
    if detail == DETAIL_NONE:
        return ""

    lines = []
    for mark in marks:
        parts = [f"{mark.index}"]
        if detail == DETAIL_TEXT:
            parts.append(mark.element.element)
            if mark.element.text:
                parts.append(f"'{mark.element.text}'")
        parts.append(_region(mark.element, height))
        lines.append(" | ".join(parts))
    return "\n".join(lines) or "None detected."


# A bare family name resolves on almost no machine. The numbers have to be
# legible to the model at the resolution it is shown the screenshot, so a
# scalable face matters more here than the choice of face.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "arial.ttf",
)


def _font(scale_ref: int):
    size = max(10, int(15 * (scale_ref / 1200.0)))
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 cannot scale the built-in face
        return ImageFont.load_default()


def _free_slot(box, text_w, text_h, placed, screen):
    """Finds somewhere for a badge that does not sit on another badge.

    Numbers that overlap each other are worse than no numbers: two legible
    digits read as one illegible one, and the model picks neither.
    """
    width, height = screen
    pad = BADGE_PAD * 2
    w, h = text_w + pad, text_h + pad
    candidates = (
        (box.x_min - w, box.y_min - h),  # outside the top-left corner
        (box.x_min, box.y_min - h),  # directly above
        (box.x_max, box.y_min - h),  # outside the top-right corner
        (box.x_min - w, box.y_min),  # to the left
        (box.x_max, box.y_min),  # to the right
        (box.x_min, box.y_min),  # inside, last resort
    )
    for x, y in candidates:
        x = max(0, min(x, width - w))
        y = max(0, min(y, height - h))
        slot = (x, y, x + w, y + h)
        if not any(
            slot[0] < other[2]
            and other[0] < slot[2]
            and slot[1] < other[3]
            and other[1] < slot[3]
            for other in placed
        ):
            return slot
    x = max(0, min(box.x_min, width - w))
    y = max(0, min(box.y_min, height - h))
    return (x, y, x + w, y + h)


def _draw_band(draw, band, font, screen) -> None:
    x0, y0, x1, y1 = band
    draw.rectangle([x0, y0, x1, y1], fill=BAND_COLOR + (36,))
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BAND_COLOR, width=4)

    label = "KEYBOARD"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    w, h = right - left + BADGE_PAD * 2, bottom - top + BADGE_PAD * 2
    # Bottom-right: the top-left of a keyboard is its most-used keys.
    bx, by = max(0, x1 - w), max(0, min(y1 - h, screen[1] - h))
    draw.rectangle([bx, by, bx + w, by + h], fill=BAND_COLOR)
    draw.text(
        (bx + BADGE_PAD - left, by + BADGE_PAD - top), label, fill=(255, 255, 255), font=font
    )


def render(
    image: Image.Image,
    marks: List[Mark],
    color=(214, 16, 46),
    band: Optional[tuple] = None,
) -> Image.Image:
    """Draws each number near its element, avoiding other badges.

    Small and outside the box where there is room: the number has to be
    readable without hiding the thing it names, since a badge covering a label
    costs the planner the word it navigates by.
    """
    marked = image.copy()
    draw = ImageDraw.Draw(marked, "RGBA")
    screen = marked.size
    font = _font(max(screen))
    if band:
        _draw_band(draw, band, font, screen)
    outline = max(1, int(1.5 * (max(screen) / 1000.0)))
    placed = []

    for mark in marks:
        box = mark.element.bounding_box
        draw.rectangle(
            [box.x_min, box.y_min, box.x_max, box.y_max], outline=color, width=outline
        )

        label = str(mark.index)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        slot = _free_slot(box, right - left, bottom - top, placed, screen)
        placed.append(slot)

        draw.rectangle(list(slot), fill=color)
        draw.text(
            (slot[0] + BADGE_PAD - left, slot[1] + BADGE_PAD - top),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    return marked
