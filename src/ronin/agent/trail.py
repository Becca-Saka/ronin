"""
Session-scoped navigation trail — accumulated per-screen knowledge injected into
each planner call so subsequent scenarios don't re-discover known paths from scratch.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

_INTERACTIVE_ELEMENT_TYPES = {
    "button",
    "text_field",
    "checkbox",
    "switch",
    "slider",
    "icon_button",
    "fab",
    "tab",
    "dropdown",
    "radio_button",
    "toggle",
    "search_field",
}


@dataclass
class TrailElement:
    """One detected UI element stored in the trail as structured data."""

    element_type: str
    text: Optional[str]
    cx: int
    cy: int
    is_interactive: bool
    element_hash: str  # sha256(type + text + bucketed cx/cy), ~40px grid tolerance


@dataclass
class ScreenEntry:
    """Everything Ronin has learned about one screen across all scenarios."""

    screen_name: str
    elements: List[TrailElement] = field(default_factory=list)
    navigation_sequences: List[List[str]] = field(default_factory=list)
    visit_count: int = 1


class SessionTrail:
    """Session-scoped navigation knowledge injected into each planner call."""

    def __init__(self):
        self._screens: dict[str, ScreenEntry] = {}
        self._pending_phash: Optional[str] = None
        self._pending_actions: List[str] = []

    def _find_matching_key(self, phash_hex: str, threshold: int = 10) -> Optional[str]:
        """Return the existing _screens key within pHash Hamming distance, or None."""
        import imagehash

        try:
            target = imagehash.hex_to_hash(phash_hex)
        except Exception:
            return None
        for key in self._screens:
            try:
                if target - imagehash.hex_to_hash(key) <= threshold:
                    return key
            except Exception:
                continue
        return None

    def record_screen(self, phash_hex: str, screen_name: str, raw_elements) -> None:
        from ronin.utils.redactor import redact_text

        matched = self._find_matching_key(phash_hex)
        canonical = matched if matched is not None else phash_hex

        if self._pending_phash is None:
            self._pending_phash = canonical

        trail_elements: List[TrailElement] = []
        for el in raw_elements or []:
            cx = (el.bounding_box.x_min + el.bounding_box.x_max) // 2
            cy = (el.bounding_box.y_min + el.bounding_box.y_max) // 2
            el_text = redact_text(el.text) if el.text else None
            el_hash = hashlib.sha256(
                f"{el.element}{el_text or ''}{cx // 40}{cy // 40}".encode()
            ).hexdigest()[:16]
            trail_elements.append(
                TrailElement(
                    element_type=el.element,
                    text=el_text,
                    cx=cx,
                    cy=cy,
                    is_interactive=el.element in _INTERACTIVE_ELEMENT_TYPES,
                    element_hash=el_hash,
                )
            )

        if matched is None:
            self._screens[phash_hex] = ScreenEntry(
                screen_name=screen_name,
                elements=trail_elements,
            )
        else:
            entry = self._screens[matched]
            if screen_name:
                entry.screen_name = screen_name
            entry.elements = trail_elements
            entry.visit_count += 1

    def record_action(self, action_desc: str) -> None:
        from ronin.utils.redactor import redact_text

        self._pending_actions.append(redact_text(action_desc))

    def flush_transition(self, new_phash: str) -> None:
        if self._pending_phash and self._pending_actions:
            key = self._find_matching_key(self._pending_phash) or self._pending_phash
            if key in self._screens:
                self._screens[key].navigation_sequences.append(
                    list(self._pending_actions)
                )
        self._pending_actions = []
        self._pending_phash = self._find_matching_key(new_phash) or new_phash

    def format_for_prompt(self) -> Optional[str]:
        from ronin.utils.redactor import redact_text

        if not self._screens:
            return None

        lines = [
            "Screens already visited this session. Use element [indices] when planning actions.",
            "You do not need to re-discover these screens.",
        ]

        for entry in self._screens.values():
            visit_label = f"(visited {entry.visit_count} time{'s' if entry.visit_count != 1 else ''})"
            lines.append(f"\n{entry.screen_name or 'Unknown Screen'} {visit_label}:")

            interactive = [e for e in entry.elements if e.is_interactive]
            passive = [e for e in entry.elements if not e.is_interactive]
            ordered = (interactive + passive)[:20]

            for idx, el in enumerate(ordered):
                text_part = f' | "{el.text}"' if el.text else ""
                lines.append(
                    f"  [{idx}] {el.element_type}{text_part} | ({el.cx}, {el.cy})"
                )

            if entry.navigation_sequences:
                seq = entry.navigation_sequences[-1]
                lines.append(f"  Navigation: {' → '.join(seq)}")

        return redact_text("\n".join(lines))
