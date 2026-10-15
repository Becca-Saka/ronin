"""Execution capabilities available to generated QA scenarios."""

from ronin.testing.models import GeneratedTestCase


_UNSUPPORTED_REQUIREMENTS = {
    "network inspection": "network traffic capture is not configured",
    "https": "HTTP transport inspection is not configured",
    "http header": "HTTP request inspection is not configured",
    "authorization header": "HTTP request inspection is not configured",
    "accessibility tree": "semantic accessibility-tree inspection is not configured",
    "accessible label": "semantic accessibility-tree inspection is not configured",
    "accessible name": "semantic accessibility-tree inspection is not configured",
    "screen reader": "screen-reader verification is not configured",
    "focus order": "keyboard/screen-reader focus traversal is not configured",
    "contrast ratio": "deterministic contrast measurement is not configured",
    "wcag": "deterministic contrast measurement is not configured",
    "colour contrast": "deterministic contrast measurement is not configured",
    "color contrast": "deterministic contrast measurement is not configured",
}


def unsupported_reason(test_case: GeneratedTestCase) -> str | None:
    """Return why a scenario cannot be evidenced by the current runtime."""
    text = "\n".join(
        [*test_case.steps_instructions, *test_case.expected_outcomes]
    ).lower()
    reasons = {
        reason for phrase, reason in _UNSUPPORTED_REQUIREMENTS.items() if phrase in text
    }
    if not reasons:
        return None
    return "Unsupported evidence requirement(s): " + "; ".join(sorted(reasons))
