"""Pattern-based masking for text Ronin is about to persist.

Nothing here runs on what the planner reads. It sees the screen as it is — it
has to recognise the field captioned "Password" to type into it, and it cannot
audit a rule about an exposed card number while card numbers are hidden from
it. These patterns are a backstop behind a journey's declared secrets, for a
value an author forgot to declare.
"""

import re

SENSITIVE_PATTERNS = [
    # Catches Visa, Mastercard, Amex, and Discover card patterns with actual prefix validation
    (re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4,7}\b'), "[REDACTED_CARD]"),
    # General CVV pattern if preceded by cvv/cvc keywords
    (re.compile(r'(?i)\b(cvv|cvc|security code|verification code)[ \t]*:?[ \t]*\b\d{3,4}\b'), r"\1: [REDACTED_CVV]"),
    # A value spelled out after a credential label, e.g. "Password: hunter2"
    (re.compile(
        r'(?i)\b(password|passcode|pin|secret|otp|token|ssn|social security|iban|routing)\b'
        r'[ \t]*[:=][ \t]*(\S+)'
    ), r"\1: [REDACTED]"),
]


def redact_text(text: str) -> str:
    """Redacts card numbers, CVVs and labelled credential values from free text.

    A backstop for persisted text only, behind a journey's declared `secrets`:
    it catches a value the author forgot to declare. Never call it on what the
    planner reads — see the module docstring.
    """
    if not isinstance(text, str):
        return text if text is not None else ""
    for pattern, repl in SENSITIVE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def mask_typed_text(text: str) -> str:
    """Describes typed input without reproducing it.

    What gets typed is routinely a password or a card number, and action
    descriptions end up in logs, prompts, trails and reports. The planner
    already has the values in its instructions, so it loses nothing here.
    """
    if not isinstance(text, str) or not text:
        return ""
    return f"{len(text)} character(s)"


def safe_text(text, scrubber=None) -> str:
    """Text fit to persist: declared secrets masked, credential patterns caught.

    The order matters and is the same everywhere it is applied, so it is
    written once: the journey's own values first, then the patterns that catch
    a value nobody declared.
    """
    cleaned = redact_text(text)
    return scrubber.scrub(cleaned) if scrubber else cleaned


def redact_elements(elements):
    """Redacts secret values in detected GUI elements, leaving their labels intact."""
    if not elements:
        return elements

    for el in elements:
        if el.text:
            el.text = redact_text(el.text)
    return elements


def scrub_elements(elements, scrubber):
    """Masks declared secrets in detected element text, for persisted copies only.

    Never call this on elements headed for the planner. It reads the screen to
    navigate it and to audit it; both break if the text is altered first.
    """
    for element in elements or []:
        if element.text:
            element.text = scrubber.scrub(element.text)
    return elements
