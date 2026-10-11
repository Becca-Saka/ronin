"""Keeping secret values out of prompts and logs.

Labels are not secrets. A field captioned "Password" has to stay readable or
the planner cannot tell which field to type into — only the value entered
into such a field is worth hiding.
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
    """Redacts card numbers, CVVs and labelled credential values from free text."""
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


def redact_elements(elements):
    """Redacts secret values in detected GUI elements, leaving their labels intact."""
    if not elements:
        return elements

    for el in elements:
        if el.text:
            el.text = redact_text(el.text)
    return elements
