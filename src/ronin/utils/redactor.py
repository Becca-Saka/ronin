import re

SENSITIVE_PATTERNS = [
    # Catches Visa, Mastercard, Amex, and Discover card patterns with actual prefix validation
    (re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4,7}\b'), "[REDACTED_CARD]"),
    # General CVV pattern if preceded by cvv/cvc keywords
    (re.compile(r'(?i)\b(cvv|cvc|security code|verification code)[ \t]*:?[ \t]*\b\d{3,4}\b'), r"\1: [REDACTED_CVV]")
]

def redact_text(text: str) -> str:
    """Redacts card numbers and CVVs from free text strings."""
    if not isinstance(text, str):
        return text if text is not None else ""
    for pattern, repl in SENSITIVE_PATTERNS:
        text = pattern.sub(repl, text)
    return text

def redact_elements(elements):
    """Filters sensitive text values out of GUIElement instances using an expanded keyword list."""
    if not elements:
        return elements
    
    sensitive_keywords = [
        "password", "pin", "cvv", "card", "secret", "passcode",
        "token", "otp", "ssn", "account", "iban", "routing",
        "security code", "expiry", "expiration", "cvc", "dob",
        "date of birth", "social security"
    ]
    for el in elements:
        if el.text:
            el_name_lower = el.element.lower()
            text_lower = el.text.lower()
            
            # Check if element name or element text suggest it's a credentials field
            if any(kw in el_name_lower or kw in text_lower for kw in sensitive_keywords):
                el.text = "[REDACTED]"
            else:
                el.text = redact_text(el.text)
    return elements
