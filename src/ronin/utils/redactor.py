"""Masking the values a journey declared sensitive, in text Ronin persists.

A journey names its values in `env` and says which of them are secret. That
declaration is the only thing Ronin masks. Guessing at a value from its shape,
or from the label beside it, hid text nobody asked to hide and made reports
unreadable where the value was ordinary.

Nothing here runs on what the planner reads. It sees the screen as it is: it
has to recognise the field captioned "Password" to type into it, and it cannot
audit a rule about an exposed card number while card numbers are hidden from
it.
"""


def safe_text(text, scrubber=None) -> str:
    """Text fit to persist, with the journey's declared secrets masked."""
    if not isinstance(text, str):
        return text if text is not None else ""
    return scrubber.scrub(text) if scrubber else text


def scrub_elements(elements, scrubber):
    """Masks declared secrets in detected element text, for persisted copies only.

    Never call this on elements headed for the planner. It reads the screen to
    navigate it and to audit it; both break if the text is altered first.
    """
    for element in elements or []:
        if element.text:
            element.text = scrubber.scrub(element.text)
    return elements
