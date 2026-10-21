"""Terminal output shared by the CLI commands.

Every command reports the same shapes — a section heading, a pass/warn/fail
line, an indented fix hint — so the glyphs, colours and tty detection live
here rather than being re-declared per command.
"""

import logging
import os
import sys
from typing import Optional, TextIO

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"
INFO = "info"

GLYPHS = {OK: "✓", WARN: "!", FAIL: "✗", SKIP: "-", INFO: "·"}
LEVEL_COLORS = {OK: GREEN, WARN: YELLOW, FAIL: RED, SKIP: DIM, INFO: DIM}


class Console:
    """Writes styled CLI output, dropping the colour when nobody can see it.

    Honours the NO_COLOR and FORCE_COLOR conventions so piped output and CI
    logs stay readable.
    """

    def __init__(self, stream: Optional[TextIO] = None, color: Optional[bool] = None):
        self.stream = stream if stream is not None else sys.stdout
        self.color = self._detect_color() if color is None else color

    def _detect_color(self) -> bool:
        if os.getenv("NO_COLOR"):
            return False
        if os.getenv("FORCE_COLOR"):
            return True
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def paint(self, text: str, *codes: str) -> str:
        """Wraps text in ANSI codes, or returns it untouched when colour is off."""
        if not self.color or not codes:
            return text
        return f"{''.join(codes)}{text}{RESET}"

    def write(self, text: str = "") -> None:
        # Unbuffered: install steps interleave their own subprocess output.
        print(text, file=self.stream, flush=True)

    def title(self, text: str) -> None:
        self.write(self.paint(text, BOLD))

    def heading(self, text: str) -> None:
        """A section heading, preceded by a blank line."""
        self.write()
        self.write(self.paint(text, BOLD))

    def result(
        self,
        level: str,
        label: str,
        detail: str = "",
        hint: Optional[str] = None,
        indent: int = 2,
    ) -> None:
        """One status line, with its fix hint indented underneath when present."""
        glyph = self.paint(GLYPHS[level], LEVEL_COLORS[level])
        suffix = f" {detail}" if detail else ""
        self.write(f"{' ' * indent}{glyph} {label}:{suffix}")
        if hint:
            self.hint(hint, indent=indent + 4)

    def hint(self, text: str, indent: int = 6) -> None:
        self.write(f"{' ' * indent}{self.paint(text, DIM)}")

    def confirm(self, prompt: str, assume_yes: bool = False) -> bool:
        """Asks a yes/no question, reading the answer from stdin wherever it comes from.

        Piped answers work (`yes | ronin setup`); a closed or empty stdin reads
        as EOF and declines, so an unattended run never blocks or assumes yes.
        """
        if assume_yes:
            return True
        try:
            answer = input(f"  {self.paint('?', YELLOW)} {prompt} [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.write()
            return False
        if not sys.stdin.isatty():
            self.write(answer)
        return answer in ("", "y", "yes")

    def count(self, level: str, text: str) -> None:
        """A coloured tally line, e.g. '2 warning(s):'."""
        self.write(self.paint(text, LEVEL_COLORS[level]))

    def bullet(self, level: str, text: str, indent: int = 2) -> None:
        self.write(f"{' ' * indent}{GLYPHS[level]} {text}")


def configure_logging(level: str) -> None:
    """Sets the verbosity for the whole process.

    `force=True` because a subcommand may set this after `ronin` already did,
    and the later word is the one the user typed closest to the command.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
