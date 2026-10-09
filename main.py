"""Deprecated entrypoint — use the `ronin` command instead.

Kept so `uv run main.py ...` keeps working for existing scripts. It forwards
to ronin.cli.main, translating the old `test-journey` subcommand to `test`.
"""

import sys

from ronin.cli.main import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    print(
        "main.py is deprecated; run 'uv run ronin' (or 'ronin' once installed) instead.",
        file=sys.stderr,
    )
    argv = ["test" if arg == "test-journey" else arg for arg in argv]
    sys.exit(main(argv))
