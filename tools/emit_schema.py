"""Writes ronin-schema.json from the models that validate a ronin.yaml.

Editors read the JSON Schema; Ronin reads the models. Generating one from the
other means a key cannot be offered by an editor and refused by a run.

    uv run python tools/emit_schema.py          # write
    uv run python tools/emit_schema.py --check  # fail if it is out of date
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ronin.schema import RoninFile  # noqa: E402

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "ronin-schema.json"
SCHEMA_ID = "https://raw.githubusercontent.com/Becca-Saka/ronin/master/ronin-schema.json"


def build() -> str:
    schema = RoninFile.model_json_schema()
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": SCHEMA_ID,
        **schema,
    }
    return json.dumps(schema, indent=2, sort_keys=False) + "\n"


def main() -> int:
    rendered = build()
    if "--check" in sys.argv:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.is_file() else ""
        if current != rendered:
            print(f"{SCHEMA_PATH.name} is out of date. Run: uv run python tools/emit_schema.py")
            return 1
        print(f"{SCHEMA_PATH.name} is up to date.")
        return 0
    SCHEMA_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
