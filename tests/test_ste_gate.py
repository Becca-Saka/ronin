"""What the controlled-language gate reads, and what it counts once.

The gate is only as good as its extraction. Two defects it had made it
report on prose nobody wrote and stay silent about prose a model reads: it
counted every docstring twice, and it read neither a prompt constant nor a
model-facing field description.
"""

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ste_gate  # noqa: E402


def _written(tmp: Path, body: str) -> Path:
    path = tmp / "sample.py"
    path.write_text(body)
    return path


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())

    def test_a_docstring_is_read_once(self):
        """`get_docstring` dedents and the raw literal does not, so both reach
        the linter and every sentence is counted twice."""
        path = _written(self.tmp, '"""One sentence about one thing."""\n')

        prose = ste_gate.prose_of(path)

        self.assertEqual(prose.count("One sentence about one thing."), 1)

    def test_a_nested_docstring_is_read_once(self):
        path = _written(
            self.tmp,
            'class Thing:\n'
            '    """What this thing is for."""\n\n'
            '    def act(self):\n'
            '        """What acting does."""\n',
        )

        prose = ste_gate.prose_of(path)

        self.assertEqual(prose.count("What this thing is for."), 1)
        self.assertEqual(prose.count("What acting does."), 1)

    def test_a_prompt_constant_is_read(self):
        """A prompt reaches a model, so it is the prose that matters most."""
        path = _written(
            self.tmp,
            'PROMPT = """You are a reader. Answer with one sentence about the '
            'screen in front of you."""\n',
        )

        self.assertIn("You are a reader.", ste_gate.prose_of(path))

    def test_a_model_facing_description_is_read(self):
        path = _written(
            self.tmp,
            'from pydantic import Field\n'
            'x = Field(description="What must be true for this to pass.")\n',
        )

        self.assertIn("What must be true for this to pass.", ste_gate.prose_of(path))

    def test_a_short_assigned_string_is_not_prose(self):
        """A name, a path and a log key are strings too."""
        path = _written(self.tmp, 'LINTER_ENV = "RONIN_STE_LINT"\nSEP = ", "\n')

        self.assertEqual(ste_gate.prose_of(path).strip(), "")

    def test_a_comment_is_read_and_machinery_is_not(self):
        path = _written(
            self.tmp,
            "# A real comment about a real reason.\n"
            "import os  # noqa: F401\n",
        )

        prose = ste_gate.prose_of(path)

        self.assertIn("A real comment about a real reason.", prose)
        self.assertNotIn("noqa", prose)

    def test_a_file_with_no_prose_yields_nothing(self):
        path = _written(self.tmp, "VALUE = 1\n")

        self.assertEqual(ste_gate.prose_of(path).strip(), "")


class NamingTests(unittest.TestCase):
    def test_a_path_outside_the_repository_keeps_its_own_spelling(self):
        """`relative_to` raises on one, and a gate that crashes on the file it
        was handed reports nothing about the files it could have read."""
        outside = Path("/tmp/elsewhere/sample.py")

        self.assertEqual(ste_gate._named(outside), str(outside))

    def test_a_path_inside_the_repository_is_named_relative_to_it(self):
        inside = ste_gate.REPOSITORY_ROOT / "src" / "ronin" / "errors.py"

        self.assertEqual(ste_gate._named(inside), "src/ronin/errors.py")

    def test_a_relative_path_is_left_alone(self):
        self.assertEqual(ste_gate._named(Path("src/ronin/errors.py")), "src/ronin/errors.py")


class BaselineTests(unittest.TestCase):
    def test_the_baseline_names_only_files_that_exist(self):
        """An entry for a deleted file hides nothing and reads as live debt."""
        for name in ste_gate._load_baseline():
            self.assertTrue((ste_gate.REPOSITORY_ROOT / name).is_file(), name)

    def test_no_baseline_entry_allows_nothing(self):
        """A zero entry is the same as absence, so it is noise."""
        for name, allowed in ste_gate._load_baseline().items():
            self.assertGreater(allowed, 0, name)


if __name__ == "__main__":
    unittest.main()
