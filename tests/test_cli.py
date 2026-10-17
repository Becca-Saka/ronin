"""What the command line answers, and what it refuses to start."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _workflow(name: str = "", **settings):
    """One unnamed workflow holding one journey, as the loader would return it."""
    from ronin.testing.suite import Workflow

    return Workflow(name, settings, [SimpleNamespace(id="j", name="J")])


class ExitCodeTests(unittest.TestCase):
    """Only the entry point ends the process. Everything below answers with a code."""

    def test_a_usage_problem_is_reported_once_and_exits_two(self):
        from ronin.cli.main import main
        from ronin.errors import UsageError

        with (
            patch("ronin.cli.main.load_config", side_effect=UsageError("no app binary")),
            patch("sys.stderr", new=io.StringIO()) as captured,
        ):
            code = main(["doctor"])

        self.assertEqual(code, 2)
        self.assertIn("no app binary", captured.getvalue())

    def test_an_unloadable_suite_answers_two_without_exiting(self):
        from ronin.testing.cli import RoninCLI

        cli = RoninCLI()

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.yaml"
            code = cli.run([str(missing)])

        self.assertEqual(code, 2)

    def test_no_journey_file_is_a_usage_error_not_an_exit(self):
        from ronin.errors import UsageError
        from ronin.testing.cli import RoninCLI

        cli = RoninCLI()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("ronin.testing.cli.Path.cwd", return_value=Path(directory)),
            self.assertRaises(UsageError),
        ):
            cli._resolve_journey_file(None)

    def test_a_setting_ronin_config_does_not_know_is_not_looked_up_there(self):
        """grounding comes from a flag or the journey, or falls back to the default."""
        from ronin.agent.plan.planner import GROUNDING_COORDINATES
        from ronin.testing.cli import RoninCLI
        from ronin.testing.models import JourneyReport

        cli = RoninCLI()
        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )

        with (
            patch(
                "ronin.testing.cli.load_workflows",
                return_value=[_workflow()],
            ),
            patch.object(RoninCLI, "_resolve_journey_file", return_value=Path("j.yaml")),
            patch("ronin.testing.cli.resolve_app_path", return_value=Path("app.apk")),
            patch("ronin.testing.cli.RoninTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            cli.run(["j.yaml"])

        self.assertEqual(
            runner_class.call_args.kwargs["grounding"], GROUNDING_COORDINATES
        )

    def test_the_suite_result_decides_the_code(self):
        from ronin.testing.cli import RoninCLI
        from ronin.testing.models import JourneyReport

        def report(status):
            return JourneyReport(
                journey_id="j", journey_name="J", description="d", status=status
            )

        for status, expected in (("PASSED", 0), ("FAILED", 1)):
            cli = RoninCLI()
            with (
                patch(
                "ronin.testing.cli.load_workflows",
                return_value=[_workflow()],
            ),
                patch.object(RoninCLI, "_resolve_journey_file", return_value=Path("j.yaml")),
                patch("ronin.testing.cli.resolve_app_path", return_value=Path("app.apk")),
                patch("ronin.testing.cli.RoninTestRunner") as runner_class,
            ):
                runner_class.return_value.run_journey.return_value = report(status)
                self.assertEqual(cli.run(["j.yaml"]), expected)


class ConfigFileTests(unittest.TestCase):
    """A setting that does nothing must not look like one that was applied."""

    def _write(self, text):
        path = Path(tempfile.mkdtemp()) / "ronin.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_misspelled_setting_is_named_and_the_rest_still_load(self):
        from ronin.config import read_file

        path = self._write("platfrom: ios\nmodel: a/model\n")

        with self.assertLogs("ronin.config", level="WARNING") as captured:
            settings = read_file(path)

        self.assertEqual(settings, {"model": "a/model"})
        self.assertIn("platfrom", captured.output[0])

    def test_a_file_of_known_settings_warns_about_nothing(self):
        from ronin.config import read_file

        path = self._write("platform: ios\nmax_steps: 7\n")

        with self.assertNoLogs("ronin.config", level="WARNING"):
            settings = read_file(path)

        self.assertEqual(settings, {"platform": "ios", "max_steps": 7})


class ReuseAppFlagTests(unittest.TestCase):
    """Skipping the reinstall, which only `ronin run` could do before."""

    def _runner_kwargs(self, argv):
        from ronin.testing.cli import RoninCLI
        from ronin.testing.models import JourneyReport

        cli = RoninCLI()
        report = JourneyReport(
            journey_id="j", journey_name="J", description="d", status="PASSED"
        )
        with (
            patch(
                "ronin.testing.cli.load_workflows",
                return_value=[_workflow()],
            ),
            patch.object(RoninCLI, "_resolve_journey_file", return_value=Path("j.yaml")),
            patch("ronin.testing.cli.resolve_app_path", return_value=Path("app.apk")),
            patch("ronin.testing.cli.RoninTestRunner") as runner_class,
        ):
            runner_class.return_value.run_journey.return_value = report
            cli.run(argv)
        return runner_class.call_args.kwargs

    def test_the_app_is_reinstalled_by_default(self):
        """A stale build would be tested silently, so this is opt-in."""
        self.assertFalse(self._runner_kwargs(["j.yaml"])["reuse_app"])

    def test_the_flag_reaches_the_runner(self):
        self.assertTrue(self._runner_kwargs(["j.yaml", "--reuse-app"])["reuse_app"])


class PlatformNameTests(unittest.TestCase):
    """A platform Ronin does not know stops the run."""

    def test_a_known_name_resolves(self):
        from ronin.agent.device.platform import Platform

        self.assertIs(Platform.named("android"), Platform.ANDROID)
        self.assertIs(Platform.named("iOS"), Platform.IOS)

    def test_a_misspelled_platform_is_refused(self):
        """'androdi' used to run the iOS path, failing far from the typo."""
        from ronin.agent.device.platform import Platform
        from ronin.errors import UsageError

        with self.assertRaises(UsageError) as caught:
            Platform.named("androdi")

        self.assertIn("androdi", str(caught.exception))
