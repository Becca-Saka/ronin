"""The order a run commits its money in."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class SpendOrderTests(unittest.TestCase):
    """Scenario generation is the most expensive call a run makes.

    It is also useless if the device never comes up, so the device has to be
    proven first. A boot failure after generation costs the suite to discover.
    """

    def _run(self):
        from ronin.testing.models import UserJourney
        from ronin.testing.runner import RoninTestRunner

        order = []

        class Session:
            def __init__(self, *_, **__):
                self.telemetry_dir = Path(tempfile.mkdtemp())
                self.screenshots_dir = self.telemetry_dir
                self.session_id = "s1"
                self.installed_by_run = False
                self.root_log_file = self.telemetry_dir / "run.log"
                self.appium_log_path = self.telemetry_dir / "appium.log"

            def start(self, **_):
                order.append("device")
                return SimpleNamespace()

            def checkpoint_recording(self):
                pass

            def close(self, **_):
                pass

        class Generator:
            def __init__(self, *_, **__):
                pass

            def generate_scenarios(self, _journey):
                order.append("generator")
                raise RuntimeError("stop here")

        with (
            patch("ronin.testing.runner.DeviceSession", Session),
            patch("ronin.testing.runner.ScenarioGenerator", Generator),
            patch("ronin.testing.runner.QAAgent"),
            patch("ronin.testing.runner.ensure_device_ready") as ready,
            patch("ronin.testing.runner.for_grounding"),
        ):
            ready.side_effect = lambda *_: order.append("asked adb")
            RoninTestRunner(device_name="Pixel_8", app_path="app.apk").run_journey(
                UserJourney(id="j", name="J", description="A journey")
            )
        return order

    def test_the_device_is_open_before_the_generator_is_asked(self):
        self.assertEqual(self._run(), ["asked adb", "device", "generator"])
