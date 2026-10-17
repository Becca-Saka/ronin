import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from ronin.agent.device.platform import Platform
from ronin.agent.plan.planner import GROUNDING_COORDINATES
from ronin.app_discovery import find_journey_files, resolve_app_path
from ronin.config import SETTINGS, Config, load_config
from ronin.errors import UsageError
from ronin.schema import GROUNDINGS, PLATFORMS
from .runner import RoninTestRunner
from .suite import Workflow, load_workflows, select

logger = logging.getLogger(__name__)


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class RoninCLI:
    """Encapsulates the CLI parsing, test suite execution, and summary reporting for Ronin."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Ronin QA Journey-Based Testing CLI")
        parser.add_argument(
            "journey_file",
            nargs="?",
            help="Path to the journey YAML/JSON file. Found in this directory when omitted.",
        )
        parser.add_argument("--platform", choices=list(PLATFORMS), help="Target platform")
        parser.add_argument("--device", help="Name of emulator/device")
        parser.add_argument("--app-path", help="App binary path")
        parser.add_argument("--output-dir", help="Directory to save test outputs and reports")
        parser.add_argument("--max-steps", type=int, help="Max steps per test scenario execution")
        parser.add_argument("--settle-time", type=float, help="Settle time after actions")
        parser.add_argument("--model", help="LiteLLM model to use")
        parser.add_argument("--fast-model", help="Cheaper/faster LiteLLM model for ReAct step planning")
        parser.add_argument("--use-multimodal-vision", action="store_true", default=None, help="Toggle LLM coordinate overrides")
        parser.add_argument("--headless", action="store_true", default=None, help="Run android emulator headlessly")
        parser.add_argument(
            "--grounding",
            choices=list(GROUNDINGS),
            help="How the planner names a target: 'coordinates' (default) or "
            "'marks', where every detected element is numbered on the screenshot "
            "and the planner selects a number, which Ronin validates against the "
            "detected set before anything is executed",
        )
        parser.add_argument(
            "--reuse-app",
            action="store_true",
            help="Skip reinstalling the app, using whatever build is on the device. "
            "Only correct when that build is the one under test: a rebuilt debug APK "
            "keeps its version, so a stale one is used without warning",
        )
        parser.add_argument(
            "--workflow",
            help="Which workflow(s) in the file to run, comma separated. "
            "Every workflow runs when this is not given",
        )
        parser.add_argument(
            "--no-record",
            action="store_true",
            help="Do not record the screen. The run reports results either way, "
            "but no video is kept",
        )
        parser.add_argument(
            "--keep-app",
            action="store_true",
            help="Leave the app installed when the run ends. By default a run uninstalls "
            "the app it installed, so a rotation of apps does not fill the device. An app "
            "that was already on the device is never removed",
        )
        parser.add_argument(
            "--replay",
            action="store_true",
            help="Re-execute navigation already worked out in an earlier scenario instead of "
            "re-planning it. Faster, but replayed screens are not perceived or audited",
        )
        return parser

    def _resolve_journey_file(self, given: Optional[str]) -> Path:
        """Finds the journey file to run, looking in the working directory."""
        if given:
            return Path(given)

        candidates = find_journey_files(Path.cwd())
        if len(candidates) == 1:
            logger.info("Using journey file %s", candidates[0].name)
            return candidates[0]
        if not candidates:
            raise UsageError(
                f"No journey file found in {Path.cwd()}.\n"
                "Pass one: ronin test <journey.yaml>"
            )
        listed = "\n".join(f"  {path.name}" for path in candidates)
        raise UsageError(
            f"Several journey files found in {Path.cwd()}:\n{listed}\n"
            "Pass the one you want: ronin test <journey.yaml>"
        )

    def _runner_for(self, workflow: Workflow, args: Any, filepath: Path) -> "RoninTestRunner":
        """Builds the runner one workflow needs.

        Settings come from the flag if given, then the workflow, then
        ronin.yaml and the RONIN_* variables, then the defaults.
        """

        def setting(name: str, flag: Any) -> Any:
            if flag is not None:
                return flag
            if name in workflow.settings:
                return workflow.settings[name]
            # grounding is a flag and a file key only, so the caller supplies
            # the default rather than ronin.config.
            return self.config.get(name) if name in SETTINGS else None

        platform_str = setting("platform", args.platform)
        app_path = resolve_app_path(
            setting("app_path", args.app_path),
            platform_str,
            Path.cwd(),
            filepath.resolve().parent,
        )
        return RoninTestRunner(
            platform=Platform.named(platform_str),
            device_name=setting("device", args.device),
            app_path=str(app_path),
            model=setting("model", args.model),
            assets_dir=Path(setting("output_dir", args.output_dir)),
            max_steps=int(setting("max_steps", args.max_steps)),
            settle_time=float(setting("settle_time", args.settle_time)),
            use_multimodal_vision=_as_bool(
                setting("use_multimodal_vision", args.use_multimodal_vision)
            ),
            headless=_as_bool(setting("headless", args.headless)),
            fast_model=setting("fast_model", args.fast_model),
            record_session=(
                False if args.no_record else _as_bool(setting("record", None))
            ),
            replay=bool(args.replay),
            reuse_app=bool(args.reuse_app),
            keep_app=bool(args.keep_app),
            grounding=setting("grounding", args.grounding) or GROUNDING_COORDINATES,
        )

    def run(self, args_list: Optional[List[str]] = None) -> int:
        """Executes all loaded journeys, prints a summary, and returns a code.

        The caller owns the process. The code is:

        - 0 when every journey passed
        - 1 when any journey failed
        - 2 when Ronin could not load the suite
        """
        args = self.parser.parse_args(args_list)

        try:
            filepath = self._resolve_journey_file(args.journey_file)
            workflows = select(load_workflows(filepath), args.workflow)
        except UsageError as e:
            logger.error("%s", e)
            return 2
        except Exception as e:
            logger.error("Failed to load journey file: %s", e)
            return 2

        total = sum(len(workflow.journeys) for workflow in workflows)
        logger.info("==================================================")
        logger.info(
            "Ronin CLI: Executing %d journey(s) across %d workflow(s)",
            total,
            len(workflows),
        )
        logger.info("==================================================")

        all_passed = True
        results = []

        for workflow in workflows:
            if workflow.name:
                logger.info("Workflow: %s", workflow.name)
            try:
                runner = self._runner_for(workflow, args, filepath)
            except Exception as setup_err:
                logger.error(
                    "Could not prepare workflow '%s': %s", workflow.name, setup_err
                )
                for journey in workflow.journeys:
                    results.append((workflow.name, journey.id, journey.name, "ERROR"))
                all_passed = False
                continue

            for journey in workflow.journeys:
                logger.info("Starting run for journey: %s (%s)", journey.name, journey.id)
                try:
                    report = runner.run_journey(journey)
                    passed = (report.status == "PASSED")
                    results.append(
                        (workflow.name, journey.id, journey.name, "PASSED" if passed else "FAILED")
                    )
                    if not passed:
                        all_passed = False
                except Exception as journey_err:
                    logger.error("Error executing journey %s: %s", journey.id, journey_err)
                    results.append((workflow.name, journey.id, journey.name, "ERROR"))
                    all_passed = False

        logger.info("==================================================")
        logger.info("Execution Summary:")
        for flow_name, j_id, j_name, status in results:
            status_icon = "🟢" if status == "PASSED" else "🔴"
            label = f"{flow_name}/{j_name}" if flow_name else j_name
            logger.info("  %s %s (%s): %s", status_icon, label, j_id, status)
        logger.info("==================================================")

        if all_passed:
            logger.info("All user journeys passed successfully.")
            return 0
        logger.error("One or more user journeys failed or encountered errors.")
        return 1


def main():
    cli = RoninCLI()
    sys.exit(cli.run())
