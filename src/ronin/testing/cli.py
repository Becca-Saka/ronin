import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from ronin.agent.device.platform import Platform
from ronin.app_discovery import find_journey_files, resolve_app_path
from ronin.config import Config, load_config
from .journey_env import resolve, substitute
from .models import UserJourney
from .runner import RoninTestRunner

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
        parser.add_argument("--platform", choices=["android", "ios"], help="Target platform")
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
            raise SystemExit(
                f"No journey file found in {Path.cwd()}.\n"
                "Pass one: ronin test <journey.yaml>"
            )
        listed = "\n".join(f"  {path.name}" for path in candidates)
        raise SystemExit(
            f"Several journey files found in {Path.cwd()}:\n{listed}\n"
            "Pass the one you want: ronin test <journey.yaml>"
        )

    def load_journeys(self, filepath: Path) -> List[UserJourney]:
        """Loads one or more UserJourneys from a JSON or YAML file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Journey file not found: {filepath}")
            
        content = filepath.read_text(encoding="utf-8")
        
        if filepath.suffix in (".yaml", ".yml"):
            try:
                import yaml
                raw = yaml.safe_load(content)
            except ImportError:
                logger.warning("PyYAML not found, using simple fallback parser.")
                raw = self._parse_simple_yaml(content)
        else:
            raw = json.loads(content)
            
        self.raw_config = raw
        journeys = []
        
        # Support unified suite structure: dictionary with a 'journeys' list key
        if isinstance(raw, dict) and "journeys" in raw:
            raw_journeys = raw["journeys"]
            if isinstance(raw_journeys, list):
                for item in raw_journeys:
                    journeys.append(self._build_journey_object(item))
            else:
                raise ValueError("'journeys' key must be a list of journey dictionaries.")
        # Support list of journeys
        elif isinstance(raw, list):
            for item in raw:
                journeys.append(self._build_journey_object(item))
        # Support single journey dictionary
        elif isinstance(raw, dict):
            journeys.append(self._build_journey_object(raw))
        else:
            raise ValueError("Invalid journey file format: must be a list of journeys, a single journey dictionary, or a suite with a 'journeys' key.")
            
        return journeys

    def _build_journey_object(self, data: dict) -> UserJourney:
        values, secrets = resolve(data)
        description = data.get("journey", data.get("description", ""))
        return UserJourney(
            id=data.get("id", "journey_01"),
            name=data.get("name", "User Journey"),
            description=substitute(description, values),
            security_rules=[
                substitute(rule, values) for rule in data.get("security_rules", [])
            ],
            secrets=secrets,
        )

    def _parse_simple_yaml(self, content: str) -> dict:
        """Fallback simple YAML parser to parse top-level fields and simple list blocks."""
        data = {}
        lines = content.splitlines()
        i = 0
        current_list_key = None
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            # Check if it's a list item
            if stripped.startswith("-"):
                if current_list_key:
                    if current_list_key not in data:
                        data[current_list_key] = []
                    item_val = stripped[1:].strip().strip('"').strip("'")
                    data[current_list_key].append(item_val)
                i += 1
                continue
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                
                if val == ">" or val == "|":
                    block = []
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        if not next_line.strip():
                            block.append("")
                            i += 1
                            continue
                        indent = len(next_line) - len(next_line.lstrip())
                        if indent > 0:
                            block.append(next_line.strip())
                            i += 1
                        else:
                            break
                    data[key] = " ".join(block) if val == ">" else "\n".join(block)
                    current_list_key = None
                    continue
                else:
                    data[key] = val.strip('"').strip("'")
                    current_list_key = key
                    
            i += 1
        return data

    async def run(self, args_list: List[str] = None):
        """Parses arguments, executes all loaded journeys, prints a summary, and exits."""
        args = self.parser.parse_args(args_list)
        filepath = self._resolve_journey_file(args.journey_file)
        
        try:
            journeys = self.load_journeys(filepath)
        except Exception as e:
            logger.error("Failed to load journey file: %s", e)
            sys.exit(2)
            
        # Settings come from the flag if given, then the suite's device block,
        # then ronin.config (project/global files, env), then the defaults.
        device_cfg = {}
        if hasattr(self, "raw_config") and isinstance(self.raw_config, dict):
            device_cfg = self.raw_config.get("device", {})

        def setting(name: str, flag: Any, *journey_keys: str) -> Any:
            if flag is not None:
                return flag
            for key in journey_keys or (name,):
                if key in device_cfg:
                    return device_cfg[key]
            return self.config.get(name)

        platform_str = setting("platform", args.platform)
        # The journey file's directory counts as a search root, so a suite can
        # live beside the app it tests and be run from anywhere.
        app_path = resolve_app_path(
            setting("app_path", args.app_path),
            platform_str,
            Path.cwd(),
            filepath.resolve().parent,
        )
        device_name = setting("device", args.device, "emulator", "device")
        headless_val = _as_bool(setting("headless", args.headless))

        record_val = device_cfg.get("record", device_cfg.get("record_session", True))
        record_val = _as_bool(record_val)

        runner = RoninTestRunner(
            platform=Platform.ANDROID if platform_str == "android" else Platform.IOS,
            device_name=device_name,
            app_path=str(app_path),
            model=setting("model", args.model),
            assets_dir=Path(setting("output_dir", args.output_dir)),
            max_steps=int(setting("max_steps", args.max_steps)),
            settle_time=float(setting("settle_time", args.settle_time)),
            use_multimodal_vision=_as_bool(
                setting("use_multimodal_vision", args.use_multimodal_vision)
            ),
            headless=headless_val,
            fast_model=setting("fast_model", args.fast_model),
            record_session=record_val,
            replay=bool(args.replay),
        )
        
        logger.info("==================================================")
        logger.info("Ronin CLI: Executing test suite for %d journeys", len(journeys))
        logger.info("==================================================")
        
        all_passed = True
        results = []
        
        for journey in journeys:
            logger.info("Starting run for journey: %s (%s)", journey.name, journey.id)
            try:
                report = await runner.run_journey(journey)
                passed = (report.status == "PASSED")
                results.append((journey.id, journey.name, "PASSED" if passed else "FAILED"))
                if not passed:
                    all_passed = False
            except Exception as journey_err:
                logger.error("Error executing journey %s: %s", journey.id, journey_err)
                results.append((journey.id, journey.name, "ERROR"))
                all_passed = False
                
        logger.info("==================================================")
        logger.info("Execution Summary:")
        for j_id, j_name, status in results:
            status_icon = "🟢" if status == "PASSED" else "🔴"
            logger.info("  %s %s (%s): %s", status_icon, j_name, j_id, status)
        logger.info("==================================================")
        
        if all_passed:
            logger.info("All user journeys passed successfully.")
            sys.exit(0)
        else:
            logger.error("One or more user journeys failed or encountered errors.")
            sys.exit(1)


def main():
    cli = RoninCLI()
    asyncio.run(cli.run())

