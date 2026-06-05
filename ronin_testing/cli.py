import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from device.platform import Platform
from .models import UserJourney
from .runner import RoninTestRunner

logger = logging.getLogger(__name__)

def parse_simple_yaml(content: str) -> dict:
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
        
        # Check for list items
        if stripped.startswith("-") and current_list_key:
            item = stripped[1:].strip().strip('"').strip("'")
            if current_list_key not in data or not isinstance(data[current_list_key], list):
                data[current_list_key] = []
            data[current_list_key].append(item)
            i += 1
            continue

        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            
            # Check for block indicator
            if val in (">", "|"):
                block = []
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() == "":
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

def load_journey_file(filepath: Path) -> UserJourney:
    """Loads a UserJourney from a JSON or YAML file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Journey file not found: {filepath}")
        
    content = filepath.read_text(encoding="utf-8")
    
    if filepath.suffix in (".yaml", ".yml"):
        try:
            import yaml
            raw = yaml.safe_load(content)
        except ImportError:
            logger.warning("PyYAML not found, using simple fallback parser.")
            raw = parse_simple_yaml(content)
    else:
        raw = json.loads(content)
        
    # Build UserJourney object
    return UserJourney(
        id=raw.get("id", "journey_01"),
        name=raw.get("name", "User Journey"),
        description=raw.get("journey", raw.get("description", "")),
        security_rules=raw.get("security_rules", [])
    )

async def run_cli_async(args):
    filepath = Path(args.journey_file)
    try:
        journey = load_journey_file(filepath)
    except Exception as e:
        logger.error("Failed to load journey file: %s", e)
        sys.exit(2)
        
    # Setup runner parameters
    runner = RoninTestRunner(
        platform=Platform.ANDROID if args.platform == "android" else Platform.IOS,
        device_name=args.device,
        app_path=args.app_path,
        model=args.model,
        assets_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        settle_time=args.settle_time,
        use_multimodal_vision=args.use_multimodal_vision,
        headless=args.headless,
        fast_model=args.fast_model
    )
    
    logger.info("Executing journey test suite for: %s", journey.name)
    report = await runner.run_journey(journey)
    
    # Evaluate exit code
    if report.status == "PASSED":
        logger.info("Journey E2E validation PASSED.")
        sys.exit(0)
    else:
        logger.error("Journey E2E validation FAILED.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Ronin QA Journey-Based Testing CLI")
    parser.add_argument("journey_file", help="Path to the journey YAML/JSON file")
    parser.add_argument("--platform", default="android", choices=["android", "ios"], help="Target platform")
    parser.add_argument("--device", default="Pixel_8a", help="Name of emulator/device")
    parser.add_argument("--app-path", default="example/testing_app/build/app/outputs/flutter-apk/app-debug.apk", help="App binary path")
    parser.add_argument("--output-dir", default="assets", help="Directory to save test outputs and reports")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per test scenario execution")
    parser.add_argument("--settle-time", type=float, default=2.0, help="Settle time after actions")
    parser.add_argument("--model", default="openrouter/google/gemini-2.5-flash", help="LiteLLM model to use")
    parser.add_argument("--fast-model", default=None, help="Cheaper/faster LiteLLM model for ReAct step planning")
    parser.add_argument("--use-multimodal-vision", action="store_true", help="Toggle LLM coordinate overrides")
    parser.add_argument("--headless", action="store_true", help="Run android emulator headlessly")
    
    args = parser.parse_args()
    
    # Execute async loop
    asyncio.run(run_cli_async(args))
