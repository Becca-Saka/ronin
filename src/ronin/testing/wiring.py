"""Building the runner one workflow needs, from the settings that apply to it.

A setting can come from four places, and the order matters. The flag the user
typed wins, then the workflow block, then ronin.yaml and the RONIN_* variables,
then the built-in default. Resolving that in one place keeps every caller from
re-deciding it.
"""

import logging
from pathlib import Path
from typing import Any

from ronin.agent.device.platform import Platform
from ronin.agent.plan.planner import GROUNDING_COORDINATES
from ronin.app_discovery import resolve_app_path
from ronin.config import SETTINGS, Config

from .runner import RoninTestRunner
from .suite import Workflow

logger = logging.getLogger(__name__)


def as_bool(raw: Any) -> bool:
    """A setting read as a flag, whichever spelling a file or a variable used."""
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def runner_for(
    workflow: Workflow, args: Any, filepath: Path, config: Config
) -> RoninTestRunner:
    """The runner for one workflow, with every setting resolved in precedence order."""

    def setting(name: str, flag: Any) -> Any:
        if flag is not None:
            return flag
        if name in workflow.settings:
            return workflow.settings[name]
        # grounding is a flag and a file key only, so the caller supplies the
        # default rather than ronin.config.
        return config.get(name) if name in SETTINGS else None

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
        headless=as_bool(setting("headless", args.headless)),
        fast_model=setting("fast_model", args.fast_model),
        record_session=(False if args.no_record else as_bool(setting("record", None))),
        replay=bool(args.replay),
        reuse_app=bool(args.reuse_app),
        keep_app=bool(args.keep_app),
        grounding=setting("grounding", args.grounding) or GROUNDING_COORDINATES,
        journey_file=filepath,
        workflow=workflow.name or "",
        locked=bool(args.locked),
        persist_inputs=not bool(args.no_persist_inputs),
        record_secrets=bool(args.record_secrets),
    )
