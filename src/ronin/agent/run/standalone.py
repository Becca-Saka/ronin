"""One goal, against a device session this module opens and closes."""

from pathlib import Path

from ronin.agent.device.session import DeviceSession
from ronin.agent.device.platform import Platform
from ronin.agent.perception.screen import ScreenPerception
from ronin.agent.run.outcomes import DeviceRunResult


def run_goal(
    agent,
    goal: str,
    device_name: str,
    app_path: str,
    platform: Platform = Platform.ANDROID,
    assets_dir: Path = Path("assets"),
    headless: bool = True,
    record_session: bool = True,
    keep_device: bool = False,
    reuse_app: bool = False,
) -> DeviceRunResult:
    """Opens a session, runs the goal against it, and closes it either way."""
    session = DeviceSession(
        device_name=device_name,
        app_path=app_path,
        platform=platform,
        record_session=record_session,
        session_dir=assets_dir,
        keep_device=keep_device,
        reuse_app=reuse_app,
    )
    try:
        device_control = session.start(is_headless=headless)
        return agent.run(
            goal=goal,
            device_control=device_control,
            perception=ScreenPerception(session),
            session_dir=session.screenshots_dir,
        )
    finally:
        session.close()
