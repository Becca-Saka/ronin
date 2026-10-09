import logging
import platform
import sys
from pathlib import Path

from ronin.utils.subprocess_runner import run_command

logger = logging.getLogger(__name__)


class PlatformError(Exception):
    pass


def _run(cmd: list[str], check=False):
    return run_command(cmd, capture_output=True, text=True, check=check)


def check_platform_for_ios():
    """Raises PlatformError if the current OS cannot run iOS automation."""
    system = platform.system()
    if system == "Windows":
        raise PlatformError(
            "iOS automation via XCUITest requires macOS. "
            "Windows is not supported — XCUITest only runs on Apple hardware."
        )
    if system == "Linux":
        raise PlatformError(
            "iOS automation via XCUITest requires macOS. "
            "Linux is not supported unless you are using a cloud Mac (e.g. AWS EC2 Mac)."
        )


def is_xcode_installed() -> bool:
    """Check if Xcode command line tools are installed."""
    result = _run(["xcode-select", "-p"])
    return result.returncode == 0 and bool(result.stdout.strip())


def install_xcode_cli():
    """
    Attempt to install Xcode CLI tools.
    Uses the interactive GUI path (xcode-select --install) for desktops,
    and the headless softwareupdate path for CI/SSH environments.
    """
    # Try the standard interactive install first
    logger.warning("Xcode Command Line Tools not found.")
    logger.warning("A system dialog will appear asking you to install them.")
    logger.warning("Please click 'Install' and accept the license agreement.")
    logger.warning("This may take 5-10 minutes.")

    confirm = input("Press Enter to continue or type 'skip' to abort: ").strip().lower()
    if confirm == "skip":
        raise PlatformError("Xcode installation skipped by user.")

    result = _run(["xcode-select", "--install"])

    if result.returncode != 0 and "already installed" not in result.stderr:
        # Fall back to headless softwareupdate (works over SSH/CI)
        logger.info("GUI install unavailable, trying headless install via softwareupdate...")
        _install_xcode_headless()

    # Block until installed (poll every 5s)
    import time

    logger.info("Waiting for Xcode CLI tools installation to complete...")
    for _ in range(120):  # wait up to 10 minutes
        if is_xcode_installed():
            logger.info("✓ Xcode Command Line Tools installed successfully.")
            return
        time.sleep(5)

    raise PlatformError(
        "Xcode CLI tools installation timed out. "
        "Please install manually: xcode-select --install"
    )


def _install_xcode_headless():
    """Headless install using softwareupdate — works over SSH with no GUI."""
    # Touch a marker file that tells softwareupdate to show CLI tools
    marker = Path("/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress")
    marker.touch()
    try:
        list_result = _run(["softwareupdate", "-l"])
        # Find the CLI tools package name
        package = None
        for line in list_result.stdout.splitlines():
            if "Command Line Tools" in line:
                package = line.strip().lstrip("*").strip()
                break

        if not package:
            raise PlatformError(
                "Could not find Xcode CLI tools in softwareupdate. "
                "Please install manually: xcode-select --install"
            )

        logger.info("Installing: %s", package)
        run_command(["softwareupdate", "-i", package, "--verbose"], check=True, capture_output=False)
    finally:
        marker.unlink(missing_ok=True)


def ensure_xcode():
    """Check for Xcode CLI tools and install if missing."""
    if is_xcode_installed():
        logger.info(
            "Xcode CLI tools found at: %s", _run(['xcode-select', '-p']).stdout.strip()
        )
        return
    install_xcode_cli()


def validate_app_path_for_platform(app_path: str, platform_name: str):
    """Ensure the app file extension matches the target platform."""
    path = Path(app_path)
    suffix = path.suffix.lower()

    if platform_name == "android":
        if suffix != ".apk":
            raise PlatformError(
                f"Android requires an .apk file, got: '{path.name}'. "
                "Build your app with: flutter build apk"
            )
    elif platform_name == "ios":
        if suffix not in (".app", ".ipa"):
            raise PlatformError(
                f"iOS requires a .app or .ipa file, got: '{path.name}'.\n"
                "  - Simulator: use a .app bundle (flutter build ios --simulator)\n"
                "  - Real device: use a .ipa file (flutter build ipa)"
            )
    else:
        raise PlatformError(
            f"Unknown platform: '{platform_name}'. Must be 'android' or 'ios'."
        )


def run_ios_preflight(app_path: str):
    """Run all iOS-specific checks in order."""
    check_platform_for_ios()
    ensure_xcode()
    validate_app_path_for_platform(app_path, "ios")
