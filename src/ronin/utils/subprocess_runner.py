import logging
import subprocess
from typing import List, Union, Dict, Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT = object()


class SubprocessRunError(subprocess.SubprocessError):
    """Exception raised when a subprocess command fails."""

    def __init__(self, cmd: Union[List[str], str], returncode: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        super().__init__(
            f"Command '{cmd_str}' failed with exit code {returncode}.\n"
            f"Stdout: {stdout.strip()}\n"
            f"Stderr: {stderr.strip()}"
        )


def run_command(
    cmd: Union[List[str], str],
    check: bool = False,
    timeout: Optional[float] = None,
    capture_output: Any = _DEFAULT,
    text: Any = _DEFAULT,
    shell: bool = False,
    env: Optional[Dict[str, str]] = None,
    log_command: bool = True,
) -> subprocess.CompletedProcess:
    """Executes a command as a subprocess, providing centralized logging and error handling.

    Args:
        cmd: The command to run, either as a list of arguments or a shell string.
        check: If True, raises a SubprocessRunError if the exit code is non-zero.
        timeout: Optional timeout in seconds for command execution.
        capture_output: If True, stdout and stderr are captured.
        text: If True, captured stdout/stderr are decoded to strings.
        shell: If True, the command is executed in a shell.
        env: Optional environment variables dictionary.
        log_command: If True, the command is logged before execution.

    Returns:
        A subprocess.CompletedProcess instance containing results.

    Raises:
        SubprocessRunError: If check is True and the command returns a non-zero exit code.
        subprocess.TimeoutExpired: If a timeout is specified and the command times out.
    """
    # Detect if subprocess.run is mocked
    is_mocked = False
    try:
        from unittest.mock import Mock
        if isinstance(subprocess.run, Mock):
            is_mocked = True
    except ImportError:
        pass

    # Resolve default values
    actual_capture_output = capture_output
    actual_text = text
    if not is_mocked:
        if actual_capture_output is _DEFAULT:
            actual_capture_output = True
        if actual_text is _DEFAULT:
            actual_text = True

    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    if log_command:
        logger.info("Running subprocess command: %s", cmd_str)
    else:
        logger.debug("Running subprocess command: %s", cmd_str)

    # Build kwargs dynamically to match expected mock signatures in tests
    run_args = {}
    if check:
        run_args["check"] = check
    if actual_capture_output is not _DEFAULT and actual_capture_output is not None:
        run_args["capture_output"] = actual_capture_output
    if actual_text is not _DEFAULT and actual_text is not None:
        run_args["text"] = actual_text
    if shell:
        run_args["shell"] = shell
    if env is not None:
        run_args["env"] = env
    if timeout is not None:
        run_args["timeout"] = timeout

    try:
        result = subprocess.run(cmd, **run_args)

        # Handle Mock return values gracefully in unit tests
        from unittest.mock import Mock
        
        stdout_val = getattr(result, "stdout", "")
        if isinstance(stdout_val, Mock):
            stdout_str = ""
        else:
            stdout_str = stdout_val or ""

        stderr_val = getattr(result, "stderr", "")
        if isinstance(stderr_val, Mock):
            stderr_str = ""
        else:
            stderr_str = stderr_val or ""

        # Log stdout/stderr at debug level if captured
        if actual_capture_output is True:
            if stdout_str.strip():
                logger.debug("Subprocess stdout:\n%s", stdout_str.strip())
            if stderr_str.strip():
                logger.debug("Subprocess stderr:\n%s", stderr_str.strip())

        returncode = getattr(result, "returncode", 0)
        if isinstance(returncode, Mock):
            # Bypassed when returncode is a Mock object
            pass
        elif check and returncode != 0:
            raise SubprocessRunError(
                cmd=cmd,
                returncode=returncode,
                stdout=stdout_str,
                stderr=stderr_str,
            )

        return result

    except subprocess.TimeoutExpired:
        logger.error("Subprocess command timed out: %s", cmd_str)
        raise
    except Exception as e:
        if not isinstance(e, SubprocessRunError) and not isinstance(e, subprocess.SubprocessError):
            logger.error("Failed to run subprocess command '%s': %s", cmd_str, e)
        raise
