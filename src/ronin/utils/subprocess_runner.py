"""One way to run a command, and one failure to catch when it does not work.

Callers care about two things: the output, or the reason there is none. A
missing binary and a non-zero exit are the same event to them, and both arrive
as SubprocessRunError. Nothing has to catch OSError alongside it.
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

DID_NOT_RUN = -1


class SubprocessRunError(subprocess.SubprocessError):
    """A command could not be run, or ran and failed."""

    def __init__(self, cmd: list[str] | str, returncode: int, stdout: str, stderr: str):
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
    cmd: list[str] | str,
    check: bool = False,
    timeout: Optional[float] = None,
    capture_output: bool = True,
    text: bool = True,
    shell: bool = False,
    env: Optional[dict[str, str]] = None,
    log_command: bool = True,
) -> subprocess.CompletedProcess:
    """Runs a command as a subprocess, with one place to log it and to fail.

    Args:
        cmd: The command to run, as a list of arguments or a shell string.
        check: If True, a non-zero exit code raises SubprocessRunError.
        timeout: Seconds to wait before giving up on the command.
        capture_output: If True, Ronin captures stdout and stderr. If False,
            the command writes to the terminal and the result carries no output.
        text: If True, Ronin decodes the captured output to strings.
        shell: If True, a shell runs the command.
        env: Environment variables for the command.
        log_command: If True, Ronin logs the command at info rather than debug.

    Returns:
        The CompletedProcess, whatever its exit code, unless check is True.

    Raises:
        SubprocessRunError: If Ronin could not start the command, or if check
            is True and the command exited non-zero.
        subprocess.TimeoutExpired: If the command ran longer than timeout.
    """
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    logger.log(
        logging.INFO if log_command else logging.DEBUG,
        "Running subprocess command: %s",
        cmd_str,
    )

    try:
        # `check` stays out of this call. subprocess.run would raise
        # CalledProcessError, which is not the failure every caller here
        # catches.
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            shell=shell,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("Subprocess command timed out: %s", cmd_str)
        raise
    except OSError as exc:
        raise SubprocessRunError(cmd, DID_NOT_RUN, "", str(exc)) from exc

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if stdout.strip():
        logger.debug("Subprocess stdout:\n%s", stdout.strip())
    if stderr.strip():
        logger.debug("Subprocess stderr:\n%s", stderr.strip())

    if check and result.returncode != 0:
        raise SubprocessRunError(cmd, result.returncode, stdout, stderr)
    return result
