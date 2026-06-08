"""Safe subprocess execution environment for SymPy-based equation verification."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from loguru import logger


class SandboxError(Exception):
    """Raised when sandbox execution fails."""


class SandboxTimeoutError(SandboxError):
    """Raised when sandbox execution times out."""


def run_sympy_sandbox(
    code: str,
    python_executable: str = "python3",
    timeout_seconds: int = 10,
    max_output_bytes: int = 65536,
) -> tuple[str, str, int]:
    """Execute SymPy verification code in a sandboxed subprocess.

    The code is written to a temporary file and executed in an isolated
    subprocess. Only stdout and stderr are captured.

    Args:
        code: Python/SymPy source code to execute.
        python_executable: Path or name of the Python interpreter.
        timeout_seconds: Maximum execution time before killing the process.
        max_output_bytes: Maximum bytes to read from stdout/stderr.

    Returns:
        Tuple of (stdout, stderr, returncode).

    Raises:
        SandboxTimeoutError: If execution exceeds timeout_seconds.
        SandboxError: If the subprocess fails to start.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="sympy_check_",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(code)
        tmp.flush()

    try:
        logger.debug(f"Executing sandbox: {tmp_path}")

        result = subprocess.run(
            [python_executable, str(tmp_path)],
            capture_output=True,
            timeout=timeout_seconds,
            text=False,  # We'll decode manually
            cwd=tmp_path.parent,
        )

        stdout = result.stdout.decode("utf-8", errors="replace")[:max_output_bytes]
        stderr = result.stderr.decode("utf-8", errors="replace")[:max_output_bytes]
        returncode = result.returncode

        logger.debug(
            f"Sandbox exited with code {returncode} "
            f"(stdout={len(stdout)}B, stderr={len(stderr)}B)"
        )

        return stdout, stderr, returncode

    except subprocess.TimeoutExpired as exc:
        logger.warning(f"Sandbox timed out after {timeout_seconds}s: {tmp_path}")
        raise SandboxTimeoutError(
            f"Execution timed out after {timeout_seconds} seconds"
        ) from exc

    except Exception as exc:
        logger.error(f"Sandbox execution failed: {exc}")
        raise SandboxError(f"Sandbox execution error: {exc}") from exc

    finally:
        # Clean up temporary file
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
