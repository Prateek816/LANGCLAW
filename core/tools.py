from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

_venv_dir : str|None = None

def _detect_venv()->str|None:
    """Find the project's virtual environment directory.

    Priority:
      1. Already running inside a venv (sys.prefix != sys.base_prefix)
      2. .venv/ in CWD
      3. venv/ in CWD
    """
    if sys.prefix != sys.base_prefix:
        return sys.prefix

    for name in (".venv", "venv"):
        candidate = os.path.join(os.getcwd(), name)
        python = os.path.join(candidate, "bin", "python")
        if os.path.isfile(python):
            return candidate

    return None

def _venv_python() -> str:
    """Return the Python executable inside the detected venv, or sys.executable."""
    venv = _venv_dir or _detect_venv()
    if venv:
        candidate = os.path.join(venv, "bin", "python")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable

def _venv_env() -> dict[str, str]:
    """Build an env dict that activates the project venv for subprocesses."""
    env = os.environ.copy()
    venv = _venv_dir or _detect_venv()
    if venv:
        venv_bin = os.path.join(venv, "bin")
        env["VIRTUAL_ENV"] = venv
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env.pop("PYTHONHOME", None)
    else:
        python_dir = os.path.dirname(sys.executable)
        env["PATH"] = f"{python_dir}{os.pathsep}{env.get('PATH', '')}"
    return env

def configure_venv(venv_dir: str | None = None) -> str | None:
    """Explicitly set or auto-detect the venv. Called by Agent.__init__."""
    global _venv_dir
    if venv_dir:
        _venv_dir = os.path.realpath(venv_dir)
    else:
        _venv_dir = _detect_venv()
    if _venv_dir:
        logger.info("[tools] Using venv: %s", _venv_dir)
    return _venv_dir

def _sanitize_filename(name: str) -> str:
    """Strip path separators and '..' segments from a filename."""
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    if not name:
        raise ValueError("Empty or invalid filename after sanitization.")
    return name

# ── Primitive tool implementations ────────────────────────────────────────────

def _files_dir() -> str:
    """Return the shared files directory, creating it if needed."""
    from .. import config as _cfg
    return str(_cfg.files_dir())

def run_command(command: str) -> str:
    """Execute a shell command and return combined stdout/stderr.

    The command inherits the project's virtual environment so that
    ``python``, ``pip``, and any installed CLI tools resolve correctly.
    The working directory is set to ``~/.pythonclaw/context/files/`` so
    that any files created or downloaded by the command land there.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, env=_venv_env(), cwd=_files_dir(),
        )
        return result.stdout if result.returncode == 0 else f"Error (exit {result.returncode}):\n{result.stderr}"
    except Exception as exc:
        return f"Execution error: {exc}"
    
def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    try:
        if not os.path.exists(path):
            return f"Error: '{path}' not found."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Read error: {exc}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Writes are restricted to sandbox directories (configured via set_sandbox).
    """
    try:
        resolved = _resolve_in_sandbox(path)
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except PermissionError as exc:
        return f"Blocked: {exc}"
    except Exception as exc:
        return f"Write error: {exc}"


def list_files(path: str = ".") -> str:
    """List files in a directory, one per line."""
    try:
        return "\n".join(sorted(os.listdir(path)))
    except Exception as exc:
        return f"List error: {exc}"


_MAX_SEND_FILE_BYTES = 100 * 1024 * 1024  # 100 MB

# Channel-provided callback: send_file_fn(path, caption) → None
_file_sender: callable | None = None


def set_file_sender(fn: callable | None) -> None:
    """Register a callback for sending files to the current channel."""
    global _file_sender
    _file_sender = fn


def send_file(path: str, caption: str = "") -> str:
    """Send a file to the user via the active channel (Telegram/Discord/WhatsApp/Web)."""
    resolved = os.path.realpath(os.path.abspath(path))
    if not os.path.isfile(resolved):
        return f"Error: file not found: {path}"

    size = os.path.getsize(resolved)
    if size > _MAX_SEND_FILE_BYTES:
        size_mb = size / (1024 * 1024)
        return f"Error: file too large ({size_mb:.1f} MB). Maximum allowed is 100 MB."

    if _file_sender is None:
        return (
            f"File ready at: {resolved} ({size / 1024:.1f} KB). "
            "No active channel to send through — user can download it directly."
        )

    try:
        _file_sender(resolved, caption)
        name = os.path.basename(resolved)
        return f"File '{name}' ({size / 1024:.1f} KB) sent successfully."
    except Exception as exc:
        return f"Error sending file: {exc}"


AVAILABLE_TOOLS: dict[str, callable] = {
    "run_command": run_command,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "send_file": send_file,
}