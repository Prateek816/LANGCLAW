from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any, Optional

from langsmith import traceable

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
    import config as _cfg
    return str(_cfg.files_dir())

@traceable(run_type="tool", name="Shell Command")
def run_command(command: str) -> str:
    """Execute a shell command and return combined stdout/stderr.

    The command inherits the project's virtual environment so that
    ``python``, ``pip``, and any installed CLI tools resolve correctly.
    The working directory is set to ``~/.langclaw/context/files/`` so
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


def send_file(path: str, caption: str = "", sender: callable | None = None) -> str:
    """Send a file to the user via the active channel (Telegram/Discord/WhatsApp/Web).

    Args:
        path: Path to the file to send.
        caption: Optional caption for the file.
        sender: Channel-specific file sender callback. If None, falls back to
                the global _file_sender (deprecated path).
    """
    resolved = os.path.realpath(os.path.abspath(path))
    if not os.path.isfile(resolved):
        return f"Error: file not found: {path}"

    size = os.path.getsize(resolved)
    if size > _MAX_SEND_FILE_BYTES:
        size_mb = size / (1024 * 1024)
        return f"Error: file too large ({size_mb:.1f} MB). Maximum allowed is 100 MB."

    # Prefer session-scoped sender; fall back to global for backward compat
    active_sender = sender or _file_sender
    if active_sender is None:
        return (
            f"File ready at: {resolved} ({size / 1024:.1f} KB). "
            "No active channel to send through — user can download it directly."
        )

    try:
        active_sender(resolved, caption)
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

# ── Sandbox (path restriction) ───────────────────────────────────────────────

_sandbox_roots: list[str] = []


def set_sandbox(roots: list[str]) -> None:
    """Configure the allowed root directories for file-write operations.

    Called by Agent.__init__ to restrict write_file / create_skill to the
    project's working tree.  An empty list disables sandboxing (not recommended).
    """
    _sandbox_roots.clear()
    for r in roots:
        _sandbox_roots.append(os.path.realpath(r))


def _resolve_in_sandbox(path: str) -> str:
    """Resolve *path* to an absolute real path and verify it lives inside the sandbox.

    Returns the resolved path on success.
    Raises ``PermissionError`` if the path escapes every sandbox root.
    """
    resolved = os.path.realpath(os.path.abspath(path))

    if not _sandbox_roots:
        return resolved

    for root in _sandbox_roots:
        if resolved == root or resolved.startswith(root + os.sep):
            return resolved

    raise PermissionError(
        f"Path '{path}' (resolved to '{resolved}') is outside the allowed directories: "
        + ", ".join(_sandbox_roots)
    )



_tavily_client = None
_tavily_api_key = None


def _get_tavily_client():
    """Return a cached TavilyClient, rebuilding only when the API key changes."""
    global _tavily_client, _tavily_api_key
    #change here to get the api key from config
    _tavily_api_key = os.getenv("TAVILY_API_KEY") or _tavily_api_key
    api_key = os.getenv("TAVILY_API_KEY") or _tavily_api_key
    if not api_key:
        return None
    if _tavily_client is None or _tavily_api_key != api_key:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key)
        _tavily_api_key = api_key
    return _tavily_client


@traceable(run_type="tool", name="Web Search")
def web_search(
    query: str,
    *,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 3,
    time_range: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    """Search the web using the Tavily API and return formatted results."""
    try:
        from tavily import TavilyClient  # noqa: F401
    except ImportError:
        return (
            "Error: tavily-python is not installed. "
            "Install it with: pip install tavily-python"
        )

    client = _get_tavily_client()
    if client is None:
        return "Error: Tavily API key not configured (set TAVILY_API_KEY or tavily.apiKey in langclaw.json)"

    try:
        kwargs: dict = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "include_answer": True,
        }
        if time_range:
            kwargs["time_range"] = time_range
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains

        response = client.search(**kwargs)
    except Exception as exc:
        logger.warning("[web_search] Tavily API error: %s", exc)
        return f"Web search error: {exc}"

    parts: list[str] = []

    answer = response.get("answer")
    if answer:
        parts.append(f"**Summary:** {answer}\n")

    results = response.get("results", [])
    if results:
        parts.append("**Sources:**")
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"\n{i}. [{title}]({url})")
            if content:
                parts.append(f"   {content}")

    if not parts:
        return "No results found."

    return "\n".join(parts)


AVAILABLE_TOOLS["web_search"] = web_search




# ── Meta-skill: create_skill ("God Mode") ────────────────────────────────────

@traceable(run_type="tool", name="Create Skill")
def create_skill(
    name: str,
    description: str,
    instructions: str,
    category: str = "",
    resources: dict[str, str] | None = None,
    dependencies: list[str] | None = None,
) -> str:
    """Create a new skill on disk and install its dependencies.

    This is the "god mode" tool — the agent uses it to extend its own
    capabilities at runtime.  After creation, the caller must invalidate
    the SkillRegistry cache so the new skill appears in the catalog.

    All paths are validated against the sandbox.  Resource filenames are
    sanitized to prevent directory traversal.
    """
    import config as _cfg
    skills_dir = os.path.join(str(_cfg.LANGCLAW_HOME), "context", "skills")
    _resolve_in_sandbox(skills_dir)
    os.makedirs(skills_dir, exist_ok=True)

    # Build target directory (sanitize name and category)
    safe_name = _sanitize_filename(name.replace(" ", "_").lower())
    if category:
        safe_category = _sanitize_filename(category.replace(" ", "_").lower())
        skill_dir = os.path.join(skills_dir, safe_category, safe_name)
        cat_dir = os.path.join(skills_dir, safe_category)
        cat_md = os.path.join(cat_dir, "CATEGORY.md")
        if not os.path.isfile(cat_md):
            os.makedirs(cat_dir, exist_ok=True)
            with open(cat_md, "w", encoding="utf-8") as f:
                f.write(f"---\nname: {safe_category}\ndescription: Auto-created category for {category} skills.\n---\n")
    else:
        skill_dir = os.path.join(skills_dir, safe_name)

    _resolve_in_sandbox(skill_dir)
    os.makedirs(skill_dir, exist_ok=True)

    # Write SKILL.md
    skill_md_content = (
        f"---\nname: {safe_name}\n"
        f"description: >\n  {description}\n"
        f"---\n\n{instructions}\n"
    )
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    with open(skill_md_path, "w", encoding="utf-8") as f:
        f.write(skill_md_content)

    # Write resource files (filenames are sanitized to prevent traversal)
    written_files = ["SKILL.md"]
    if resources:
        for filename, content in resources.items():
            safe_fn = _sanitize_filename(filename)
            fpath = os.path.join(skill_dir, safe_fn)
            _resolve_in_sandbox(fpath)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            if safe_fn.endswith((".sh", ".py")):
                os.chmod(fpath, 0o755)
            written_files.append(safe_fn)

    # Install dependencies (into the project venv)
    dep_results: list[str] = []
    if dependencies:
        pip_python = _venv_python()
        for dep in dependencies:
            try:
                proc = subprocess.run(
                    [pip_python, "-m", "pip", "install", dep],
                    capture_output=True, text=True, timeout=120,
                    env=_venv_env(),
                )
                if proc.returncode == 0:
                    dep_results.append(f"  ✓ {dep}")
                else:
                    dep_results.append(f"  ✗ {dep}: {proc.stderr.strip()}")
            except Exception as exc:
                dep_results.append(f"  ✗ {dep}: {exc}")

    # Build result summary
    parts = [
        f"Skill '{safe_name}' created at {skill_dir}/",
        f"Files: {', '.join(written_files)}",
    ]
    if dep_results:
        parts.append("Dependencies:\n" + "\n".join(dep_results))
    parts.append("Skill is now available via use_skill().")

    return "\n".join(parts)


AVAILABLE_TOOLS["create_skill"] = create_skill


