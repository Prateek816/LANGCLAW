from __future__ import annotations

"""LangChain-wrapped versions of all PythonClaw tools.

Each tool is exposed as a @tool-decorated function (or a StructuredTool where
multiple inputs are required) and collected in ALL_TOOLS / by category.
"""

import logging
from typing import Optional

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

# Import the original implementations
from .tools import (
    run_command,
    read_file,
    write_file,
    list_files,
    send_file,
    web_search,
    create_skill,
)

logger = logging.getLogger(__name__)


# ── Primitive Tools ───────────────────────────────────────────────────────────

@tool
def lc_run_command(command: str) -> str:
    """Execute a shell command and return combined stdout/stderr.

    The command inherits the project's virtual environment so that
    ``python``, ``pip``, and any installed CLI tools resolve correctly.
    The working directory is set to ``~/.pythonclaw/context/files/`` so
    that any files created or downloaded by the command land there.
    """
    return run_command(command)


@tool
def lc_read_file(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    return read_file(path)


class WriteFileInput(BaseModel):
    path: str = Field(description="Path to the file to write (must be within project root).")
    content: str = Field(description="The content to write.")


lc_write_file = StructuredTool.from_function(
    func=write_file,
    name="write_file",
    description=(
        "Write content to a file, creating parent directories as needed. "
        "Writes are restricted to sandbox directories."
    ),
    args_schema=WriteFileInput,
)


@tool
def lc_list_files(path: str = ".") -> str:
    """List files in a directory, one per line. Defaults to the current directory."""
    return list_files(path)


class SendFileInput(BaseModel):
    path: str = Field(description="Absolute or relative path to the file to send.")
    caption: str = Field(default="", description="Optional caption or description for the file.")


lc_send_file = StructuredTool.from_function(
    func=send_file,
    name="send_file",
    description=(
        "Send a file to the user via the active channel (Telegram/Discord/WhatsApp/Web). "
        "Max 100 MB. Use when the user asks to download or receive a file."
    ),
    args_schema=SendFileInput,
)


# ── Web Search Tool ───────────────────────────────────────────────────────────

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query. Be specific for better results.")
    search_depth: str = Field(
        default="basic",
        description="Search depth: 'basic' (fast) or 'advanced' (more thorough).",
    )
    topic: str = Field(
        default="general",
        description="Search category: 'general', 'news', or 'finance'.",
    )
    max_results: int = Field(
        default=3,
        description="Number of results to return (1-10). Use 2-3 for most queries.",
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Filter results by recency: 'day', 'week', 'month', or 'year'. Omit for no filter.",
    )
    include_domains: Optional[list[str]] = Field(
        default=None,
        description="Restrict results to these domains.",
    )
    exclude_domains: Optional[list[str]] = Field(
        default=None,
        description="Exclude results from these domains.",
    )


lc_web_search = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description=(
        "Search the web for real-time information using the Tavily API. "
        "Use this when you need up-to-date information, current events, "
        "facts you're unsure about, or anything that benefits from live web data."
    ),
    args_schema=WebSearchInput,
)


# ── Meta-Skill Tool ───────────────────────────────────────────────────────────

class CreateSkillInput(BaseModel):
    name: str = Field(
        description="Skill name (lowercase, underscores). E.g. 'weather_forecast'."
    )
    description: str = Field(
        description="One-line description of what the skill does and when to use it."
    )
    instructions: str = Field(
        description=(
            "Full Markdown instructions for the skill body (content after the YAML frontmatter). "
            "Include ## Instructions, usage examples, and ## Resources sections."
        )
    )
    category: str = Field(
        default="",
        description="Optional category folder (e.g. 'data', 'dev', 'web'). Empty for flat layout.",
    )
    resources: Optional[dict[str, str]] = Field(
        default=None,
        description=(
            "Map of filename → file content for bundled scripts. "
            "E.g. {'fetch.py': 'import requests\\n...', 'config.yaml': '...'}."
        ),
    )
    dependencies: Optional[list[str]] = Field(
        default=None,
        description="List of pip packages to install. E.g. ['requests', 'beautifulsoup4'].",
    )


lc_create_skill = StructuredTool.from_function(
    func=create_skill,
    name="create_skill",
    description=(
        "Create a brand-new skill on the fly when no existing skill can handle the user's request. "
        "Writes a SKILL.md and optional resource scripts to the skills directory, "
        "installs pip dependencies, and makes the skill immediately available. "
        "Use this when you need a capability that doesn't exist yet."
    ),
    args_schema=CreateSkillInput,
)


# ── Memory Tools ──────────────────────────────────────────────────────────────
# Memory operations delegate to the MemoryManager; the LC wrappers here
# call through the AVAILABLE_TOOLS registry so the real implementations
# are resolved at runtime (keeping this module decoupled from memory internals).

def _memory_dispatch(tool_name: str, **kwargs) -> str:
    """Resolve and call a memory tool from AVAILABLE_TOOLS at runtime."""
    from .tools import AVAILABLE_TOOLS  # late import to avoid circular deps
    fn = AVAILABLE_TOOLS.get(tool_name)
    if fn is None:
        return f"Error: memory tool '{tool_name}' is not registered."
    try:
        return fn(**kwargs)
    except Exception as exc:
        return f"Memory error: {exc}"


class RememberInput(BaseModel):
    key: str = Field(description="Topic or category to store the information under.")
    content: str = Field(description="The information to remember.")


lc_remember = StructuredTool.from_function(
    func=lambda key, content: _memory_dispatch("remember", key=key, content=content),
    name="remember",
    description="Store a piece of information in long-term memory.",
    args_schema=RememberInput,
)


@tool
def lc_recall(query: str) -> str:
    """Search long-term memory using semantic + keyword retrieval.

    Pass a descriptive query to get the most relevant memories.
    Use query='*' to retrieve ALL memories.
    """
    return _memory_dispatch("recall", query=query)


@tool
def lc_memory_get(path: str) -> str:
    """Read a specific memory file by path.

    Use 'MEMORY.md' for long-term memory or 'YYYY-MM-DD.md' for daily logs.
    """
    return _memory_dispatch("memory_get", path=path)


@tool
def lc_memory_list_files() -> str:
    """List all memory files (MEMORY.md + daily logs)."""
    return _memory_dispatch("memory_list_files")


@tool
def lc_forget(key: str) -> str:
    """Delete a memory entry by key from long-term memory."""
    return _memory_dispatch("forget", key=key)


@tool
def lc_update_index(content: str) -> str:
    """Update the INDEX.md system info file.

    Use this to store curated environment info, API notes, and configuration
    that should persist across sessions.
    """
    return _memory_dispatch("update_index", content=content)


# ── Skill Tools ───────────────────────────────────────────────────────────────

def _skill_dispatch(tool_name: str, **kwargs) -> str:
    from .tools import AVAILABLE_TOOLS
    fn = AVAILABLE_TOOLS.get(tool_name)
    if fn is None:
        return f"Error: skill tool '{tool_name}' is not registered."
    try:
        return fn(**kwargs)
    except Exception as exc:
        return f"Skill error: {exc}"


@tool
def lc_use_skill(skill_name: str) -> str:
    """Activate a skill by name.

    This loads the skill's detailed instructions and workflow into context.
    Only call this when you've identified the right skill from the catalog
    in the system prompt.
    """
    return _skill_dispatch("use_skill", skill_name=skill_name)


@tool
def lc_list_skill_resources(skill_name: str) -> str:
    """List resource files bundled with a skill (scripts, schemas, reference docs).

    Use after activating a skill to discover what files are available.
    """
    return _skill_dispatch("list_skill_resources", skill_name=skill_name)


# ── Cron Tools ────────────────────────────────────────────────────────────────

def _cron_dispatch(tool_name: str, **kwargs) -> str:
    from .tools import AVAILABLE_TOOLS
    fn = AVAILABLE_TOOLS.get(tool_name)
    if fn is None:
        return f"Error: cron tool '{tool_name}' is not registered."
    try:
        return fn(**kwargs)
    except Exception as exc:
        return f"Cron error: {exc}"


class CronAddInput(BaseModel):
    job_id: str = Field(description="Unique job identifier (no spaces).")
    cron: str = Field(description="5-field cron expression, e.g. '0 9 * * *'.")
    prompt: str = Field(description="The prompt the agent will run on each trigger.")
    deliver_to_chat_id: Optional[int] = Field(
        default=None,
        description="Optional Telegram chat_id to deliver the result to.",
    )


lc_cron_add = StructuredTool.from_function(
    func=lambda **kw: _cron_dispatch("cron_add", **kw),
    name="cron_add",
    description=(
        "Schedule a recurring LLM task. "
        "Use standard 5-field cron syntax: 'min hour day month weekday'. "
        "Example: '0 9 * * *' = 9 am daily."
    ),
    args_schema=CronAddInput,
)


@tool
def lc_cron_remove(job_id: str) -> str:
    """Remove a previously scheduled cron job by its ID."""
    return _cron_dispatch("cron_remove", job_id=job_id)


@tool
def lc_cron_list() -> str:
    """List all currently scheduled cron jobs (both static and dynamic)."""
    return _cron_dispatch("cron_list")


# ── Knowledge Base Tool ───────────────────────────────────────────────────────

@tool
def lc_consult_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information using hybrid retrieval."""
    from .tools import AVAILABLE_TOOLS
    fn = AVAILABLE_TOOLS.get("consult_knowledge_base")
    if fn is None:
        return "Error: knowledge base tool is not registered."
    return fn(query=query)


# ── Grouped exports ───────────────────────────────────────────────────────────

PRIMITIVE_LC_TOOLS = [
    lc_run_command,
    lc_read_file,
    lc_write_file,
    lc_list_files,
    lc_send_file,
]

MEMORY_LC_TOOLS = [
    lc_remember,
    lc_recall,
    lc_memory_get,
    lc_memory_list_files,
    lc_forget,
    lc_update_index,
]

SKILL_LC_TOOLS = [
    lc_use_skill,
    lc_list_skill_resources,
]

META_SKILL_LC_TOOLS = [
    lc_create_skill,
]

WEB_SEARCH_LC_TOOLS = [
    lc_web_search,
]

CRON_LC_TOOLS = [
    lc_cron_add,
    lc_cron_remove,
    lc_cron_list,
]

KNOWLEDGE_LC_TOOLS = [
    lc_consult_knowledge_base,
]

# Convenience: every tool in one flat list
ALL_LC_TOOLS = (
    PRIMITIVE_LC_TOOLS
    + MEMORY_LC_TOOLS
    + SKILL_LC_TOOLS
    + META_SKILL_LC_TOOLS
    + WEB_SEARCH_LC_TOOLS
    + CRON_LC_TOOLS
    + KNOWLEDGE_LC_TOOLS
)