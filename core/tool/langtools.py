"""
LangChain-compatible wrappers for all built-in tools.

Usage
-----
    from langchain_tools import get_all_langchain_tools

    tools = get_all_langchain_tools()          # all tools
    agent = create_react_agent(llm, tools, prompt)

You can also import individual tools or groups:
    from langchain_tools import (
        primitive_tools,
        memory_tools,
        web_search_tool,
        skill_tools,
        cron_tools,
        meta_skill_tools,
    )
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Import all implementation functions from your tools module ────────────────
# Adjust this import path to wherever your tools.py lives.
from core.tool.tools import (
    run_command,
    read_file,
    write_file,
    list_files,
    send_file,
    web_search,
    create_skill,
)

# Memory functions may live in a separate module — adjust as needed.
# from memory import remember, recall, memory_get, memory_list_files, forget, update_index

# ── Primitive Tools ───────────────────────────────────────────────────────────

@tool
def lc_run_command(command: str) -> str:
    """Execute a shell command. Use to run scripts, install packages, or perform system operations."""
    return run_command(command)


@tool
def lc_read_file(path: str) -> str:
    """Read the contents of a file. Use to inspect code, logs, or data."""
    return read_file(path)


@tool
def lc_write_file(path: str, content: str) -> str:
    """Write content to a file (must be within the project directory). Creates parent directories automatically."""
    return write_file(path, content)


@tool
def lc_list_files(path: str = ".") -> str:
    """List files in a directory. Use to discover available scripts or files."""
    return list_files(path)


@tool
def lc_send_file(path: str, caption: str = "") -> str:
    """Send a file to the user via the active channel. Max 100 MB. Use when the user asks to download or receive a file."""
    return send_file(path, caption)


# ── Web Search Tool ───────────────────────────────────────────────────────────

@tool
def lc_web_search(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 3,
    time_range: Optional[str] = None,
) -> str:
    """Search the web for real-time information using the Tavily API.
    Use this when you need up-to-date information, current events,
    facts you're unsure about, or anything that benefits from live web data.

    Args:
        query: The search query. Be specific for better results.
        search_depth: 'basic' (fast) or 'advanced' (more thorough).
        topic: 'general', 'news', or 'finance'.
        max_results: Number of results to return (1-10). Use 2-3 for most queries.
        time_range: Filter by recency — 'day', 'week', 'month', or 'year'. Leave None for no filter.
    """
    return web_search(
        query,
        search_depth=search_depth,
        topic=topic,
        max_results=max_results,
        time_range=time_range,
    )


# ── Skill Tools ───────────────────────────────────────────────────────────────
# These call back into your agent's skill system.
# Replace the stubs below with real imports once your skill runner is importable.

@tool
def lc_use_skill(skill_name: str) -> str:
    """Activate a skill by name. This loads the skill's detailed instructions
    and workflow into context. Only call this when you've identified the right
    skill from the catalog in the system prompt."""
    # Replace with: from skills import use_skill; return use_skill(skill_name)
    raise NotImplementedError(
        f"lc_use_skill: wire up your skill runner for '{skill_name}'"
    )


@tool
def lc_list_skill_resources(skill_name: str) -> str:
    """List resource files bundled with a skill (scripts, schemas, reference docs).
    Use after activating a skill to discover what files are available."""
    # Replace with: from skills import list_skill_resources; return list_skill_resources(skill_name)
    raise NotImplementedError(
        f"lc_list_skill_resources: wire up your skill runner for '{skill_name}'"
    )


# ── Memory Tools ──────────────────────────────────────────────────────────────
# Stubs — replace `raise NotImplementedError` bodies with real calls once
# your memory module is importable, e.g.:
#   from memory import remember as _remember
#   return _remember(key, content)

@tool
def lc_remember(key: str, content: str) -> str:
    """Store a piece of information in long-term memory.

    Args:
        key: Topic or category to store under.
        content: The information to remember.
    """
    # from memory import remember as _remember
    # return _remember(key, content)
    raise NotImplementedError("lc_remember: wire up your memory module")


@tool
def lc_recall(query: str) -> str:
    """Search long-term memory using semantic + keyword retrieval.
    Pass a descriptive query to get the most relevant memories.
    Use query='*' to retrieve ALL memories.

    Args:
        query: Topic or question to search memory for. Use '*' for all memories.
    """
    # from memory import recall as _recall
    # return _recall(query)
    raise NotImplementedError("lc_recall: wire up your memory module")


@tool
def lc_memory_get(path: str) -> str:
    """Read a specific memory file by path.
    Use 'MEMORY.md' for long-term memory or 'YYYY-MM-DD.md' for daily logs.

    Args:
        path: Filename relative to memory dir (e.g. 'MEMORY.md', '2026-03-03.md').
    """
    # from memory import memory_get as _memory_get
    # return _memory_get(path)
    raise NotImplementedError("lc_memory_get: wire up your memory module")


@tool
def lc_memory_list_files() -> str:
    """List all memory files (MEMORY.md + daily logs)."""
    # from memory import memory_list_files as _mlf
    # return _mlf()
    raise NotImplementedError("lc_memory_list_files: wire up your memory module")


@tool
def lc_forget(key: str) -> str:
    """Delete a memory entry by key from long-term memory.

    Args:
        key: The key to remove from memory.
    """
    # from memory import forget as _forget
    # return _forget(key)
    raise NotImplementedError("lc_forget: wire up your memory module")


@tool
def lc_update_index(content: str) -> str:
    """Update the INDEX.md system info file.
    Use this to store curated environment info, API notes, and configuration
    that should persist across sessions.

    Args:
        content: Full Markdown content for INDEX.md.
    """
    # from memory import update_index as _update_index
    # return _update_index(content)
    raise NotImplementedError("lc_update_index: wire up your memory module")


# ── Meta-Skill Tool ───────────────────────────────────────────────────────────

@tool
def lc_create_skill(
    name: str,
    description: str,
    instructions: str,
    category: str = "",
    resources: Optional[Dict[str, str]] = None,
    dependencies: Optional[List[str]] = None,
) -> str:
    """Create a brand-new skill on the fly when no existing skill can handle the request.
    Writes a SKILL.md and optional resource scripts to the skills directory,
    installs pip dependencies, and makes the skill immediately available.
    Use this when you need a capability that doesn't exist yet.

    Args:
        name: Skill name (lowercase, underscores). E.g. 'weather_forecast'.
        description: One-line description of what the skill does and when to use it.
        instructions: Full Markdown instructions for the skill body.
        category: Optional category folder (e.g. 'data', 'dev', 'web'). Empty for flat layout.
        resources: Map of filename -> file content for bundled scripts.
        dependencies: List of pip packages to install.
    """
    return create_skill(
        name=name,
        description=description,
        instructions=instructions,
        category=category,
        resources=resources,
        dependencies=dependencies,
    )


# ── Cron Tools ────────────────────────────────────────────────────────────────
# Stubs — wire up your CronScheduler once it's importable.

@tool
def lc_cron_add(
    job_id: str,
    cron: str,
    prompt: str,
    deliver_to_chat_id: Optional[int] = None,
) -> str:
    """Schedule a recurring LLM task using standard 5-field cron syntax.
    Example: '0 9 * * *' = 9 am daily.

    Args:
        job_id: Unique job identifier (no spaces).
        cron: 5-field cron expression, e.g. '0 9 * * *'.
        prompt: The prompt the agent will run on each trigger.
        deliver_to_chat_id: Optional Telegram chat_id to deliver the result to.
    """
    # from cron import scheduler; return scheduler.add(job_id, cron, prompt, deliver_to_chat_id)
    raise NotImplementedError("lc_cron_add: wire up your CronScheduler")


@tool
def lc_cron_remove(job_id: str) -> str:
    """Remove a previously scheduled cron job by its ID.

    Args:
        job_id: The job ID to remove.
    """
    # from cron import scheduler; return scheduler.remove(job_id)
    raise NotImplementedError("lc_cron_remove: wire up your CronScheduler")


@tool
def lc_cron_list() -> str:
    """List all currently scheduled cron jobs (both static and dynamic)."""
    # from cron import scheduler; return scheduler.list_jobs()
    raise NotImplementedError("lc_cron_list: wire up your CronScheduler")


# ── Grouped exports ───────────────────────────────────────────────────────────

primitive_tools = [
    lc_run_command,
    lc_read_file,
    lc_write_file,
    lc_list_files,
    lc_send_file,
]

web_search_tool = [lc_web_search]

skill_tools = [
    lc_use_skill,
    lc_list_skill_resources,
]

memory_tools = [
    lc_remember,
    lc_recall,
    lc_memory_get,
    lc_memory_list_files,
    lc_forget,
    lc_update_index,
]

meta_skill_tools = [lc_create_skill]

cron_tools = [
    lc_cron_add,
    lc_cron_remove,
    lc_cron_list,
]


def get_all_langchain_tools():
    """Return every tool as a flat list, ready to pass to a LangChain agent."""
    return (
        primitive_tools
        + web_search_tool
        + skill_tools
        + memory_tools
        + meta_skill_tools
        + cron_tools
    )