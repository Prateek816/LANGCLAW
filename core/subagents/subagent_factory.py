"""
Public API
----------
    invoke_subagent(config, prompt, context) -> str

The main agent calls this through a StructuredTool; it never needs to
know how the sub-agent is assembled internally.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.schema import SystemMessage
from langchain_core.tools import BaseTool

# ── Your tool imports ─────────────────────────────────────────────────────────
# Adjust paths to match your project layout.
from core.tool.tools import (
    run_command,
    read_file,
    write_file,
    list_files,
    send_file,
    web_search,
)
from langchain_core.tools import tool
from typing import Optional as Opt

# ── Model import (dummy — replace with your real LLM init) ───────────────────
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="...", base_url="...", api_key="...")
from langchain_core.language_models import BaseChatModel   # type: ignore

# ── Model import ─────────────────────────────────────────────────────────────
from core.subagents.model import SubAgentConfig, ToolAccessPolicy
from core.llm.config import Provider , LLMConfig
from core.llm.factory import get_llm as LLM
import config as _cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-wrapped primitive tools
# ─────────────────────────────────────────────────────────────────────────────

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
    """Send a file to the user via the active channel. Max 100 MB."""
    return send_file(path, caption)


@tool
def lc_web_search(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 3,
    time_range: Opt[str] = None,
) -> str:
    """Search the web for real-time information using the Tavily API.
    Use when you need up-to-date facts, current events, or live web data.

    Args:
        query: The search query. Be specific for better results.
        search_depth: 'basic' (fast) or 'advanced' (more thorough).
        topic: 'general', 'news', or 'finance'.
        max_results: Number of results to return (1–10).
        time_range: 'day', 'week', 'month', or 'year'. None = no filter.
    """
    return web_search(
        query,
        search_depth=search_depth,
        topic=topic,
        max_results=max_results,
        time_range=time_range,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — maps the string names used in markdown → actual tool objects
# Add any new tools here; the resolver picks them up automatically.
# ─────────────────────────────────────────────────────────────────────────────

_ALL_TOOLS: list[BaseTool] = [
    lc_run_command,
    lc_read_file,
    lc_write_file,
    lc_list_files,
    lc_send_file,
    lc_web_search,
]

_TOOL_REGISTRY: dict[str, BaseTool] = {t.name: t for t in _ALL_TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_tools(config: SubAgentConfig) -> list[BaseTool]:
    """
    Return the tool list the sub-agent should receive, based on its
    ToolAccessPolicy and the tools listed in the markdown file.

    Policy.ALL      → every tool in _ALL_TOOLS
    Policy.EXPLICIT → only the tools named in config.tools (unknown names warned)
    Policy.NONE     → empty list
    """
    if config.tool_access_policy == ToolAccessPolicy.NONE:
        logger.debug("[%s] tool_access_policy=NONE → no tools granted", config.name)
        return []

    if config.tool_access_policy == ToolAccessPolicy.ALL or not config.tools:
        logger.debug("[%s] tool_access_policy=ALL → granting all %d tool(s)", config.name, len(_ALL_TOOLS))
        return list(_ALL_TOOLS)

    # EXPLICIT — resolve by name
    resolved: list[BaseTool] = []
    for tool_name in config.tools:
        lc_name = tool_name if tool_name.startswith("lc_") else f"lc_{tool_name}"
        # Try both "web_search" and "lc_web_search" spellings
        found = _TOOL_REGISTRY.get(lc_name) or _TOOL_REGISTRY.get(tool_name)
        if found:
            resolved.append(found)
        else:
            logger.warning(
                "[%s] Unknown tool %r in markdown definition — skipping. "
                "Available tools: %s",
                config.name,
                tool_name,
                list(_TOOL_REGISTRY.keys()),
            )

    logger.debug("[%s] Resolved %d / %d tool(s)", config.name, len(resolved), len(config.tools))
    return resolved


def _build_system_prompt(config: SubAgentConfig, context: str) -> str:
    """
    Combine the agent's static system prompt from the markdown file with
    the runtime context injected by the orchestrator.

    The context block is appended only when non-empty so the agent's
    prompt stays clean for context-free invocations.
    """
    parts = [config.prompt.strip()]

    if context and context.strip():
        parts.append(
            "\n\n--- Orchestrator Context ---\n"
            "The following information was passed by the main orchestrator agent.\n"
            "Use it to inform your work but do not repeat it verbatim in your answer.\n\n"
            f"{context.strip()}\n"
            "--- End Orchestrator Context ---"
        )

    parts.append(
        "\n\nWhen you have completed the task, output a clear, self-contained "
        "summary that the orchestrator can directly use or relay to the user. "
        "Do NOT ask follow-up questions — produce the best answer you can with "
        "the information available."
    )

    return "\n".join(parts)


def _get_llm() -> BaseChatModel:
    provider = _cfg.get_str("llm", "provider")
    model = _cfg.get_str("llm", "model")
    base_url = _cfg.get_str("llm", "base_url", default="")

    cfg = LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url or None,
    )

    if base_url and "localhost" in base_url:
        cfg.streaming = False

    llm = LLM(config=cfg)

    if llm is None:
        raise RuntimeError("Failed to create LLM")

    return llm

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def invoke_subagent(
    config: SubAgentConfig,
    prompt: str,
    context: str,
) -> str:
    """
    Build a fresh LangChain agent from *config* and run it against *prompt*.

    Each invocation gets its own isolated memory window so sub-agents
    never bleed state into each other across main-agent tool calls.

    Parameters
    ----------
    config  : validated SubAgentConfig produced by the markdown parser
    prompt  : the task instruction forwarded from the main agent
    context : shared orchestrator state (memory, prior results, constraints)

    Returns
    -------
    A plain string result ready for the main agent to consume.
    """
    logger.info("[%s] invoke_subagent — prompt=%r", config.name, prompt[:80])

    tools       = _resolve_tools(config)
    system_msg  = _build_system_prompt(config, context)
    llm         = _get_llm()

    # Window memory — keeps the last k exchanges so the sub-agent can
    # self-correct across its own ReAct loop without accumulating stale history.
    memory = ConversationBufferWindowMemory(
        k=config.max_iterations,        # window size mirrors the iteration cap
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )

    # STRUCTURED_CHAT handles multi-arg tools correctly (e.g. lc_web_search).
    # Falls back gracefully when the tool list is empty.
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={
            "system_message": SystemMessage(content=system_msg),
        },
        max_iterations=config.max_iterations,
        early_stopping_method="generate",   # produce an answer even on iteration cap
        handle_parsing_errors=True,         # recover from malformed LLM JSON
        verbose=logger.isEnabledFor(logging.DEBUG),
    )

    try:
        result = agent.invoke({"input": prompt})
        output: str = result.get("output", "")
        if not output.strip():
            logger.warning("[%s] Agent returned an empty output.", config.name)
            return f"[{config.name}] No output produced."
        logger.info("[%s] Completed. Output length: %d chars", config.name, len(output))
        return output

    except Exception as exc:
        logger.exception("[%s] Agent raised an exception: %s", config.name, exc)
        return f"[{config.name}] Failed with error: {exc}"