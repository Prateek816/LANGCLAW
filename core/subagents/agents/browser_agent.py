"""
core/subagents/agents/browser_agent.py

Browser-use subagent — executes any browser automation task using
a system Chrome instance and a locally configured OpenAI-compatible
LLM proxy (e.g. LangClaw's Gemini proxy).

Environment variables
---------------------
BROWSER_AGENT_BASE_URL   : OpenAI-compatible base URL  (default: http://localhost:8000/v1)
BROWSER_AGENT_MODEL      : model name to request        (default: chatgpt-4o-latest)
BROWSER_AGENT_API_KEY    : API key sent in the header   (default: dummy)
BROWSER_AGENT_MAX_STEPS  : hard cap on agent steps      (default: 25)
"""

from __future__ import annotations

import asyncio
import logging
import os

from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI

logger = logging.getLogger(__name__)

# ── registry contract ──────────────────────────────────────────

NAME = "browser_agent"

DESCRIPTION = (
    "Controls a real Chrome browser to complete any web-based task autonomously. "
    "Use this when the task requires navigating websites, filling forms, reading "
    "live web content, interacting with web apps (Gmail, GitHub, etc.), or doing "
    "anything that needs an actual browser session. "
    "Pass the full task description as `prompt`. "
    "Use `context` to supply any extra constraints (e.g. 'only read, do not send')."
)


# ── helpers ────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    """Construct the LLM client from environment variables."""
    return ChatOpenAI(
        model=os.getenv("BROWSER_AGENT_MODEL", "chatgpt-4o-latest"),
        api_key=os.getenv("BROWSER_AGENT_API_KEY", "dummy"),
        base_url=os.getenv("BROWSER_AGENT_BASE_URL", "http://localhost:8000/v1"),
    )


def _build_task(prompt: str, context: str) -> str:
    """
    Merge the orchestrator's prompt + context into a single task string
    that browser-use will execute.

    If context is empty or whitespace, only the prompt is used so the
    agent doesn't get confused by empty constraint blocks.
    """
    if context and context.strip():
        return (
            f"{prompt.strip()}\n\n"
            f"Additional constraints / context from the orchestrator:\n"
            f"{context.strip()}"
        )
    return prompt.strip()


async def _run_browser_task(task: str) -> str:
    """
    Spin up a browser-use Agent, run the task to completion, and return
    the final result as a plain string.

    The Browser is created and closed inside this coroutine so each
    subagent call gets a clean, isolated browser session.
    """
    max_steps: int = int(os.getenv("BROWSER_AGENT_MAX_STEPS", "25"))

    browser = Browser.from_system_chrome()
    try:
        agent = Agent(
            task=task,
            browser=browser,
            llm=_build_llm(),
        )
        result = await agent.run(max_steps=max_steps)
        # browser-use returns an AgentHistoryList; .final_result() gives
        # the last extracted value or a plain summary string.
        final = result.final_result()
        return final if isinstance(final, str) else str(final)

    except Exception as exc:
        logger.exception("browser_agent: task failed — %s", exc)
        return f"[browser_agent] Task failed with error: {exc}"

    finally:
        try:
            await browser.close()
        except Exception:
            pass  # best-effort cleanup


# ── public entry point ─────────────────────────────────────────

def call(prompt: str, context: str) -> str:
    """
    Synchronous entry point required by BaseRegistry / StructuredTool.

    Handles three event-loop situations:
    1. No loop running (normal script / thread)   → asyncio.run()
    2. Loop running but this call is in a thread  → new loop in thread
    3. Inside an already-running async context    → nest_asyncio fallback

    LangClaw's main agent is sync-wrapped, so case 1 is the common path.
    """
    task = _build_task(prompt, context)
    logger.info("browser_agent: starting task → %r", task[:120])

    try:
        # Case 1 & 2 — no running loop in the current thread
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop closed")
        return loop.run_until_complete(_run_browser_task(task))

    except RuntimeError:
        # Loop is closed or doesn't exist — create a fresh one
        return asyncio.run(_run_browser_task(task))

    except Exception as exc:
        # Case 3 — already inside a running loop (e.g. Jupyter / some async frameworks)
        # Try nest_asyncio as a last resort; it's an optional dep.
        try:
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run_browser_task(task))
        except ImportError:
            logger.error(
                "browser_agent: called from inside a running event loop. "
                "Install `nest_asyncio` or call this agent from a sync context."
            )
            return f"[browser_agent] Event loop conflict: {exc}"