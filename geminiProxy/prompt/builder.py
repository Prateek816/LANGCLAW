"""
─────────────────────
Converts an OpenAI-format message list (plus optional tool definitions
and tool_choice) into a single plain-text prompt for Gemini Web.
"""
from __future__ import annotations

import json
from typing import Any

from config import settings
from schemas.openai_input import (
    AssistantMessage,
    ChatCompletionRequest,
    Message,
    SystemMessage,
    Tool,
    ToolMessage,
    UserMessage,
)

SENTINEL = settings.tool_call_sentinel

# ── Tool-call output format shown to Gemini ───────────────────────────────
# Gemini is told to repeat this block once per tool it wants to call.

TOOL_CALL_FORMAT = f"""\
{SENTINEL}
{{
  "name": "<function_name>",
  "arguments": {{
    "<param>": "<value>"
  }}
}}"""


# ── Helpers ───────────────────────────────────────────────────────────────

def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if hasattr(part, "text"):
                parts.append(part.text)
            elif hasattr(part, "image_url"):
                parts.append("[image]")
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _format_tool_list(tools: list[Tool]) -> str:
    lines: list[str] = []
    for tool in tools:
        fn = tool.function
        desc = fn.description or "No description."
        params_json = json.dumps(fn.parameters, indent=2) if fn.parameters else "{}"
        lines.append(f"- {fn.name}: {desc}\n  Parameters: {params_json}")
    return "\n".join(lines)


def _render_message(msg: Message) -> str:
    if isinstance(msg, SystemMessage):
        return f"System: {_content_to_str(msg.content)}"

    if isinstance(msg, UserMessage):
        return f"User: {_content_to_str(msg.content)}"

    if isinstance(msg, AssistantMessage):
        if msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.dumps(json.loads(tc.function.arguments), indent=2)
                except (json.JSONDecodeError, TypeError):
                    args = tc.function.arguments
                calls.append(f"Called {tc.function.name} with:\n{args}")
            return "Assistant (tool calls):\n" + "\n---\n".join(calls)
        return f"Assistant: {_content_to_str(msg.content)}"

    if isinstance(msg, ToolMessage):
        return f"Tool result (id={msg.tool_call_id}): {_content_to_str(msg.content)}"

    return str(msg)


# ── Public API ─────────────────────────────────────────────────────────────

def build_prompt(request: ChatCompletionRequest) -> str:
    """
    Build a plain-text prompt from the full OpenAI request.

    When tools are present Gemini is told it MAY emit multiple
    TOOL_CALL_JSON: blocks — one per tool it wants to call — before
    any prose response.
    """
    has_tools = bool(request.tools)
    conversation = "\n\n".join(_render_message(m) for m in request.messages)

    if not has_tools:
        return f"{conversation}\n\nAssistant:"

    tool_list_str = _format_tool_list(request.tools)  # type: ignore[arg-type]

    return (
        f"You have access to these tools:\n{tool_list_str}\n\n"
        f"If you need to call one or more tools, output ONLY the following block "
        f"repeated once for each tool call — no explanation, no prose, nothing else:\n\n"
        f"{TOOL_CALL_FORMAT}\n\n"
        f"Repeat that block for every tool you want to call. "
        f"If you need to call get_weather AND search_web, output two blocks back to back.\n\n"
        f"If no tool is needed, respond normally.\n\n"
        f"---\n\n"
        f"{conversation}\n\nAssistant:"
    )