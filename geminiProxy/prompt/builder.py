"""
app/prompt/builder.py
─────────────────────
Converts an OpenAI-format message list (plus optional tool definitions
and tool_choice) into a single plain-text prompt suitable for pasting
into the Gemini Web UI.

Design goals
────────────
1. Gemini must understand the conversation history, including any
   previous tool call / tool result turns.
2. When tools are provided Gemini must be instructed to output tool
   calls in a specific, parseable format rather than prose.
3. The prompt must be self-contained so that every Gemini page can
   be opened fresh with no prior context.
"""
from __future__ import annotations

import json
from typing import Any

from app.schemas.openai_input import (
    AssistantMessage,
    ChatCompletionRequest,
    Message,
    SystemMessage,
    Tool,
    ToolMessage,
    UserMessage,
)

# ── Sentinel used by the response parser ──────────────────────────────────
from app.config import settings

SENTINEL = settings.tool_call_sentinel

# ── Tool-call output schema shown to Gemini ───────────────────────────────
TOOL_CALL_FORMAT_EXAMPLE = """\
{sentinel}
{{
  "name": "<function_name>",
  "arguments": {{
    "<param>": "<value>"
  }}
}}""".format(
    sentinel=SENTINEL
)

# ── System preamble injected before user content ──────────────────────────
TOOL_SYSTEM_PREAMBLE = """\

## Available tools
{tool_list}

## Tool-calling rules
- If you decide a tool should be called, output ONLY the following block
  and nothing else — no explanation, no prose before or after:

{format_example}

- Replace <function_name> with the exact tool name and fill in the
  arguments object with the required parameters.
- If no tool is needed, respond normally in plain text.
- Never invent tool names that are not in the Available tools list.
- Never call more than one tool per response.
"""

NO_TOOL_SYSTEM_PREAMBLE = """\
You are a helpful AI assistant. Respond naturally to the conversation below.
"""


# ── Helpers ───────────────────────────────────────────────────────────────


def _content_to_str(content: Any) -> str:
    """Flatten message content (str or list[ContentPart]) to plain text."""
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
        desc = fn.description or "No description provided."
        params_json = (
            json.dumps(fn.parameters, indent=2) if fn.parameters else "{}"
        )
        lines.append(
            f"### {fn.name}\n{desc}\nParameters (JSON Schema):\n{params_json}"
        )
    return "\n\n".join(lines)


def _format_message(msg: Message) -> str:
    """Render a single message as a labelled block of text."""
    if isinstance(msg, SystemMessage):
        content = _content_to_str(msg.content)
        return f"[SYSTEM]\n{content}"

    if isinstance(msg, UserMessage):
        content = _content_to_str(msg.content)
        return f"[USER]\n{content}"

    if isinstance(msg, AssistantMessage):
        # The assistant turn may contain tool calls or plain text.
        if msg.tool_calls:
            calls_repr: list[str] = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    args_pretty = json.dumps(args, indent=2)
                except (json.JSONDecodeError, TypeError):
                    args_pretty = tc.function.arguments
                calls_repr.append(
                    f"Tool call: {tc.function.name}\nArguments:\n{args_pretty}"
                )
            return "[ASSISTANT — TOOL CALL]\n" + "\n---\n".join(calls_repr)
        content = _content_to_str(msg.content)
        return f"[ASSISTANT]\n{content}"

    if isinstance(msg, ToolMessage):
        content = _content_to_str(msg.content)
        return f"[TOOL RESULT (id={msg.tool_call_id})]\n{content}"

    # Fallback for unexpected types
    return f"[MESSAGE]\n{str(msg)}"


# ── Public API ─────────────────────────────────────────────────────────────


def build_prompt(request: ChatCompletionRequest) -> str:
    """
    Convert a ChatCompletionRequest into a single plain-text prompt
    to send to Gemini Web.

    Structure:
        <system preamble (tool instructions or generic)>
        ──────────────────
        <message 1>
        <message 2>
        …
        ──────────────────
        [ASSISTANT]          ← dangling prefix so Gemini continues here
    """
    has_tools = bool(request.tools)

    # ── Build the system preamble ─────────────────────────────────────────
    if has_tools:
        tool_list_str = _format_tool_list(request.tools)  # type: ignore[arg-type]
        preamble = TOOL_SYSTEM_PREAMBLE.format(
            tool_list=tool_list_str,
            format_example=TOOL_CALL_FORMAT_EXAMPLE,
        )
    else:
        preamble = NO_TOOL_SYSTEM_PREAMBLE

    # ── Render each message ───────────────────────────────────────────────
    message_blocks: list[str] = []
    for msg in request.messages:
        message_blocks.append(_format_message(msg))

    divider = "─" * 50

    conversation = f"\n{divider}\n".join(message_blocks)

    prompt = (
        preamble.strip()
        + f"\n\n{divider}\nCONVERSATION\n{divider}\n\n"
        + conversation
        + f"\n\n{divider}\n[ASSISTANT]\n"
    )

    return prompt