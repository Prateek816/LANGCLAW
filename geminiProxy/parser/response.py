"""
app/parser/response.py
──────────────────────
Parse the raw text string returned by Gemini Web and convert it into
the appropriate OpenAI-compatible response object.

Two outcomes are possible:
  1. The response is a normal assistant text reply.
  2. The response contains a tool-call block (identified by the sentinel).

Tool-call detection
───────────────────
Gemini is instructed to output exactly:

    TOOL_CALL_JSON:
    {
      "name": "<fn>",
      "arguments": { ... }
    }

The parser searches for the sentinel string, extracts the JSON that
follows, validates it, and builds a ToolCall response.  If parsing
fails for any reason the whole response is treated as plain text.
"""
from __future__ import annotations

import json
import logging
import re
import uuid

from app.config import settings
from app.schemas.openai_output import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ToolCall,
    ToolCallFunction,
    Usage,
)

logger = logging.getLogger(__name__)

SENTINEL = settings.tool_call_sentinel


def _extract_tool_call_json(text: str) -> dict | None:
    """
    Look for the sentinel in *text* and return the JSON object that
    follows it, or None if not found / not valid JSON.
    """
    idx = text.find(SENTINEL)
    if idx == -1:
        return None

    after_sentinel = text[idx + len(SENTINEL) :].strip()

    # Find the first {...} block
    brace_start = after_sentinel.find("{")
    if brace_start == -1:
        return None

    # Walk forward to find the matching closing brace
    depth = 0
    brace_end = -1
    for i, ch in enumerate(after_sentinel[brace_start:], start=brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    if brace_end == -1:
        return None

    json_str = after_sentinel[brace_start:brace_end]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Tool-call JSON parse error: %s | raw: %r", exc, json_str)
        return None

    if not isinstance(data, dict):
        return None

    return data


def _build_tool_call_response(model: str, data: dict) -> ChatCompletionResponse:
    """Build an OpenAI tool_calls response from the parsed JSON dict."""
    fn_name = data.get("name", "")
    raw_args = data.get("arguments", {})

    # arguments must be a JSON-encoded string in the OpenAI schema
    if isinstance(raw_args, dict):
        arguments_str = json.dumps(raw_args)
    elif isinstance(raw_args, str):
        # Validate it's actually JSON
        try:
            json.loads(raw_args)
            arguments_str = raw_args
        except json.JSONDecodeError:
            arguments_str = json.dumps({"raw": raw_args})
    else:
        arguments_str = json.dumps(raw_args)

    tool_call = ToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        function=ToolCallFunction(name=fn_name, arguments=arguments_str),
    )

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=Usage(),
    )


def _build_text_response(model: str, text: str) -> ChatCompletionResponse:
    """Build a standard assistant text response."""
    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=text.strip(),
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(),
    )


# ── Public API ─────────────────────────────────────────────────────────────


def parse_gemini_response(raw_text: str, model: str) -> ChatCompletionResponse:
    """
    Parse *raw_text* (Gemini's reply) and return a ChatCompletionResponse.

    Tries tool-call detection first; falls back to plain text.
    """
    raw_text = raw_text.strip()

    if not raw_text:
        logger.warning("Gemini returned an empty response.")
        return _build_text_response(model, "")

    tool_data = _extract_tool_call_json(raw_text)
    if tool_data is not None:
        fn_name = tool_data.get("name", "")
        if fn_name:
            logger.debug("Detected tool call: %s", fn_name)
            return _build_tool_call_response(model, tool_data)
        else:
            logger.warning(
                "Tool-call sentinel found but 'name' is missing; "
                "treating as plain text."
            )

    return _build_text_response(model, raw_text)