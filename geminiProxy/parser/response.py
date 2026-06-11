"""
──────────────────────
Parse raw Gemini text into an OpenAI-compatible response.

Scans for ALL TOOL_CALL_JSON: blocks in the response and returns
them as parallel tool_calls — or falls back to plain text.
"""
from __future__ import annotations

import json
import logging
import uuid

from config import settings
from schemas.openai_output import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ToolCall,
    ToolCallFunction,
    Usage,
)

logger = logging.getLogger(__name__)

SENTINEL = settings.tool_call_sentinel


def _extract_all_tool_calls(text: str) -> list[dict]:
    """
    Find every TOOL_CALL_JSON: block in *text* and return a list of
    parsed dicts.  Blocks that fail JSON parsing are skipped.
    """
    results: list[dict] = []
    search_from = 0

    while True:
        idx = text.find(SENTINEL, search_from)
        if idx == -1:
            break

        after = text[idx + len(SENTINEL):].strip()

        # Find the opening brace
        brace_start = after.find("{")
        if brace_start == -1:
            break

        # Walk to the matching closing brace
        depth = 0
        brace_end = -1
        for i, ch in enumerate(after[brace_start:], start=brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break

        if brace_end == -1:
            break

        json_str = after[brace_start:brace_end]
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and data.get("name"):
                results.append(data)
                logger.debug("Parsed tool call: %s", data.get("name"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed tool-call JSON: %s | %r", exc, json_str)

        # Advance past this block and look for the next sentinel
        search_from = idx + len(SENTINEL) + brace_end

    return results


def _build_tool_calls_response(model: str, tool_data_list: list[dict]) -> ChatCompletionResponse:
    tool_calls: list[ToolCall] = []

    for data in tool_data_list:
        fn_name = data.get("name", "")
        raw_args = data.get("arguments", {})

        if isinstance(raw_args, dict):
            arguments_str = json.dumps(raw_args)
        elif isinstance(raw_args, str):
            try:
                json.loads(raw_args)
                arguments_str = raw_args
            except json.JSONDecodeError:
                arguments_str = json.dumps({"raw": raw_args})
        else:
            arguments_str = json.dumps(raw_args)

        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                function=ToolCallFunction(name=fn_name, arguments=arguments_str),
            )
        )

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=None,
                    tool_calls=tool_calls,
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=Usage(),
    )


def _build_text_response(model: str, text: str) -> ChatCompletionResponse:
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
    Parse *raw_text* and return a ChatCompletionResponse.

    Collects ALL TOOL_CALL_JSON: blocks — supports parallel tool calls.
    Falls back to plain text if none are found.
    """
    raw_text = raw_text.strip()

    if not raw_text:
        logger.warning("Gemini returned an empty response.")
        return _build_text_response(model, "")

    tool_data_list = _extract_all_tool_calls(raw_text)

    if tool_data_list:
        logger.debug("Returning %d tool call(s).", len(tool_data_list))
        return _build_tool_calls_response(model, tool_data_list)

    return _build_text_response(model, raw_text)