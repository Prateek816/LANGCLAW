from __future__ import annotations

from collections.abc import Generator, AsyncGenerator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .exceptions import LLMStreamError


def stream_response(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    **kwargs,
) -> Generator[str, None, str]:
    """
    Synchronous streaming. Yields text chunks as they arrive.

    Usage:
        for chunk in stream_response(llm, messages):
            print(chunk, end="", flush=True)

    Returns the full assembled text as the generator's return value —
    accessible via StopIteration.value if you need it.
    """
    full_text = ""
    try:
        for chunk in llm.stream(messages, **kwargs):
            text = chunk.content
            if isinstance(text, list):
                # Anthropic returns content as a list of blocks
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in text
                )
            if text:
                full_text += text
                yield text
    except Exception as exc:
        raise LLMStreamError(f"Streaming failed: {exc}") from exc

    return full_text


async def astream_response(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Async streaming. Use with `async for`.

    Usage:
        async for chunk in astream_response(llm, messages):
            print(chunk, end="", flush=True)
    """
    try:
        async for chunk in llm.astream(messages, **kwargs):
            text = chunk.content
            if isinstance(text, list):
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in text
                )
            if text:
                yield text
    except Exception as exc:
        raise LLMStreamError(f"Async streaming failed: {exc}") from exc