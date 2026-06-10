"""
app/schemas/openai_output.py
────────────────────────────
Pydantic models for the OpenAI Chat Completions response body.
Matches the OpenAI wire format exactly so that standard clients
can parse these responses without modification.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Tool call (inside a choice message) ───────────────────────────────────


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: ToolCallFunction


# ── Choice message ─────────────────────────────────────────────────────────


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


# ── Choice ─────────────────────────────────────────────────────────────────


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]
    logprobs: None = None


# ── Usage ──────────────────────────────────────────────────────────────────


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ── Top-level response ─────────────────────────────────────────────────────


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: Optional[str] = None