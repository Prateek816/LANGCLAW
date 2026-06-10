"""
app/schemas/openai_input.py
───────────────────────────
Pydantic models for the OpenAI Chat Completions request body.
All fields mirror the OpenAI API so that standard clients work
without modification.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Message content ────────────────────────────────────────────────────────


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Union[TextContentPart, ImageContentPart]


# ── Messages ───────────────────────────────────────────────────────────────


class SystemMessage(BaseModel):
    role: Literal["system"]
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None


class UserMessage(BaseModel):
    role: Literal["user"]
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: "ToolCallFunction"


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


ToolCall.model_rebuild()


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: Optional[Union[str, list[ContentPart]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ToolMessage(BaseModel):
    role: Literal["tool"]
    content: Union[str, list[ContentPart]]
    tool_call_id: str


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# ── Tool definitions ───────────────────────────────────────────────────────


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


# ── Tool choice ────────────────────────────────────────────────────────────


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Union[Literal["none", "auto", "required"], ToolChoiceObject]


# ── Top-level request ──────────────────────────────────────────────────────


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier (ignored; Gemini is used)")
    messages: list[Message]
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = "auto"
    stream: Optional[bool] = False

    # Common OpenAI sampling params — accepted but not forwarded to Gemini Web
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}