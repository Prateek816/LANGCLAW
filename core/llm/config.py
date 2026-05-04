from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["openai", "gemini", "anthropic", "ollama"]


@dataclass
class LLMConfig:
    provider: Provider
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = True
    # Extra kwargs forwarded to the LangChain constructor (e.g. base_url for Ollama)
    extra: dict = field(default_factory=dict)

    # Sensible defaults per provider — call LLMConfig.default("gemini") etc.
    _DEFAULTS: dict = field(init=False, repr=False, default_factory=lambda: {
        "openai":    {"model": "gpt-4o-mini"},
        "gemini":    {"model": "gemini-2.0-flash"},
        "anthropic": {"model": "claude-3-5-haiku-20241022"},
        "ollama":    {"model": "llama3.2"},
    })

    @classmethod
    def default(cls, provider: Provider, **overrides) -> "LLMConfig":
        defaults = {
            "openai":    {"model": "gpt-4o-mini"},
            "gemini":    {"model": "gemini-2.0-flash"},
            "anthropic": {"model": "claude-3-5-haiku-20241022"},
            "ollama":    {"model": "llama3.2"},
        }
        if provider not in defaults:
            raise ValueError(f"Unknown provider: {provider!r}")
        return cls(provider=provider, **{**defaults[provider], **overrides})