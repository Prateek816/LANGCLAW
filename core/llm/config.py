"""
LLM configuration dataclass.

Supported providers
-------------------
Built-in  : openai | gemini | anthropic | ollama | groq
Compatible: custom | openai_compatible   (require base_url)

Usage
-----
# Quick default
cfg = LLMConfig.default("groq", temperature=0.3)

# Fully explicit
cfg = LLMConfig(provider="openai", model="gpt-4o")

# Custom / OpenAI-compatible endpoint
cfg = LLMConfig(
    provider="custom",
    model="my-model",
    base_url="https://api.example.com/v1",
)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Extend this union whenever a new provider is added.
Provider = Literal[
    "openai",
    "gemini",
    "anthropic",
    "ollama",
    "groq",
    "custom",
    "openai_compatible",
]

# Providers that ship with a known base URL — no base_url required from the user.
_BUILTIN_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "gemini", "anthropic", "ollama", "groq"}
)

# Providers that route through ChatOpenAI and require the user to supply base_url.
_OPENAI_COMPATIBLE_PROVIDERS: frozenset[str] = frozenset(
    {"custom", "openai_compatible"}
)

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai":    {"model": "gpt-4o-mini"},
    "gemini":    {"model": "gemini-2.0-flash"},
    "anthropic": {"model": "claude-3-5-haiku-20241022"},
    "ollama":    {"model": "llama3.2"},
    "groq":      {"model": "llama-3.3-70b-versatile"},
}


@dataclass
class LLMConfig:
    """
    Unified configuration for every LLM provider.

    Parameters
    ----------
    provider:
        One of the Provider literals.
    model:
        Model name/identifier passed directly to the SDK.
    temperature:
        Sampling temperature (0–2 for most providers).
    max_tokens:
        Maximum tokens in the completion.
    streaming:
        Enable streaming mode on the LangChain wrapper.
    base_url:
        Base URL override.  *Required* for custom/openai_compatible providers;
        *optional* override for built-in ones (e.g. a local Ollama server).
    extra:
        Arbitrary kwargs forwarded verbatim to the LangChain constructor.
        Use sparingly — prefer first-class fields for things you rely on.
    """

    provider: Provider
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = True
    base_url: str | None = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Backward-compat shim: callers that set `baseURL` (old camelCase)
    # are silently redirected to `base_url`.  Remove after one release cycle.
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Pick up legacy camelCase from extra dict if someone passed it there
        if "baseURL" in self.extra and self.base_url is None:
            self.base_url = self.extra.pop("baseURL")

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, provider: Provider, **overrides) -> "LLMConfig":
        """
        Return a config with sensible defaults for *provider*.

        Only available for built-in providers.  Custom endpoints have no
        meaningful default model, so use the constructor directly.

        Example
        -------
        cfg = LLMConfig.default("groq", temperature=0.3)
        """
        if provider not in _PROVIDER_DEFAULTS:
            known = ", ".join(sorted(_PROVIDER_DEFAULTS))
            raise ValueError(
                f"LLMConfig.default() only supports built-in providers "
                f"({known}).  For custom endpoints construct LLMConfig directly."
            )
        return cls(provider=provider, **{**_PROVIDER_DEFAULTS[provider], **overrides})

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """
        Construct from a plain dict (e.g. parsed JSON config).

        Unknown keys are forwarded into *extra* rather than raising.

        Example
        -------
        cfg = LLMConfig.from_dict({
            "provider": "custom",
            "model": "qwen3",
            "base_url": "https://my-endpoint.com/v1",
        })
        """
        known_fields = {
            "provider", "model", "temperature",
            "max_tokens", "streaming", "base_url",
        }
        init_kwargs: dict = {}
        extra: dict = {}

        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extra[key] = value

        if extra:
            logger.debug("Forwarding extra keys to LLMConfig: %s", list(extra.keys()))
            init_kwargs["extra"] = extra

        return cls(**init_kwargs)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_custom(self) -> bool:
        """True when the provider uses a user-supplied OpenAI-compatible endpoint."""
        return self.provider in _OPENAI_COMPATIBLE_PROVIDERS

    @property
    def is_builtin(self) -> bool:
        """True for the four first-class providers with their own SDK wrapper."""
        return self.provider in _BUILTIN_PROVIDERS