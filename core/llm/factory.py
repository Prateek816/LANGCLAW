"""
LLM factory — maps provider names to LangChain chat model instances.

Public API (unchanged)
----------------------
    get_llm(provider, model, **kwargs)          -> BaseChatModel
    get_llm(config=LLMConfig(...))              -> BaseChatModel

Adding a new provider
---------------------
Just add a function decorated with @_register("<name>") anywhere below
the decorator definition.  It receives an LLMConfig and returns a
BaseChatModel.  No other changes required.

Custom / OpenAI-compatible endpoints
-------------------------------------
    # Via provider="custom" or "openai_compatible"
    get_llm(
        "custom", "my-model",
        base_url="https://api.example.com/v1",
    )

    # Via config dict (e.g. from JSON)
    cfg = LLMConfig.from_dict({
        "provider": "openai_compatible",
        "model": "qwen3",
        "base_url": "https://my-endpoint.com/v1",
    })
    llm = get_llm(config=cfg)
"""

from __future__ import annotations

import os
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from .config import LLMConfig, Provider
from .exceptions import (
    LLMConfigurationError,
    LLMMissingBaseURLError,
    ProviderNotSupportedError,
)

# ---------------------------------------------------------------------------
# Builder registry
# ---------------------------------------------------------------------------

_BuilderFn = Callable[[LLMConfig], BaseChatModel]
_BUILDERS: dict[str, _BuilderFn] = {}


def _register(provider: str) -> Callable[[_BuilderFn], _BuilderFn]:
    """Decorator that registers a builder function under *provider*."""
    def decorator(fn: _BuilderFn) -> _BuilderFn:
        _BUILDERS[provider] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _require_api_key(env_var: str, provider: str) -> str:
    """
    Read *env_var* from the environment and raise a helpful error if absent.
    Call this at the top of every builder that needs an API key.
    """
    value = os.getenv(env_var)
    if not value:
        raise LLMConfigurationError(
            f"{env_var} is not set. "
            f"Export it in your shell or add it to your .env file to use "
            f"the {provider!r} provider."
        )
    return value


def _require_base_url(config: LLMConfig) -> str:
    """
    Validate and return config.base_url for providers that need it.
    Raises LLMMissingBaseURLError when absent.
    """
    if not config.base_url:
        raise LLMMissingBaseURLError(config.provider)
    return config.base_url


# ---------------------------------------------------------------------------
# Built-in provider builders
# ---------------------------------------------------------------------------

@_register("openai")
def _build_openai(config: LLMConfig) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = _require_api_key("OPENAI_API_KEY", "openai")

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=config.streaming,
        api_key=api_key,
        # Allow optional base_url override (e.g. Azure OpenAI or a proxy)
        **({"base_url": config.base_url} if config.base_url else {}),
        **config.extra,
    )


@_register("gemini")
def _build_gemini(config: LLMConfig) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = _require_api_key("GOOGLE_API_KEY", "gemini")

    return ChatGoogleGenerativeAI(
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        streaming=config.streaming,
        google_api_key=api_key,
        **config.extra,
    )


@_register("anthropic")
def _build_anthropic(config: LLMConfig) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = _require_api_key("ANTHROPIC_API_KEY", "anthropic")

    return ChatAnthropic(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=config.streaming,
        api_key=api_key,
        **config.extra,
    )


@_register("ollama")
def _build_ollama(config: LLMConfig) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    # Ollama is local — no API key required.
    # base_url defaults to localhost:11434 inside the SDK; override via config.
    return ChatOllama(
        model=config.model,
        temperature=config.temperature,
        num_predict=config.max_tokens,
        **({"base_url": config.base_url} if config.base_url else {}),
        **config.extra,
    )


@_register("groq")
def _build_groq(config: LLMConfig) -> BaseChatModel:
    from langchain_groq import ChatGroq

    api_key = _require_api_key("GROQ_API_KEY", "groq")

    return ChatGroq(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=config.streaming,
        api_key=api_key,
        **config.extra,
    )


# ---------------------------------------------------------------------------
# Custom / OpenAI-compatible endpoint builder
#
# Both "custom" and "openai_compatible" route through ChatOpenAI with a
# user-supplied base_url.  The OPENAI_API_KEY env var is used when present;
# many self-hosted endpoints accept any non-empty string, so we fall back
# to "not-needed" rather than hard-failing.  Users who need a real key just
# set it in .env as usual.
# ---------------------------------------------------------------------------

def _build_openai_compatible(config: LLMConfig) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    base_url = _require_base_url(config)

    # Use the real key if available; many OSS endpoints accept any value.
    api_key = os.getenv("OPENAI_API_KEY") or "not-needed"

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=config.streaming,
        api_key=api_key,
        base_url=base_url,
        **config.extra,
    )


# Register the same builder under both alias names
_BUILDERS["custom"] = _build_openai_compatible
_BUILDERS["openai_compatible"] = _build_openai_compatible


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_llm(
    provider: Provider | None = None,
    model: str | None = None,
    config: LLMConfig | None = None,
    **kwargs,
) -> BaseChatModel:
    """
    Instantiate and return a LangChain chat model.

    Two usage styles are supported (external API is unchanged):

    Style 1 — quick, positional (used by existing callers):
        llm = get_llm("openai", "gpt-4o")
        llm = get_llm("groq", "llama-3.3-70b-versatile")

    Style 2 — config-driven (recommended for production):
        cfg = LLMConfig.default("gemini", temperature=0.3)
        llm = get_llm(config=cfg)

    Style 3 — custom / OpenAI-compatible endpoint:
        llm = get_llm("custom", "my-model", base_url="https://api.example.com/v1")
        # or via config:
        cfg = LLMConfig(provider="openai_compatible", model="qwen3",
                        base_url="https://my-endpoint.com/v1")
        llm = get_llm(config=cfg)

    Parameters
    ----------
    provider:
        Provider identifier string.  Required when *config* is not supplied.
    model:
        Model name/identifier.  Falls back to the provider default when omitted.
    config:
        Pre-built LLMConfig.  When given, *provider*, *model*, and extra kwargs
        are ignored.
    **kwargs:
        Forwarded into LLMConfig (temperature, max_tokens, streaming, base_url …).
    """
    if config is None:
        if provider is None:
            raise ValueError(
                "Provide either `config=LLMConfig(...)` or a `provider` string."
            )

        # Resolve model default lazily so callers can omit it
        resolved_model = model or _default_model_for(provider)

        config = LLMConfig(
            provider=provider,
            model=resolved_model,
            **kwargs,
        )

    builder = _BUILDERS.get(config.provider)
    if builder is None:
        raise ProviderNotSupportedError(
            config.provider, known_providers=list(_BUILDERS)
        )

    return builder(config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_model_for(provider: str) -> str:
    """
    Return the default model name for *provider*, or an empty string for
    unknown/custom providers (the caller must then supply model explicitly).
    """
    from .config import _PROVIDER_DEFAULTS
    entry = _PROVIDER_DEFAULTS.get(provider, {})
    return entry.get("model", "")


def list_providers() -> list[str]:
    """Return all currently registered provider names (useful for diagnostics)."""
    return sorted(_BUILDERS)