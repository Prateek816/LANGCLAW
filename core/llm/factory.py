from __future__ import annotations

import os
from functools import lru_cache
from typing import Union

from langchain_core.language_models.chat_models import BaseChatModel

from .config import LLMConfig, Provider
from .exceptions import LLMConfigurationError, ProviderNotSupportedError


def get_llm(
    provider: Provider | None = None,
    model: str | None = None,
    config: LLMConfig | None = None,
    **kwargs,
) -> BaseChatModel:
    """
    Main entry point. Two usage styles:

        # Quick, explicit
        llm = get_llm("openai", "gpt-4o")

        # Config-driven (recommended for production)
        cfg = LLMConfig.default("gemini", temperature=0.3)
        llm = get_llm(config=cfg)
    """
    if config is None:
        if provider is None:
            raise ValueError("Provide either `config` or `provider`.")
        config = LLMConfig(
            provider=provider,
            model=model or LLMConfig.default(provider).model,
            **kwargs,
        )

    builder = _BUILDERS.get(config.provider)
    if builder is None:
        raise ProviderNotSupportedError(config.provider)

    return builder(config)


# ── private builders ──────────────────────────────────────────────────────────

def _build_openai(config: LLMConfig) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is not set.")

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=config.streaming,
        api_key=api_key,
        **config.extra,
    )


def _build_gemini(config: LLMConfig) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMConfigurationError("GOOGLE_API_KEY is not set.")

    return ChatGoogleGenerativeAI(
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        streaming=config.streaming,
        google_api_key=api_key,
        **config.extra,
    )


def _build_anthropic(config: LLMConfig) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMConfigurationError("ANTHROPIC_API_KEY is not set.")

    return ChatAnthropic(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=config.streaming,
        anthropic_api_key=api_key,
        **config.extra,
    )


def _build_ollama(config: LLMConfig) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.model,
        temperature=config.temperature,
        num_predict=config.max_tokens,
        # Ollama doesn't need an API key; base_url can be overridden via extra
        **config.extra,
    )


_BUILDERS = {
    "openai":    _build_openai,
    "gemini":    _build_gemini,
    "anthropic": _build_anthropic,
    "ollama":    _build_ollama,
}