"""
LLM-layer exception hierarchy.

All exceptions derive from LLMError so callers can catch broadly
or specifically as needed.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base error for all LLM-related failures."""


class ProviderNotSupportedError(LLMError):
    """
    Raised when an unrecognised provider string is requested.

    The list of known providers is injected at raise-time so it never
    goes stale as new providers are added to the registry.
    """

    def __init__(self, provider: str, known_providers: list[str] | None = None):
        if known_providers:
            known_str = ", ".join(sorted(known_providers))
            msg = (
                f"Provider {provider!r} is not supported. "
                f"Known providers: {known_str}."
            )
        else:
            msg = f"Provider {provider!r} is not supported."
        super().__init__(msg)
        self.provider = provider


class LLMConfigurationError(LLMError):
    """
    Raised when required configuration (API keys, base_url, etc.) is missing
    or invalid.
    """


class LLMStreamError(LLMError):
    """Raised when a streaming response fails mid-generation."""


class LLMMissingBaseURLError(LLMConfigurationError):
    """
    Raised when a custom/openai-compatible provider is configured but
    no base_url was provided.
    """

    def __init__(self, provider: str):
        super().__init__(
            f"Provider {provider!r} requires a base_url. "
            f"Set it in your config: base_url='https://api.example.com/v1'"
        )
        self.provider = provider