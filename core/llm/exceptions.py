class LLMError(Exception):
    """Base error for all LLM-related failures."""


class ProviderNotSupportedError(LLMError):
    def __init__(self, provider: str):
        super().__init__(
            f"Provider {provider!r} is not supported. "
            f"Choose from: openai, gemini, anthropic, ollama"
        )


class LLMConfigurationError(LLMError):
    """Raised when API keys or config are missing/invalid."""


class LLMStreamError(LLMError):
    """Raised when streaming fails mid-response."""