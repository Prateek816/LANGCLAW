import logging
from typing import Any
from langchain_groq import ChatGroq
from .base import BaseProvider , LLMFactory
import os
from core.config import LLMConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq



class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini models."""
    def create_client(self, **kwargs: Any) -> ChatGoogleGenerativeAI:
        api_key = LLMConfig.google_api_key
        model = LLMConfig.model

        if not api_key:
            raise ValueError("Missing Gemini config: GOOGLE_API_KEY not set.")

        logging.debug(f"Initializing Gemini model: {model}")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            **kwargs,
        )
class GroqProvider(BaseProvider):
    """Provider implementation for Groq models."""
    def create_client(self, **kwargs: Any) -> ChatGroq:
        api_key = LLMConfig.groq_api_key
        model = LLMConfig.model

        if not api_key:
            raise ValueError("Missing Groq config: GROQ_API_KEY not set.")

        logging.debug(f"Initializing Groq model: {model}")
        return ChatGroq(
            api_key=api_key,
            model=model,
            **kwargs,
        )

def _register_configured_provider():
    """Register only the LLM provider that's configured in settings."""
    provider_name = LLMConfig.provider.lower()

    # Mapping helps keep registration clean as you add more providers
    provider_mapping = {
        "groq": GroqProvider,
        "gemini": GeminiProvider,
    }

    if provider_name in provider_mapping:
        LLMFactory.register_provider(provider_name, provider_mapping[provider_name])
    else:
        logging.warning(f"Unknown LLM provider configured: {provider_name}")


# Register the configured provider on module import
_register_configured_provider()
    