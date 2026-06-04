# LLM Module

A unified, provider-agnostic interface for interacting with large language models. Supports OpenAI, Google Gemini, Anthropic, Ollama, and Groq through a single factory function.

## Features

- **Multi-provider support** — switch between OpenAI, Gemini, Anthropic, Ollama, and Groq with a single config change
- **Streaming support** — synchronous and asynchronous streaming out of the box
- **Lazy imports** — only the LangChain provider package for your chosen provider gets loaded
- **Type-safe configuration** — dataclass-based config with sensible defaults
- **Custom exceptions** — structured error hierarchy for LLM failures

## Installation

Install the core dependency and the provider package for your chosen LLM:

```bash
# Core (always needed)
pip install langchain-core

# Choose one or more providers:
pip install langchain-openai        # OpenAI
pip install langchain-google-genai  # Google Gemini
pip install langchain-anthropic     # Anthropic Claude
pip install langchain-ollama        # Ollama (local)
pip install langchain-groq          # Groq
```

## Configuration

Set environment variables for the providers you plan to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
# Ollama requires no API key — just a running Ollama server
```

## Quick Start

### Simple (provider + model)

```python
from core.llm import get_llm

llm = get_llm("openai", "gpt-4o")
response = llm.invoke("Hello, world!")
print(response.content)
```

### Config-driven

```python
from core.llm import get_llm, LLMConfig

config = LLMConfig.default("gemini", temperature=0.3, max_tokens=4096)
llm = get_llm(config=config)
response = llm.invoke("Explain quantum computing in one sentence.")
print(response.content)
```

### Streaming

```python
from core.llm import get_llm, stream_response
from langchain_core.messages import HumanMessage

llm = get_llm("groq", "llama-3.3-70b-versatile")
messages = [HumanMessage(content="Write a haiku about coding.")]

for chunk in stream_response(llm, messages):
    print(chunk, end="", flush=True)
```

### Async Streaming

```python
import asyncio
from core.llm import get_llm
from core.llm.streaming import astream_response
from langchain_core.messages import HumanMessage

async def main():
    llm = get_llm("anthropic", "claude-3-5-haiku-20241022")
    messages = [HumanMessage(content="Tell me a joke.")]
    async for chunk in astream_response(llm, messages):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## API Reference

### `get_llm(provider, model, config, **kwargs)`

Factory function that returns a LangChain `BaseChatModel`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `str \| None` | `"openai"`, `"gemini"`, `"anthropic"`, `"ollama"`, or `"groq"` |
| `model` | `str \| None` | Model name (e.g., `"gpt-4o"`, `"llama3.2"`) |
| `config` | `LLMConfig \| None` | Pre-built config object (takes precedence) |
| `**kwargs` | | Forwarded to `LLMConfig` constructor |

**Returns:** A LangChain `BaseChatModel` instance.

### `LLMConfig` (dataclass)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `Provider` | (required) | LLM backend |
| `model` | `str` | (required) | Model identifier |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `2048` | Max output tokens |
| `streaming` | `bool` | `True` | Enable streaming |
| `extra` | `dict` | `{}` | Extra kwargs for the LangChain constructor |

**Class method:** `LLMConfig.default(provider, **overrides)` — returns a config with provider-specific defaults.

### `stream_response(llm, messages, **kwargs)`

Synchronous streaming generator.

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm` | `BaseChatModel` | The LLM instance |
| `messages` | `list[BaseMessage]` | LangChain message objects |

**Returns:** A generator yielding text chunks. The `StopIteration.value` contains the fully assembled response.

### `astream_response(llm, messages, **kwargs)`

Async streaming generator. Same interface as `stream_response`, used with `async for`.

## Provider Defaults

| Provider | Default Model | API Key Env Var |
|----------|--------------|-----------------|
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `gemini` | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| `anthropic` | `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` |
| `ollama` | `llama3.2` | (none — local) |
| `groq` | (no default) | `GROQ_API_KEY` |

## Custom Exceptions

```
LLMError
  ├── ProviderNotSupportedError   # Unknown provider string
  ├── LLMConfigurationError       # Missing/invalid API key or config
  └── LLMStreamError              # Streaming failure mid-response
```

## File Overview

| File | Purpose |
|------|---------|
| `__init__.py` | Package entry point; exports `get_llm`, `LLMConfig`, `stream_response` |
| `config.py` | `LLMConfig` dataclass and `Provider` type alias |
| `factory.py` | `get_llm()` factory with per-provider builder functions |
| `streaming.py` | `stream_response()` and `astream_response()` wrappers |
| `exceptions.py` | Custom exception hierarchy |