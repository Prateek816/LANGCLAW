# Tool Module

A two-layer tool system for LLM agents: pure Python implementations with sandboxing (`tools.py`) and LangChain `@tool` wrappers (`langtools.py`). Provides file operations, shell execution, web search, and a meta-skill for runtime skill creation.

## Features

- **Sandboxed file writes** — restricted to configured root directories
- **Auto venv detection** — subprocess commands run in the project's virtual environment
- **Web search** — Tavily-powered search with summary and source extraction
- **Meta-skill creation** — the agent can create new tools at runtime
- **Dual interface** — raw Python functions + LangChain `@tool` wrappers
- **Channel-agnostic file sending** — works across Telegram, Discord, WhatsApp, Web
- **OpenAI-compatible schemas** — JSON function-call schemas for direct OpenAI usage

## Installation

```bash
# Core (always needed)
pip install langchain-core

# For web search
pip install tavily-python
```

## Quick Start

### Using LangChain Tools

```python
from core.tool.langtools import get_all_langchain_tools

tools = get_all_langchain_tools()

# Use a tool
result = tools[0].invoke({"command": "echo hello"})
print(result)
```

### Using Raw Implementations

```python
from core.tool.tools import run_command, read_file, write_file

# Run a shell command
output = run_command("ls -la")

# Read a file
content = read_file("README.md")

# Write a file (sandboxed)
write_file("output.txt", "Hello, world!")
```

### Web Search

```python
from core.tool.tools import web_search

results = web_search("latest Python releases", max_results=5)
print(results)
```

### Create a Skill at Runtime

```python
from core.tool.tools import create_skill

create_skill(
    name="deploy_checker",
    description="Check if deployment is healthy",
    instructions="curl the health endpoint and report status",
    category="devops",
    resources={"check.sh": "#!/bin/bash\ncurl localhost:8080/health"},
    dependencies=["curl"],
)
```

## API Reference

### Primitive Tools (`tools.py`)

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `run_command(command)` | `str` | `str` | Execute shell command (60s timeout, venv-aware) |
| `read_file(path)` | `str` | `str` | Read file contents as UTF-8 |
| `write_file(path, content)` | `str, str` | `str` | Write content to file (sandboxed) |
| `list_files(path=".")` | `str` | `str` | List directory contents, sorted |
| `send_file(path, caption="")` | `str, str` | `str` | Send file via active channel (max 100 MB) |

### Web Search

```python
web_search(
    query: str,
    *,
    search_depth: str = "basic",     # "basic" or "advanced"
    topic: str = "general",          # "general", "news", "finance"
    max_results: int = 3,
    time_range: str | None = None,   # "day", "week", "month", "year"
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str
```

**Requires:** `TAVILY_API_KEY` environment variable.

### Meta-Skill

```python
create_skill(
    name: str,
    description: str,
    instructions: str,
    category: str = "",
    resources: dict[str, str] | None = None,    # filename -> content
    dependencies: list[str] | None = None,       # pip packages to install
) -> str
```

Creates a `SKILL.md` file with YAML frontmatter under `~/.langclaw/context/skills/<category>/<name>/`.

### Configuration Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `configure_venv(venv_dir=None)` | `str \| None` | `str \| None` | Detect or set the project virtual environment |
| `set_sandbox(roots)` | `list[str]` | `None` | Set allowed directories for file writes |
| `set_file_sender(fn)` | `callable \| None` | `None` | Register callback for channel file delivery |

### LangChain Wrappers (`langtools.py`)

All wrappers are prefixed with `lc_` and delegate to `tools.py`:

| Wrapper | Delegates To | Status |
|---------|-------------|--------|
| `lc_run_command` | `run_command` | Implemented |
| `lc_read_file` | `read_file` | Implemented |
| `lc_write_file` | `write_file` | Implemented |
| `lc_list_files` | `list_files` | Implemented |
| `lc_send_file` | `send_file` | Implemented |
| `lc_web_search` | `web_search` | Implemented |
| `lc_create_skill` | `create_skill` | Implemented |
| `lc_use_skill` | — | Stub |
| `lc_list_skill_resources` | — | Stub |
| `lc_remember` | — | Stub |
| `lc_recall` | — | Stub |
| `lc_memory_get` | — | Stub |
| `lc_memory_list_files` | — | Stub |
| `lc_forget` | — | Stub |
| `lc_update_index` | — | Stub |
| `lc_cron_add` | — | Stub |
| `lc_cron_remove` | — | Stub |
| `lc_cron_list` | — | Stub |

### Tool Groups

Exported from `langtools.py` as convenience lists:

| Group | Contents |
|-------|----------|
| `primitive_tools` | `lc_run_command`, `lc_read_file`, `lc_write_file`, `lc_list_files`, `lc_send_file` |
| `web_search_tool` | `lc_web_search` |
| `skill_tools` | `lc_use_skill`, `lc_list_skill_resources` |
| `memory_tools` | `lc_remember`, `lc_recall`, `lc_memory_get`, `lc_memory_list_files`, `lc_forget`, `lc_update_index` |
| `meta_skill_tools` | `lc_create_skill` |
| `cron_tools` | `lc_cron_add`, `lc_cron_remove`, `lc_cron_list` |

## Architecture

```
tools.py (backend layer)
  ├── Pure Python implementations
  ├── OpenAI-compatible JSON schemas (PRIMITIVE_TOOLS, MEMORY_TOOLS, etc.)
  ├── Sandboxing (set_sandbox / _resolve_in_sandbox)
  ├── Venv detection (configure_venv / _venv_env)
  ├── Tavily web search (web_search)
  └── Meta-skill creation (create_skill)

langtools.py (adapter layer)
  ├── LangChain @tool wrappers (lc_* functions)
  ├── Grouped exports (primitive_tools, memory_tools, etc.)
  └── get_all_langchain_tools() — flat list of all tools
```

## Security

### Sandbox

`write_file` and `create_skill` are restricted to sandbox roots:

```python
from core.tool.tools import set_sandbox

# Allow writes only in the project directory
set_sandbox(["/path/to/project"])
```

Paths are resolved to absolute real paths and checked against roots. `PermissionError` is raised for out-of-bounds writes. Path traversal (`..`, `/`) is stripped from filenames.

### Subprocess Execution

`run_command` runs with `shell=True` but inside the project's venv with a 60-second timeout. Commands execute from the shared files directory (`~/.langclaw/context/files/`).

## File Overview

| File | Purpose |
|------|---------|
| `tools.py` | Backend implementations, JSON schemas, sandbox, venv detection, web search, skill creation |
| `langtools.py` | LangChain `@tool` wrappers, grouped exports, `get_all_langchain_tools()` |

## Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `TAVILY_API_KEY` | Web search | Tavily API key for `web_search()` |
| `LANGCLAW_HOME` | File paths | Root directory (default: `~/.langclaw`) |