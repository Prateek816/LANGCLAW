# Memory Module

A markdown-backed memory system with hybrid RAG retrieval. Stores facts, conversation history, and user context as structured `.md` files on disk, with intelligent recall powered by BM25 + optional dense embedding search.

## Features

- **Markdown-native storage** — human-readable, version-controllable memory files
- **Hybrid RAG recall** — BM25 sparse search + optional dense embedding retrieval
- **Global + local memory** — supports shared global memories and per-session/group isolated memories
- **Boot context** — generates a concise memory snapshot for session initialization
- **Daily logs** — append-only timestamped history of all memory operations
- **Incremental ingestion** — tracks file hashes to skip unchanged content
- **Zero external APIs** — runs entirely locally (no API keys needed for core functionality)

## Installation

```bash
pip install rank-bm25 langchain-core langchain-community langchain-huggingface langchain-text-splitters sentence-transformers
```

## Quick Start

### Basic Usage

```python
from core.memory.manager import MemoryManager

# Initialize with a storage directory
mem = MemoryManager(memory_dir="./my_memory")

# Store facts
mem.remember("User prefers dark mode", key="ui_preference")
mem.remember("Project deadline is March 15", key="deadline")

# Recall relevant memories
results = mem.recall("what UI settings does the user like?")
print(results)

# Get all memories
all_memories = mem.recall("all")

# Delete a memory
mem.forget("deadline")
```

### Boot Context (Session Init)

```python
mem = MemoryManager(memory_dir="./my_memory")

# Get a concise snapshot for injecting into a system prompt
context = mem.boot_context(max_chars=3000)
print(context)
```

### Per-Group Isolation

```python
mem = MemoryManager(
    memory_dir="./groups/telegram_123/memory",        # group-local memories
    global_memory_dir="./shared/global_memory",       # shared across all groups
)
```

## API Reference

### `MemoryManager(memory_dir, global_memory_dir=None, use_dense=False)`

High-level memory API with hybrid RAG recall.

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_dir` | `str \| None` | Path for local memory storage (default: `~/.langclaw/context/memory`) |
| `global_memory_dir` | `str \| None` | Path for shared global memories |
| `use_dense` | `bool` | Enable embedding-based retrieval (default: `False`, BM25 only) |

#### Methods

| Method | Description |
|--------|-------------|
| `remember(content, key)` | Store a fact under a named key |
| `recall(query, top_k=10)` | Search memories by query; returns matching entries as a formatted string |
| `forget(key)` | Delete a memory by key |
| `boot_context(max_chars=3000)` | Generate a concise memory snapshot for session start |
| `list_files()` | List all `.md` files in the memory directory |
| `list_all()` | Return raw `{key: value}` dict of all local memories |

### `MemoryStorage(memory_dir)`

Low-level markdown file persistence. Used internally by `MemoryManager`.

| Method | Description |
|--------|-------------|
| `get(key)` | Retrieve a value by key |
| `set(key, value)` | Store a value under a key |
| `delete(key)` | Remove a key |
| `list_all()` | Return all `{key: value}` pairs |
| `read_index()` | Read `INDEX.md` (curated system info) |
| `write_index(content)` | Write `INDEX.md` |
| `read_recent_daily_logs(days=2)` | Read recent daily log files |
| `read_memory_file(path)` | Read any `.md` file in the memory directory |
| `list_memory_files()` | List all `.md` files |

## Storage Format

### MEMORY.md (Key-Value Store)

```markdown
# Long-Term Memory

## bot_name
> Updated: 2026-06-04 14:30:00

ClawBot

## user_name
> Updated: 2026-06-04 15:00:00

Alice
```

### Daily Logs (`YYYY-MM-DD.md`)

```markdown
# Daily Memory — 2026-06-04

### 14:30:00 — remember

Stored "ClawBot" under key "bot_name"

### 15:00:00 — remember

Stored "Alice" under key "user_name"
```

### INDEX.md (System Info)

Free-form markdown for curated system-level information (personality, user profile, etc.).

## How Recall Works

1. Merges local and global memories (local overrides global with same key)
2. Chunks each memory entry for retrieval
3. Runs BM25 sparse search for keyword matching
4. Optionally runs dense embedding search (sentence-transformers/all-MiniLM-L6-v2)
5. Deduplicates and returns top-k results
6. Falls back to dumping all memories if no RAG hits are found

**Special queries:** `""`, `"*"`, `"all"`, `"everything"` — dumps all memories as a bulleted list.

## File Overview

| File | Purpose |
|------|---------|
| `manager.py` | `MemoryManager` — high-level API with hybrid RAG recall, boot context, global/local merging |
| `storage.py` | `MemoryStorage` — markdown file persistence (MEMORY.md, daily logs, INDEX.md) |

## Configuration

The module reads `LANGCLAW_HOME` from the project config to determine default storage paths. Set it via environment variable or `langclaw.json`:

```bash
export LANGCLAW_HOME="~/.langclaw"
```

Default paths:
- Local memory: `{LANGCLAW_HOME}/context/memory/`
- Global memory: `{LANGCLAW_HOME}/context/memory/` (shared)
- Group memory: `{LANGCLAW_HOME}/context/groups/{group_id}/context/memory/`