# LangClaw

<img width="432" height="247" alt="LangClaw" src="https://github.com/user-attachments/assets/fee52b60-6e06-4548-918c-9b0183874712" />

An autonomous AI agent framework built in Python with LangChain. LangClaw connects LLMs to real-world tools, persistent memory, knowledge retrieval, and scheduled tasks — accessible via Telegram or a local REPL.Langclaw is deeply inspired by Openclaw.

## Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Gemini, Groq, Ollama
- **Tool execution loop** — bounded multi-round tool dispatch (up to 12 rounds per request)
- **30+ built-in skills** — email, Slack, GitHub, PDF, web scraping, image generation, Spotify, Trello, Notion, and more
- **Persistent memory** — key-value memory with BM25 recall, daily logs, and automatic fact extraction during conversation compaction
- **Hybrid RAG** — BM25 + Chroma vector retrieval with FlashRank reranking for knowledge-grounded responses
- **Cron scheduler** — static (YAML) and dynamic (agent-created) recurring jobs with Telegram delivery
- **Heartbeat monitor** — LLM health probing with alert-on-transition (no alert storms)
- **Voice input** — Deepgram-powered speech-to-text
- **Session persistence** — conversation history survives restarts
- **Per-group isolation** — each session gets its own context directory (soul, persona, memory, skills)
- **Dynamic skill creation** — the agent can create new skills at runtime ("God Mode")

## Quick Start

### Prerequisites

- Python 3.11+
- A Telegram bot token (for Telegram mode) or just a terminal (for REPL mode)
- At least one LLM provider API key

### Installation

```bash
git clone https://github.com/your-org/langclaw.git
cd langclaw
python -m venv .venv
source .venv/bin/activate
pip install -r req.txt
```

### Configuration

Create `~/.langclaw/langclaw.json` (or `langclaw.json` in the project root):

```json5
{
  // LLM provider and model
  "llm": {
    "provider": "groq",
    "model": "openai/gpt-oss-120b"
  },

  // Telegram bot token
  "channels": {
    "telegram": {
      "token": "YOUR_TELEGRAM_BOT_TOKEN",
      "requireMention": false
    }
  },

  // Optional: restrict to specific Telegram user IDs
  "access": {
    "allowedUsers": [123456789]
  }
}
```

Or use environment variables in `.env`:

```bash
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
TELEGRAM_BOT_TOKEN=your_token_here

# Optional: per-provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...    # enables web search
DEEPGRAM_API_KEY=dg-...    # enables voice input
```

### Run

```bash
# Telegram bot mode (default)
python main.py

# Telegram without cron/heartbeat
python main.py --no-cron --no-heartbeat

# Interactive REPL mode (no Telegram needed)
python main.py --repl
```

## Architecture

```
langclaw/
├── main.py                    # Entry point (Telegram bot or REPL)
├── config.py                  # JSON5 config + env var overrides
├── session_manager.py         # Session registry with concurrency control
│
├── core/
│   ├── agent.py               # The Agent — brain of the system
│   ├── compaction.py          # Conversation summarization + memory flush
│   ├── session_store.py       # Persist conversation history to disk
│   ├── skill_loader.py        # 3-tier skill system (discover/load/resources)
│   ├── stt.py                 # Speech-to-text (Deepgram)
│   │
│   ├── llm/                   # Multi-provider LLM abstraction
│   │   ├── config.py          # LLMConfig dataclass
│   │   ├── factory.py         # get_llm() — builds LangChain chat models
│   │   └── streaming.py       # Sync/async streaming helpers
│   │
│   ├── memory/                # Persistent memory system
│   │   ├── manager.py         # MemoryManager (BM25 recall, boot context)
│   │   └── storage.py         # MemoryStorage (markdown-backed key-value)
│   │
│   ├── tool/                  # Tool implementations
│   │   ├── tools.py           # Shell, file I/O, web search, skill creation
│   │   └── langtools.py       # LangChain @tool wrappers
│   │
│   └── RAG/                   # Retrieval-Augmented Generation
│       ├── rag.py             # KnowledgeRAG facade
│       ├── retriever.py       # Hybrid BM25 + Chroma retriever
│       ├── reranker.py        # FlashRank reranker (offline)
│       ├── chunker.py         # Text chunking
│       ├── BM25.py            # Persistent BM25 store
│       └── ingestion.py       # Dual-store ingestion
│
├── channels/
│   └── telegram_bot.py        # Telegram I/O layer
│
├── scheduler/
│   ├── cron.py                # APScheduler cron (static + dynamic jobs)
│   └── heartbeat.py           # LLM health monitor
│
├── templates/                 # Default identity + built-in skills
│   ├── soul/SOUL.md           # Core identity (values, ethics)
│   ├── persona/               # Role/style templates
│   ├── tools/TOOLS.md         # Environment notes template
│   └── skills/                # ~30 built-in skills
│
└── data/                      # RAG data stores (Chroma, BM25)
```

## How It Works

### Agent Loop

1. User sends a message (via Telegram or REPL)
2. System prompt is assembled from: **Soul** + **Persona** + **Tools Notes** + **Skill Catalog** + **Memory Context**
3. LLM generates a response — if it includes tool calls, they are executed and results fed back
4. This repeats for up to 12 tool rounds until the LLM produces a final text answer
5. Messages are persisted to disk; old messages are compacted with fact extraction

### Three-Tier Identity System

| Layer | Purpose | Location |
|-------|---------|----------|
| **Soul** | Immutable core values and ethics | `context/soul/` |
| **Persona** | Role, style, specialization | `context/persona/` |
| **Tools Notes** | Environment-specific info (SSH hosts, paths) | `context/tools/` |

The soul cannot be overridden by persona files, skills, or user instructions. New users are guided through onboarding on first launch.

### Skills System

Skills are self-contained capabilities with YAML metadata and markdown instructions:

```
templates/skills/
├── communication/  email, slack
├── data/           csv_analyzer, finance, news, pdf_*, scraper, weather, youtube
├── dev/            code_runner, github, http_request
├── google/         workspace
├── media/          image_gen, spotify, tts
├── meta/           skill_creator
├── productivity/   notion, obsidian, trello
├── system/         change_persona, change_soul, onboarding, session_logs, time
├── text/           translator
└── web/            summarize, tavily
```

Skills are loaded on-demand — the agent sees a compact catalog in its system prompt and loads full instructions only when needed.

### Memory

- **MEMORY.md** — curated long-term key-value store
- **Daily logs** — append-only `YYYY-MM-DD.md` files
- **INDEX.md** — curated system info
- **Boot context** — automatically injected at session start (profile keys first, budget-limited)
- **Memory flush** — before old messages are discarded during compaction, the LLM extracts key facts into long-term memory

### Context Management

- **Auto-compaction** — when token count (system prompt + history) exceeds a threshold, old messages are summarized
- **Session persistence** — conversation history is saved to `~/.langclaw/context/sessions/` and restored on restart
- **Per-group isolation** — each session gets its own context directory with separate soul, persona, memory, and skills

## Configuration Reference

All config values can be set in `langclaw.json` or overridden with environment variables.

| Config Path | Env Var | Default | Description |
|---|---|---|---|
| `llm.provider` | `LLM_PROVIDER` | `groq` | LLM provider (`openai`, `gemini`, `anthropic`, `ollama`, `groq`) |
| `llm.model` | `LLM_MODEL` | `openai/gpt-oss-120b` | Model identifier |
| `llm.temperature` | — | `0.7` | Generation temperature |
| `llm.maxTokens` | — | `2048` | Max output tokens |
| `channels.telegram.token` | `TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `channels.telegram.requireMention` | — | `false` | Require @mention in groups |
| `access.allowedUsers` | `TELEGRAM_ALLOWED_USERS` | — | Comma-separated user IDs |
| `isolation.perGroup` | — | `false` | Per-session context isolation |
| `concurrency.maxAgents` | — | `4` | Max concurrent agent executions |
| `heartbeat.intervalSec` | `HEARTBEAT_INTERVAL_SEC` | `60` | Heartbeat probe interval |
| `deepgram.language` | — | `multi` | STT language (`multi`, `zh`, `en`, `ja`, `auto`) |
| `agent.verbose` | — | `false` | Verbose agent logging |
| `logging.level` | — | `INFO` | Log level |

## Cron Jobs

Define recurring tasks in `~/.langclaw/context/cron/jobs.yaml`:

```yaml
- id: daily_summary
  cron: "0 9 * * *"
  prompt: "Summarize yesterday's key events and tasks."
  enabled: true
  deliver_to: telegram
  chat_id: 123456789
```

The agent can also create dynamic jobs at runtime via the `lc_cron_add` tool. Dynamic jobs are persisted to `dynamic_jobs.json` and survive restarts.

## Creating Custom Skills

Create a directory under `~/.langclaw/context/skills/` with a `SKILL.md`:

```markdown
---
name: my_skill
description: Does something useful
dependencies:
  - requests
---

# My Skill

Instructions for the agent on how to use this skill...
```

The agent can also create skills at runtime using the `create_skill` tool ("God Mode").

## Project Structure: Key Directories

| Path | Purpose |
|---|---|
| `~/.langclaw/` | Home directory |
| `~/.langclaw/langclaw.json` | Main config file |
| `~/.langclaw/context/soul/` | Soul identity files |
| `~/.langclaw/context/persona/` | Persona files |
| `~/.langclaw/context/memory/` | MEMORY.md + daily logs |
| `~/.langclaw/context/sessions/` | Persisted conversation history |
| `~/.langclaw/context/skills/` | Custom user skills |
| `~/.langclaw/context/knowledge/` | RAG knowledge base |
| `~/.langclaw/context/cron/` | Cron job definitions |
| `~/.langclaw/context/files/` | Shared working directory |
| `~/.langclaw/context/groups/` | Per-session isolated contexts |

## Telegram Commands

| Command | Description |
|---|---|
| `/start` | Start or restart the bot |
| `/reset` | Clear session and start fresh |
| `/status` | Show agent status (model, memory, skills) |
| `/compact` | Manually trigger conversation compaction |
| `/clear_files` | Delete all files in the shared files directory |

## Development

```bash
# Run in REPL mode for quick testing
python main.py --repl

# Run without cron/heartbeat
python main.py --no-cron --no-heartbeat

# Verbose mode (set in config or env)
AGENT_VERBOSE=1 python main.py --repl
```

## License

See [LICENSE](LICENSE) for details.