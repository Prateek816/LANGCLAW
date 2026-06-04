# Scheduler Module

A task scheduling and health monitoring system for LLM-powered applications. Provides cron-based job scheduling via YAML/JSON definitions and a heartbeat monitor that pings the LLM provider for uptime verification.

## Features

- **Cron-based scheduling** — define jobs in YAML or add them dynamically at runtime
- **Dynamic job management** — add/remove/reload jobs without restarting
- **Isolated sessions** — each cron job runs in its own session, maintaining context across runs
- **Heartbeat monitoring** — periodic LLM health checks with latency tracking
- **State-transition alerts** — Telegram notifications on failure/recovery (no alert storms)
- **Rotating logs** — heartbeat results logged to rotating files (1 MB, 3 backups)
- **Hot reload** — reload static job definitions without downtime

## Installation

```bash
pip install apscheduler pyyaml langchain-groq
```

## Quick Start

### Cron Scheduler

```python
from scheduler.cron import CronScheduler
from session_manager import SessionManager

# Initialize with a session manager
cron = CronScheduler(session_manager=SessionManager())

# Add a dynamic job at runtime
cron.add_dynamic_job(
    job_id="daily_report",
    cron_expr="0 9 * * *",        # every day at 9 AM
    prompt="Generate the daily report",
)

# Start the scheduler
cron.start()
```

### Heartbeat Monitor

```python
from scheduler.heartbeat import create_heartbeat

# Create from config (reads langclaw.json / env vars)
monitor = create_heartbeat()
monitor.start()

# Or configure manually
from scheduler.heartbeat import HeartbeatMonitor

monitor = HeartbeatMonitor(interval_sec=120)
monitor.start()
```

## Configuration

### Environment Variables

| Variable | Default | Used By | Description |
|----------|---------|---------|-------------|
| `LANGCLAW_HOME` | `~/.langclaw` | Both | Root directory for config and data |
| `HEARTBEAT_INTERVAL_SEC` | `60` | Heartbeat | Seconds between health probes |
| `HEARTBEAT_ALERT_CHAT_ID` | (none) | Heartbeat | Telegram chat ID for alerts |
| `GROQ_API_KEY` | (required) | Heartbeat | API key for LLM health probes |

### Config File (`langclaw.json`)

```json
{
  "heartbeat": {
    "intervalSec": 60,
    "alertChatId": 12345
  }
}
```

Searched at `~/.langclaw/langclaw.json`, then `./langclaw.json`.

### YAML Job Definitions

Create `<LANGCLAW_HOME>/context/cron/jobs.yaml`:

```yaml
jobs:
  - id: daily_report
    enabled: true
    cron: "0 9 * * *"
    prompt: "Generate the daily report"
    deliver_to: "telegram"
    chat_id: 12345

  - id: health_check
    enabled: true
    cron: "*/30 * * * *"
    prompt: "Check system status and report any issues"
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `id` | yes | — | Unique job identifier |
| `cron` | yes | — | 5-field cron expression (min hour dom month dow) |
| `prompt` | yes | — | LLM prompt to execute |
| `enabled` | no | `true` | Whether the job is active |
| `deliver_to` | no | — | Channel to deliver results (e.g., `"telegram"`) |
| `chat_id` | no | — | Chat ID for delivery |

## API Reference

### `CronScheduler`

Cron-based job scheduler using APScheduler.

```python
CronScheduler(
    session_manager: SessionManager,       # Required
    jobs_path: str | None = None,          # Default: <LANGCLAW_HOME>/context/cron/jobs.yaml
    telegram_bot: TelegramBot | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `None` | Load jobs (static + dynamic) and start the scheduler |
| `stop()` | `None` | Shut down the scheduler |
| `add_dynamic_job(job_id, cron_expr, prompt, deliver_to, chat_id)` | `str` | Add a job at runtime; returns status message |
| `remove_dynamic_job(job_id)` | `str` | Remove a dynamic job; returns status message |
| `list_jobs()` | `str` | Human-readable list of all active jobs with next run times |
| `reload_jobs()` | `int` | Hot-reload static jobs from YAML; returns count |

### `HeartbeatMonitor`

Periodic LLM health check with state-transition alerting.

```python
HeartbeatMonitor(
    interval_sec: int = 60,
    telegram_bot: TelegramBot | None = None,
    alert_chat_id: int | None = None,
    log_path: str | None = None,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `None` | Begin periodic health probes in background |
| `stop()` | `None` | Stop probing |

### `create_heartbeat(telegram_bot=None) -> HeartbeatMonitor`

Factory function that reads config from `langclaw.json` / env vars and returns a configured `HeartbeatMonitor`.

## How It Works

### Cron Job Execution Flow

```
jobs.yaml / dynamic_jobs.json
         |
         v
    CronScheduler.start()
         |
         v
    APScheduler (AsyncIOScheduler)
         |
         v  (on cron trigger)
    SessionManager.get_or_create("cron:{job_id}")
         |
         v
    agent.chat(prompt)
         |
         v
    Optional: TelegramBot.deliver(result)
```

Each job runs in its own session (`"cron:{job_id}"`), so conversation history is preserved across runs of the same job.

### Heartbeat Probe Flow

```
HeartbeatMonitor.start()
         |
         v  (every interval_sec)
    get_llm("groq", "openai/gpt-oss-120b")
         |
         v
    llm.invoke([{"role": "user", "content": "ping"}])
         |
         v
    Log result (rotating file)
         |
         v  (on state transition)
    TelegramBot.alert() — failure or recovery
```

State-transition logic prevents alert storms:
- First failure → sends failure alert
- Subsequent failures → no alert
- Recovery → sends recovery alert
- Normal operation → no alert

## File Overview

| File | Purpose |
|------|---------|
| `cron.py` | `CronScheduler` — YAML/JSON-based cron job scheduling with dynamic job management |
| `heartbeat.py` | `HeartbeatMonitor` — LLM uptime monitoring with rotating logs and Telegram alerts |

## Filesystem Layout

```
<LANGCLAW_HOME>/context/
├── cron/
│   ├── jobs.yaml            # Static job definitions
│   └── dynamic_jobs.json    # Persisted dynamic jobs
└── logs/
    └── heartbeat.log        # Rotating heartbeat log
```

## Integration with Agent

The `CronScheduler` is designed to be passed to the `Agent` class via the `cron_manager` parameter. The agent's cron tools (`cron_add`, `cron_remove`, `cron_list`) call the scheduler's methods, allowing the LLM to manage its own scheduled tasks.

```python
from scheduler.cron import CronScheduler

cron = CronScheduler(session_manager=session_manager)
cron.start()

# Pass to agent
agent = Agent(cron_manager=cron, ...)
```