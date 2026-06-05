"""
LangClaw — entry point.

Usage
-----
    python main.py                    # start Telegram bot
    python main.py --no-cron          # skip cron scheduler
    python main.py --no-heartbeat     # skip heartbeat monitor
    python main.py --repl             # interactive REPL mode (no Telegram)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import config
from session_manager import SessionManager
from core.agent import Agent
from core.session_store import SessionStore
from dotenv import load_dotenv
import os

load_dotenv()

def _setup_logging() -> None:
    from logging.handlers import RotatingFileHandler

    level = config.get_str("logging", "level") or "INFO"
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
    root.addHandler(console)

    # File handler — app.log with rotation (5 MB, 3 backups)
    file_handler = RotatingFileHandler(
        "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
    root.addHandler(file_handler)


_session_store: SessionStore | None = None


def _agent_factory(session_id: str) -> Agent:
    """Create an Agent for the given session."""
    context_dir = str(config.group_context_dir(session_id))
    return Agent(
        context_dir=context_dir,
        session_id=session_id,
        verbose=config.get_bool("agent", "verbose", default=False),
        session_store=_session_store,
    )


def run_telegram(sm: SessionManager, *, cron: bool, heartbeat: bool) -> None:
    """Start the Telegram bot with optional cron and heartbeat."""
    from channels.telegram_bot import create_bot

    bot = create_bot(sm)

    cron_scheduler = None
    if cron:
        try:
            from scheduler.cron import CronScheduler
            cron_scheduler = CronScheduler(sm, telegram_bot=bot)
            cron_scheduler.load_and_register_jobs()
            cron_scheduler.start()
            logging.getLogger(__name__).info("Cron scheduler started.")
        except Exception as exc:
            logging.getLogger(__name__).warning("Cron scheduler failed to start: %s", exc)

    hb = None
    if heartbeat:
        try:
            from scheduler.heartbeat import create_heartbeat
            hb = create_heartbeat(telegram_bot=bot)
            hb.start()
            logging.getLogger(__name__).info("Heartbeat monitor started.")
        except Exception as exc:
            logging.getLogger(__name__).warning("Heartbeat failed to start: %s", exc)

    try:
        bot.run_polling()
    finally:
        if cron_scheduler:
            cron_scheduler.stop()
        if hb:
            hb.stop()


def run_repl(sm: SessionManager) -> None:
    """Interactive REPL mode — chat directly in the terminal."""
    session_id = "cli"
    agent = sm.get_or_create(session_id)
    print("LangClaw REPL — type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye.")
            break

        def token_cb(chunk: str) -> None:
            print(chunk, end="", flush=True)

        print("Agent: ", end="", flush=True)
        try:
            response = agent.chat_stream(user_input, token_cb)
            print()  # newline after streaming
        except Exception as exc:
            print(f"\nError: {exc}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="LangClaw agent framework")
    parser.add_argument("--repl", action="store_true", help="Interactive REPL mode (no Telegram)")
    parser.add_argument("--no-cron", action="store_true", help="Disable cron scheduler")
    parser.add_argument("--no-heartbeat", action="store_true", help="Disable heartbeat monitor")
    args = parser.parse_args()

    _setup_logging()

    global _session_store
    _session_store = SessionStore()

    sm = SessionManager(store=_session_store)
    sm.set_factory(_agent_factory)

    if args.repl:
        run_repl(sm)
    else:
        run_telegram(sm, cron=not args.no_cron, heartbeat=not args.no_heartbeat)


if __name__ == "__main__":
    main()
