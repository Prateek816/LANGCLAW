from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime

from core.RAG.rag import KnowledgeRAG
from core.llm.factory import get_llm
from core.memory.manager import MemoryManager
from core.langtools import get_all_langchain_tools

logger = logging.getLogger(__name__)


class Agent:
    """
    Stateful LLM agent with tool use, three-tier skill loading, memory,
    and compaction.

    Parameters
    ----------
    provider           : LLM backend (DeepSeek, Grok, Claude, Gemini, …)
    session_id         : session identifier (enables per-group context isolation)
    memory_dir         : path to memory directory (auto-detected if None)
    skills_dirs        : list of skill directory paths
    knowledge_path     : path to knowledge directory for RAG
    persona_path       : path to persona .md file or directory
    soul_path          : path to SOUL.md file or directory
    verbose            : print debug info to stdout
    show_full_context  : print the full context window before each LLM call
    max_chat_history   : max non-system messages kept in the sliding window
    auto_compaction    : trigger compaction when token estimate exceeds threshold
    compaction_threshold : token threshold for auto-compaction
    compaction_recent_keep : number of recent messages kept verbatim after compaction
    cron_manager       : CronScheduler instance (enables cron_add/remove/list tools)
    """
    MAX_TOOL_ROUNDS = 12
    MAX_PARALLEL_SKILLS = 5

    def __init__(
