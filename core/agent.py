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
from core.tool.langtools import get_all_langchain_tools
from core.skill_loader import SkillRegistry
from core.tool.tools import (
    configure_venv,
    set_sandbox
)

import config as _cfg

from .compaction import (
    DEFAULT_AUTO_THRESHOLD_TOKENS,
    DEFAULT_RECENT_KEEP,
    estimate_tokens,
)

logger = logging.getLogger(__name__)

def _load_text_dir_or_file(path: str | None, label: str = "File") -> str:
    """
    Load text from a single file or from all .md/.txt files in a directory.
    Returns an empty string if *path* is None or does not exist.
    """
    if not path or not os.path.exists(path):
        return ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if os.path.isdir(path):
        parts = []
        for filename in sorted(os.listdir(path)):
            if filename.lower().endswith((".md", ".txt")):
                with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                    parts.append(f"\n\n--- {label}: {filename} ---\n" + f.read())
        return "".join(parts)
    return ""

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
        self,
        #provider:LLMProvider, LLMProvider is providing two apis chat and chat stream
        context_dir:str,
        session_id:str|None = None ,
        #memory_dir:str | None = None,
        #skills_dir :list[str]|None=None, 
        #knowledge_path:str|None=None,
        #persona_path:str|None=None,
        #soul_path:str|None=None,
        verbose:bool=False,
        show_full_context:bool=False,
        max_chat_history:int=10,
        auto_compaction:bool=True,
        compaction_threshold:int = DEFAULT_AUTO_THRESHOLD_TOKENS,
        compaction_recent_keep:int = DEFAULT_RECENT_KEEP,
        cron_manager = None,
    ):
        
        #self.provider = provider
        if session_id and _cfg.per_group_isolation():
            group_dir = str(_cfg.group_context_dir(session_id))
            os.makedirs(os.path.join(group_dir, "memory"), exist_ok=True)
            memory_dir = os.path.join(group_dir, "memory")
            if verbose:
                print(f"[Agent] Per-group memory: {memory_dir}")

        else:
            memory_dir = os.path.join(context_dir,"memory")

        knowledge_path = os.path.join(context_dir,"knowledge")
        skills_dirs = [os.path.join(context_dir, "skills")]
        persona_path = os.path.join(context_dir, "persona")
        soul_path = os.path.join(context_dir, "soul")
        tools_path = os.path.join(context_dir, "tools")

        self.llm = get_llm("groq","openai/gpt-oss-120b")
        self.session_id = session_id
        self.messages :list[dict]=[]
        self.verbose = verbose
        self.show_full_context = show_full_context
        self.max_chat_history = max_chat_history
        self.auto_compaction = auto_compaction
        self.compaction_threshold = compaction_threshold
        self.compaction_recent_keep = compaction_recent_keep
        self.compaction_count : int = 0
        self._cron_manager = cron_manager

        self.loaded_skill_names : set[str] = set()
        self.pending_injections : list[str] = []
        self.MAX_PARALLEL_SKILLS = _cfg.get_int(
            "agent","maxParallelSkills",default=5
        )
        self._bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-bg")

        mem_dir = memory_dir
        global_mem_dir : str | None = None
        if session_id and _cfg.per_group_isolation():
            global_mem_dir = os.path.join(str(_cfg.LANGCLAW_HOME), "context", "memory")
        self.memory = MemoryManager(mem_dir, global_memory_dir=global_mem_dir)

        self.rag: KnowledgeRAG | None = None
        if knowledge_path and os.path.exists(knowledge_path):
            self.rag = KnowledgeRAG(
                knowledge_dir=knowledge_path,
                #provider=provider,
                use_reranker=True,
            )
            if verbose:
                print(f"[Agent] KnowledgeRAG: '{knowledge_path}' ({len(self.rag)} chunks)")
        
        self._web_search_enabled = bool(
            _cfg.get("tavily", "apiKey", env="TAVILY_API_KEY")
        )
        if verbose and self._web_search_enabled:
            print("[Agent] Web search enabled (Tavily)")

        #Identity layer
        self.soul_instruction = _load_text_dir_or_file(soul_path, label="Soul")
        self.persona_instruction = _load_text_dir_or_file(persona_path, label="Persona")
        self.tools_notes = _load_text_dir_or_file(tools_path, label="Tools")


        # Skills — always include the built-in templates + user context/skills
        self.skills_dirs: list[str] = []
        pkg_templates = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "templates", "skills",
        )
        if os.path.isdir(pkg_templates):
            self.skills_dirs.append(pkg_templates)
        if skills_dirs:
            for d in ([skills_dirs] if isinstance(skills_dirs, str) else skills_dirs):
                if d not in self.skills_dirs:
                    self.skills_dirs.append(d)

        
        self._needs_onboarding = not self._has_user_identity(soul_path, persona_path)

        if verbose and self.soul_instruction:
            print(f"[Agent] Soul loaded ({len(self.soul_instruction)} chars)")
        if verbose and self.persona_instruction:
            print(f"[Agent] Persona loaded ({len(self.persona_instruction)} chars)")
        if verbose and self.tools_notes:
            print(f"[Agent] TOOLS.md loaded ({len(self.tools_notes)} chars)")
        if verbose and self._needs_onboarding:
            print("[Agent] No user identity found — onboarding will be triggered")

        self._init_system_prompt()

    @staticmethod
    def _has_user_identity(soul_path: str | None, persona_path: str | None) -> bool:
        """Return True if the user has customized soul or persona files."""
        for p in (soul_path, persona_path):
            if p is None:
                continue
            if os.path.isdir(p):
                for fname in os.listdir(p):
                    fpath = os.path.join(p, fname)
                    if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                        return True
            elif os.path.isfile(p) and os.path.getsize(p) > 0:
                return True
        return False
    

    # ── Initialisation ────────────────────────────────────────────────────────
    