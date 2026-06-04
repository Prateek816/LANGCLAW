from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from core.RAG.rag import KnowledgeRAG
from core.llm.factory import get_llm
from core.memory.manager import MemoryManager
from core.skill_loader import SkillRegistry

import config as _cfg

from .compaction import (
    DEFAULT_AUTO_THRESHOLD_TOKENS,
    DEFAULT_RECENT_KEEP,
    compact,
    estimate_tokens,
    memory_flush,
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
        context_dir: str,
        session_id: str | None = None,
        verbose: bool = False,
        show_full_context: bool = False,
        max_chat_history: int = 10,
        auto_compaction: bool = True,
        compaction_threshold: int = DEFAULT_AUTO_THRESHOLD_TOKENS,
        compaction_recent_keep: int = DEFAULT_RECENT_KEEP,
        cron_manager=None,
        session_store=None,
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

        from core.llm.config import Provider
        provider: Provider = _cfg.get_str("llm", "provider") or "groq"  # type: ignore[assignment]
        model = _cfg.get_str("llm", "model") or "openai/gpt-oss-120b"
        self.llm = get_llm(provider, model)
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
        self._file_sender = None  # channel-specific file sender callback
        self._session_store = session_store

        # Restore persisted session history if available
        if session_store and session_id:
            saved = session_store.load(session_id)
            if saved:
                self.messages = saved
                if verbose:
                    print(f"[Agent] Restored {len(saved)} messages from session store")

        self.loaded_skill_names : set[str] = set()
        self.MAX_PARALLEL_SKILLS = _cfg.get_int(
            "agent","maxParallelSkills",default=5
        )

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

    # ── System prompt ────────────────────────────────────────────────────────

    def _init_system_prompt(self) -> None:
        """Build the system prompt from identity layers + skills + memory."""
        parts: list[str] = []

        # Soul (core identity)
        if self.soul_instruction:
            parts.append(self.soul_instruction.strip())

        # Persona (role / style)
        if self.persona_instruction:
            parts.append(self.persona_instruction.strip())

        # Tools notes (environment-specific info)
        if self.tools_notes:
            parts.append(f"## Your Local Environment\n\n{self.tools_notes.strip()}")

        # Skill catalog — compact listing of available skills
        self._skill_registry = SkillRegistry(skills_dirs=self.skills_dirs or None)
        catalog = self._skill_registry.build_catalog()
        if catalog:
            parts.append(f"## Available Skills\n\n{catalog}")

        # RAG knowledge hint
        if self.rag:
            parts.append(
                "You have access to a knowledge base. Use the retrieve_knowledge tool "
                "to search it when the user's question may be answered by stored documents."
            )

        # Web search hint
        if self._web_search_enabled:
            parts.append(
                "You have web search available via the web_search tool. "
                "Use it when you need current information."
            )

        # Memory boot context (curated long-term memories)
        boot_ctx = self.memory.boot_context(max_chars=3000)
        if boot_ctx:
            parts.append(f"## What You Remember\n\n{boot_ctx}")

        self._system_prompt = "\n\n---\n\n".join(parts)

        if self.verbose:
            print(f"[Agent] System prompt: {len(self._system_prompt)} chars")

    # ── Tools ────────────────────────────────────────────────────────────────

    def _build_tools(self) -> list:
        """Build the LangChain tool list with runtime bindings."""
        from langchain_core.tools import StructuredTool
        from core.tool.langtools import (
            primitive_tools,
            web_search_tool,
        )

        # Exclude lc_send_file from primitive_tools — we build it per-session below
        tools = [t for t in primitive_tools if t.name != "lc_send_file"]

        # send_file — bound to this agent's channel-specific file sender
        def _send_file(path: str, caption: str = "") -> str:
            """Send a file to the user via the active channel."""
            from core.tool.tools import send_file as _send_file_impl
            return _send_file_impl(path, caption, sender=self._file_sender)

        tools.append(StructuredTool.from_function(
            func=_send_file, name="lc_send_file",
            description="Send a file to the user via the active channel. Max 100 MB.",
        ))

        if self._web_search_enabled:
            tools.extend(web_search_tool)

        # Memory tools — bound to this agent's memory manager
        memory_defs = [
            ("lc_remember", "Store a fact in long-term memory."),
            ("lc_recall", "Search long-term memory for relevant facts."),
            ("lc_memory_get", "Read a specific memory file by path."),
            ("lc_memory_list_files", "List all memory files."),
            ("lc_forget", "Remove a memory entry by key."),
            ("lc_update_index", "Update the memory INDEX.md file."),
        ]
        for name, desc in memory_defs:
            handler = self._make_memory_handler(name)
            tools.append(StructuredTool.from_function(
                func=handler, name=name, description=desc,
            ))

        # Skill tools — bound to this agent's skill registry
        skill_defs = [
            ("lc_use_skill", "Load and use a skill by name."),
            ("lc_list_skill_resources", "List resource files for a skill."),
        ]
        for name, desc in skill_defs:
            handler = self._make_skill_handler(name)
            tools.append(StructuredTool.from_function(
                func=handler, name=name, description=desc,
            ))

        # create_skill — bound to this agent's skill registry for cache invalidation
        def _create_skill(name: str, description: str, instructions: str,
                          category: str = "", resources: dict | None = None,
                          dependencies: list | None = None) -> str:
            from core.tool.tools import create_skill as _create_skill_impl
            result = _create_skill_impl(name, description, instructions,
                                        category, resources, dependencies)
            self._skill_registry.invalidate()
            return result

        tools.append(StructuredTool.from_function(
            func=_create_skill, name="create_skill",
            description="Create a new skill at runtime (God Mode). Writes SKILL.md and resource files, installs pip dependencies.",
        ))

        # Cron tools — bound to this agent's cron manager
        if self._cron_manager:
            cron_defs = [
                ("lc_cron_add", "Schedule a recurring job."),
                ("lc_cron_remove", "Remove a scheduled job."),
                ("lc_cron_list", "List all scheduled jobs."),
            ]
            for name, desc in cron_defs:
                handler = self._make_cron_handler(name)
                tools.append(StructuredTool.from_function(
                    func=handler, name=name, description=desc,
                ))

        # Knowledge retrieval tool
        if self.rag:
            def retrieve_knowledge(query: str) -> str:
                """Search the knowledge base for relevant documents."""
                assert self.rag is not None
                results = self.rag.retrieve(query, top_k=5)
                if not results:
                    return "No relevant documents found."
                return "\n\n".join(
                    f"[{r.get('source', 'unknown')}]\n{r['content']}" for r in results
                )

            tools.append(StructuredTool.from_function(
                func=retrieve_knowledge, name="retrieve_knowledge",
                description="Search the knowledge base for relevant documents.",
            ))

        return tools

    def _make_memory_handler(self, tool_name: str):
        """Create a runtime handler for a memory tool."""
        def handle_remember(content: str, key: str = "") -> str:
            return self.memory.remember(content, key or None)

        def handle_recall(query: str = "*") -> str:
            return self.memory.recall(query)

        def handle_memory_get(path: str) -> str:
            return self.memory.memory_get(path)

        def handle_memory_list_files() -> str:
            files = self.memory.list_files()
            return "\n".join(files) if files else "No memory files."

        def handle_forget(key: str) -> str:
            return self.memory.forget(key)

        def handle_update_index(content: str) -> str:
            self.memory.write_index(content)
            return "INDEX.md updated."

        handlers = {
            "lc_remember": handle_remember,
            "lc_recall": handle_recall,
            "lc_memory_get": handle_memory_get,
            "lc_memory_list_files": handle_memory_list_files,
            "lc_forget": handle_forget,
            "lc_update_index": handle_update_index,
        }
        return handlers.get(tool_name, lambda **_: f"Unknown memory tool: {tool_name}")

    def _make_skill_handler(self, tool_name: str):
        """Create a runtime handler for a skill tool."""
        def handle_use_skill(skill_name: str) -> str:
            skill = self._skill_registry.load_skill(skill_name)
            if not skill:
                return f"Skill '{skill_name}' not found."
            return f"## {skill.name}\n\n{skill.instructions}"

        def handle_list_skill_resources(skill_name: str) -> str:
            resources = self._skill_registry.list_resources(skill_name)
            if not resources:
                return f"No resources found for '{skill_name}'."
            return "\n".join(resources)

        handlers = {
            "lc_use_skill": handle_use_skill,
            "lc_list_skill_resources": handle_list_skill_resources,
        }
        return handlers.get(tool_name, lambda **_: f"Unknown skill tool: {tool_name}")

    def _make_cron_handler(self, tool_name: str):
        """Create a runtime handler for a cron tool."""
        def handle_cron_add(prompt: str, cron_expr: str, job_id: str = "") -> str:
            if not self._cron_manager:
                return "Cron scheduler not available."
            jid = self._cron_manager.add_dynamic_job(
                job_id=job_id or None,
                prompt=prompt,
                cron_expr=cron_expr,
            )
            return f"Job '{jid}' scheduled."

        def handle_cron_remove(job_id: str) -> str:
            if not self._cron_manager:
                return "Cron scheduler not available."
            ok = self._cron_manager.remove_dynamic_job(job_id)
            return f"Job '{job_id}' removed." if ok else f"Job '{job_id}' not found."

        def handle_cron_list() -> str:
            if not self._cron_manager:
                return "Cron scheduler not available."
            jobs = self._cron_manager.list_jobs()
            if not jobs:
                return "No scheduled jobs."
            return json.dumps(jobs, indent=2)

        handlers = {
            "lc_cron_add": handle_cron_add,
            "lc_cron_remove": handle_cron_remove,
            "lc_cron_list": handle_cron_list,
        }
        return handlers.get(tool_name, lambda **_: f"Unknown cron tool: {tool_name}")

    # ── Message building ─────────────────────────────────────────────────────

    def _build_messages(self, user_input: str | list) -> list[BaseMessage]:
        """Build the full message list for the LLM call."""
        msgs: list[BaseMessage] = [SystemMessage(content=self._system_prompt)]

        # Add chat history
        for m in self.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))

        # Add current user input
        if isinstance(user_input, list):
            # Multimodal (e.g., image + text)
            msgs.append(HumanMessage(content=user_input))
        else:
            msgs.append(HumanMessage(content=user_input))

        return msgs

    # ── Auto-compaction check ────────────────────────────────────────────────

    def _maybe_compact(self) -> None:
        """Trigger auto-compaction if the conversation is too long."""
        if not self.auto_compaction:
            return
        history_tokens = estimate_tokens(self.messages)
        system_tokens = estimate_tokens([{"content": self._system_prompt}])
        tokens = history_tokens + system_tokens
        if tokens >= self.compaction_threshold:
            if self.verbose:
                print(f"[Agent] Auto-compaction triggered ({tokens} tokens, system={system_tokens})")
            self.compact()

    # ── Chat (non-streaming) ─────────────────────────────────────────────────

    def chat(self, user_input: str | list) -> str:
        """
        Send a message and return the full response.

        Parameters
        ----------
        user_input : str or list
            Text string, or a LangChain-style multimodal content list.

        Returns
        -------
        The assistant's response text.
        """
        self._maybe_compact()

        tools = self._build_tools()
        llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm

        msgs = self._build_messages(user_input)

        if self.show_full_context:
            for m in msgs:
                print(f"  [{m.type}] {str(m.content)[:200]}")

        # Tool dispatch loop
        for _ in range(self.MAX_TOOL_ROUNDS):
            response = llm_with_tools.invoke(msgs)
            msgs.append(response)

            if not response.tool_calls:
                # No tool calls — final answer
                answer = response.content or ""
                # Store in history
                self.messages.append({"role": "user", "content": str(user_input)})
                self.messages.append({"role": "assistant", "content": answer})
                self._trim_history()
                self._persist()
                return answer

            # Execute tool calls
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                if self.verbose:
                    print(f"  [Tool] {tool_name}({tool_args})")

                # Find and call the tool
                result = self._execute_tool(tools, tool_name, tool_args)
                msgs.append(ToolMessage(content=result, tool_call_id=tool_id))

        # Exhausted tool rounds
        answer = "I exceeded the maximum tool rounds. Please try a simpler request."
        self.messages.append({"role": "user", "content": str(user_input)})
        self.messages.append({"role": "assistant", "content": answer})
        self._persist()
        return answer

    # ── Chat (streaming) ─────────────────────────────────────────────────────

    def chat_stream(self, user_input: str | list, token_callback=None) -> str:
        """
        Send a message with streaming. Calls token_callback for each chunk.

        Parameters
        ----------
        user_input : str or list
            Text string, or a LangChain-style multimodal content list.
        token_callback : callable(str) or None
            Called with each text chunk as it arrives.

        Returns
        -------
        The complete assistant response text.
        """
        self._maybe_compact()

        tools = self._build_tools()
        llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm

        msgs = self._build_messages(user_input)

        if self.show_full_context:
            for m in msgs:
                print(f"  [{m.type}] {str(m.content)[:200]}")

        # Tool dispatch loop
        for _ in range(self.MAX_TOOL_ROUNDS):
            # Stream the response
            full_text = ""
            accumulated = None
            for chunk in llm_with_tools.stream(msgs):
                accumulated = chunk if accumulated is None else accumulated + chunk
                text = chunk.content or ""
                if isinstance(text, list):
                    text = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in text
                    )
                if text:
                    full_text += text
                    if token_callback:
                        token_callback(text)

            if accumulated is None:
                break

            # Convert accumulated chunk to AIMessage for the message list
            response = AIMessage(
                content=accumulated.content or "",
                tool_calls=accumulated.tool_calls if accumulated.tool_calls else [],
            )
            msgs.append(response)

            if not response.tool_calls:
                # Final answer
                self.messages.append({"role": "user", "content": str(user_input)})
                self.messages.append({"role": "assistant", "content": full_text})
                self._trim_history()
                self._persist()
                return full_text

            # Execute tool calls
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                if self.verbose:
                    print(f"  [Tool] {tool_name}({tool_args})")

                result = self._execute_tool(tools, tool_name, tool_args)
                msgs.append(ToolMessage(content=result, tool_call_id=tool_id))

        answer = "I exceeded the maximum tool rounds. Please try a simpler request."
        self.messages.append({"role": "user", "content": str(user_input)})
        self.messages.append({"role": "assistant", "content": answer})
        if token_callback:
            token_callback(answer)
        self._persist()
        return answer

    # ── Tool execution ───────────────────────────────────────────────────────

    # Default timeout for tool execution (seconds). run_command has its own 60s timeout.
    TOOL_TIMEOUT = 30

    def _execute_tool(self, tools: list, tool_name: str, tool_args: dict) -> str:
        """Find and execute a tool by name with timeout protection."""
        for t in tools:
            if t.name == tool_name:
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(t.invoke, tool_args)
                        result = future.result(timeout=self.TOOL_TIMEOUT)
                    return str(result) if not isinstance(result, str) else result
                except TimeoutError:
                    return f"Tool '{tool_name}' timed out after {self.TOOL_TIMEOUT}s"
                except Exception as exc:
                    return f"Tool error: {exc}"
        return f"Unknown tool: {tool_name}"

    # ── History management ───────────────────────────────────────────────────

    def _trim_history(self) -> None:
        """Keep only the most recent messages in the sliding window.

        Before dropping old messages, flush key facts to long-term memory
        so that information is not silently lost.
        """
        max_msgs = self.max_chat_history * 2  # user + assistant pairs
        if len(self.messages) > max_msgs:
            to_drop = self.messages[:-max_msgs]
            # Only flush if dropping meaningful amount (>2 messages)
            if len(to_drop) > 2:
                try:
                    memory_flush(to_drop, self.llm, self.memory)
                except Exception as exc:
                    logger.warning("[Agent] Memory flush during trim failed (non-fatal): %s", exc)
            self.messages = self.messages[-max_msgs:]

    def _persist(self) -> None:
        """Write current messages to session store (write-through)."""
        if self._session_store and self.session_id:
            try:
                self._session_store.save(self.session_id, self.messages)
            except Exception as exc:
                logger.warning("[Agent] Session persist failed (non-fatal): %s", exc)

    # ── Compaction ───────────────────────────────────────────────────────────

    def compact(self, instruction: str | None = None) -> str:
        """
        Compact conversation history — summarize old messages, optionally
        flush key facts to memory.

        Returns a summary of what was compacted.
        """
        if not self.messages:
            return "Nothing to compact — conversation is empty."

        old_count = len(self.messages)
        self.messages, summary = compact(
            self.messages,
            llm=self.llm,
            memory=self.memory,
            recent_keep=self.compaction_recent_keep,
            instruction=instruction,
        )
        self.compaction_count += 1
        new_count = len(self.messages)

        if self.verbose:
            print(f"[Agent] Compacted {old_count} → {new_count} messages")

        return f"Compacted {old_count - new_count} messages.\n\nSummary: {summary}"