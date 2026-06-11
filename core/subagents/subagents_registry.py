"""
BaseRegistry — Auto-discovers and registers all subagents from
`core.subagents.agents`. Each agent module must expose:
    - call(prompt: str, context: str) -> str
    - NAME: str          (unique snake_case identifier)
    - DESCRIPTION: str   (used as the LangChain tool description)

Usage (in your main agent):
    from core.subagents.baseRegistry import BaseRegistry

    registry = BaseRegistry()          # auto-discovers on init
    tools = registry.list_agents()     # returns List[AgentEntry]
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# AgentEntry — what list_agents() hands back to the main agent
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentEntry:
    """
    Thin descriptor consumed by StructuredTool.from_function().

    Attributes
    ----------
    name        : unique identifier used as the LangChain tool name
    description : human-readable description forwarded to the LLM
    func        : the agent's call(prompt, context) callable
    module      : the backing module (useful for debugging / hot-reload)
    """
    name: str
    description: str
    func: Callable[[str, str], str]
    module: ModuleType = field(repr=False)


# ──────────────────────────────────────────────────────────────
# BaseRegistry
# ──────────────────────────────────────────────────────────────

class BaseRegistry:
    """
    Scans `core.subagents.agents` at instantiation time and builds an
    internal registry of every valid agent module found there.

    Discovery rules
    ---------------
    A module is registered if and only if it:
      1. lives directly inside `core.subagents.agents` (one level deep)
      2. exposes  `NAME: str`  and  `DESCRIPTION: str`  at module level
      3. exposes  `call(prompt: str, context: str) -> str`  at module level

    Any module that fails validation is skipped with a warning so a
    single broken agent never takes down the whole registry.
    """

    # The package that contains all agent modules.
    # Override this in a subclass if your layout differs.
    AGENTS_PACKAGE: str = "core.subagents.agents"

    def __init__(self) -> None:
        self._registry: dict[str, AgentEntry] = {}
        self._discover()

    # ── public API ──────────────────────────────────────────────

    def list_agents(self) -> list[AgentEntry]:
        """Return all registered agents as a list of AgentEntry objects."""
        return list(self._registry.values())

    def get(self, name: str) -> AgentEntry | None:
        """Return a single AgentEntry by name, or None if not found."""
        return self._registry.get(name)

    def register(self, entry: AgentEntry) -> None:
        """
        Manually register an AgentEntry (useful in tests or for
        programmatically-created agents that don't live in a file).
        """
        if entry.name in self._registry:
            logger.warning(
                "BaseRegistry: overwriting already-registered agent '%s'",
                entry.name,
            )
        self._registry[entry.name] = entry
        logger.debug("BaseRegistry: manually registered agent '%s'", entry.name)

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry by name."""
        if name in self._registry:
            del self._registry[name]
            logger.debug("BaseRegistry: unregistered agent '%s'", name)
        else:
            logger.warning(
                "BaseRegistry: tried to unregister unknown agent '%s'", name
            )

    def __repr__(self) -> str:
        names = list(self._registry.keys())
        return f"<BaseRegistry agents={names}>"

    # ── discovery internals ─────────────────────────────────────

    def _discover(self) -> None:
        """
        Walk every module in AGENTS_PACKAGE and attempt to register it.
        Imports are isolated so a broken module never crashes discovery.
        """
        try:
            package = importlib.import_module(self.AGENTS_PACKAGE)
        except ModuleNotFoundError as exc:
            logger.error(
                "BaseRegistry: could not import agents package '%s': %s",
                self.AGENTS_PACKAGE,
                exc,
            )
            return

        for module_info in pkgutil.iter_modules(package.__path__):
            full_name = f"{self.AGENTS_PACKAGE}.{module_info.name}"
            self._try_register_module(full_name)

        logger.info(
            "BaseRegistry: discovered %d agent(s) → %s",
            len(self._registry),
            list(self._registry.keys()),
        )

    def _try_register_module(self, full_module_name: str) -> None:
        """Import one module and register it if it passes validation."""
        try:
            module = importlib.import_module(full_module_name)
        except Exception as exc:                          # noqa: BLE001
            logger.warning(
                "BaseRegistry: failed to import '%s' — skipping. Error: %s",
                full_module_name,
                exc,
            )
            return

        entry = self._build_entry(module)
        if entry is None:
            return                                        # validation logged inside

        if entry.name in self._registry:
            logger.warning(
                "BaseRegistry: duplicate agent name '%s' in '%s' — skipping.",
                entry.name,
                full_module_name,
            )
            return

        self._registry[entry.name] = entry
        logger.debug("BaseRegistry: registered agent '%s'", entry.name)

    @staticmethod
    def _build_entry(module: ModuleType) -> AgentEntry | None:
        """
        Validate the module's public contract and return an AgentEntry,
        or None (with a warning) if anything is missing / wrong.
        """
        missing: list[str] = []

        name = getattr(module, "NAME", None)
        if not isinstance(name, str) or not name.strip():
            missing.append("NAME (str)")

        description = getattr(module, "DESCRIPTION", None)
        if not isinstance(description, str) or not description.strip():
            missing.append("DESCRIPTION (str)")

        call_fn = getattr(module, "call", None)
        if not callable(call_fn):
            missing.append("call (callable)")

        if missing:
            logger.warning(
                "BaseRegistry: module '%s' is missing %s — skipping.",
                module.__name__,
                ", ".join(missing),
            )
            return None

        # Wrap call() so the signature matches what StructuredTool expects:
        # func(prompt: str, context: str) -> str
        def _bound_call(prompt: str, context: str) -> str:
            return module.call(prompt=prompt, context=context)        # type: ignore[union-attr]

        # Give the wrapper a clean __name__ so LangChain picks it up correctly
        _bound_call.__name__ = name  # type: ignore[attr-defined]

        return AgentEntry(
            name=name,
            description=description,
            func=_bound_call,
            module=module,
        )
    
"""
Discovers sub-agents defined as Markdown files in a
given directory, builds them into :class:`AgentEntry` objects (the same
contract as :class:`BaseRegistry`), and exposes them via ``list_agents()``.

Usage
-----
    from core.subagents.customRegistry import CustomRegistry

    registry = CustomRegistry(agents_dir="core/subagents/custom_agents")
    entries  = registry.list_agents()

    # In your main agent:
    if self.allow_subagents:
        for agent in registry.list_agents():
            tools.append(StructuredTool.from_function(
                func=agent.func,
                name=agent.name,
                description=agent.description,
            ))

Markdown contract
-----------------
Each ``.md`` file must have at minimum:

    # Agent Name          ← becomes agent.name  (slugified)

    ## Description
    One-sentence summary forwarded to the orchestrator LLM.

    ## Prompt
    Full system prompt injected into the sub-agent's context window.

Optional sections: Tools, Status, Version, Tags, Max Iterations, Metadata.
See markdown_parser.py for the full spec.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable

from core.subagents.markdown_parser import parse_agents_directory
from core.subagents.model import AgentStatus, SubAgentConfig
from core.subagents.subagent_factory import invoke_subagent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AgentEntry — identical contract to BaseRegistry so the main agent sees a
# uniform interface regardless of whether an agent came from a .py or .md file.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentEntry:
    """
    Thin descriptor consumed by StructuredTool.from_function().

    Attributes
    ----------
    name        : unique identifier used as the LangChain tool name
    description : human-readable description forwarded to the LLM
    func        : the agent's call(prompt, context) callable
    module      : None for markdown agents (no backing Python module)
    """
    name: str
    description: str
    func: Callable[[str, str], str]
    module: ModuleType | None = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# CustomRegistry
# ─────────────────────────────────────────────────────────────────────────────

class CustomRegistry:
    """
    Scans a directory of Markdown agent definitions and builds an
    ``AgentEntry`` for each active agent found.

    Parameters
    ----------
    agents_dir  : Path (str or Path) to the folder containing ``.md`` files.
    glob_pattern: Glob pattern for discovery (default ``**/*.md``).
    skip_errors : When True, parse failures are logged and skipped rather
                  than raising. Defaults to True.

    Notes
    -----
    * Agents with ``## Status: disabled`` are silently excluded.
    * Discovery happens at ``__init__`` time. Call ``reload()`` to
      re-scan after adding or editing markdown files at runtime.
    """

    def __init__(
        self,
        agents_dir: str | Path,
        *,
        glob_pattern: str = "**/*.md",
        skip_errors: bool = True,
    ) -> None:
        self._agents_dir = Path(agents_dir)
        self._glob_pattern = glob_pattern
        self._skip_errors = skip_errors
        self._registry: dict[str, AgentEntry] = {}
        self._discover()

    # ── Public API ────────────────────────────────────────────────────────────

    def list_agents(self) -> list[AgentEntry]:
        """Return all registered active agents as AgentEntry objects."""
        return list(self._registry.values())

    def get(self, name: str) -> AgentEntry | None:
        """Return a single AgentEntry by name, or None if not found."""
        return self._registry.get(name)

    def reload(self) -> None:
        """Re-scan the agents directory. Useful during development."""
        logger.info("CustomRegistry: reloading from %s", self._agents_dir)
        self._registry.clear()
        self._discover()

    def __repr__(self) -> str:
        return (
            f"<CustomRegistry dir={self._agents_dir!r} "
            f"agents={list(self._registry.keys())}>"
        )

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _discover(self) -> None:
        """Parse all markdown files and populate the internal registry."""
        if not self._agents_dir.exists():
            logger.error(
                "CustomRegistry: agents directory does not exist: %s",
                self._agents_dir,
            )
            return

        configs: list[SubAgentConfig] = parse_agents_directory(
            self._agents_dir,
            glob_pattern=self._glob_pattern,
            skip_errors=self._skip_errors,
        )

        for config in configs:
            # parse_agents_directory already filters disabled agents,
            # but guard here too in case the caller passes raw configs.
            if config.status == AgentStatus.DISABLED:
                continue

            entry = self._build_entry(config)
            if entry.name in self._registry:
                logger.warning(
                    "CustomRegistry: duplicate agent name %r — keeping first, skipping %s",
                    entry.name,
                    config.source_file,
                )
                continue

            self._registry[entry.name] = entry
            logger.debug(
                "CustomRegistry: registered %r from %s",
                entry.name,
                config.source_file,
            )

        logger.info(
            "CustomRegistry: %d agent(s) ready → %s",
            len(self._registry),
            list(self._registry.keys()),
        )

    # ── Entry builder ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_entry(config: SubAgentConfig) -> AgentEntry:
        """
        Create an AgentEntry whose ``func`` closes over the config and
        delegates to ``invoke_subagent`` in subagent_factory.py.

        The closure captures ``config`` by reference to a local variable
        (``_cfg``) so that loop iteration doesn't overwrite it — the
        classic Python late-binding gotcha.
        """
        _cfg = config   # capture for closure

        def _call(prompt: str, context: str) -> str:
            return invoke_subagent(config=_cfg, prompt=prompt, context=context)

        # Give the wrapper a clean __name__ so StructuredTool uses it correctly.
        _call.__name__ = config.name

        return AgentEntry(
            name=config.name,
            description=config.description,
            func=_call,
            module=None,    # markdown agents have no backing Python module
        )