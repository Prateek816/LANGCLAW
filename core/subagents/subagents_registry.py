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