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
from pathlib import Path

from core.subagents.markdown_parser import parse_agents_directory
from core.subagents.model import AgentEntry, AgentStatus, SubAgentConfig
from core.subagents.subagent_factory import invoke_subagent

logger = logging.getLogger(__name__)


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