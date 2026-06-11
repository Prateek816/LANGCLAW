"""model.py — Schemas and data models for the deep-agent sub-agent system.

All TypedDicts, Pydantic models, and enums live here so that
task_tool.py, subagent_registry.py, and markdown_parser.py share
a single source of truth for types.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from types import ModuleType
from typing import Annotated, Any, Callable, Optional
from typing_extensions import NotRequired, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentStatus(str, enum.Enum):
    """Lifecycle status of a registered sub-agent."""

    ACTIVE = "active"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


class ToolAccessPolicy(str, enum.Enum):
    """How a sub-agent's tool access is resolved."""

    ALL = "all"          # Inherit every tool from the parent graph
    EXPLICIT = "explicit"  # Only the tools listed in `tools`
    NONE = "none"        # The agent receives no tools


# ---------------------------------------------------------------------------
# Core SubAgent config — mirrors what task_tool.py already expects
# (kept as TypedDict so existing code is not broken)
# ---------------------------------------------------------------------------


class SubAgent(TypedDict):
    """Minimal configuration required by _create_task_tool.

    This TypedDict is the contract between the registry and the tool factory.
    All fields beyond `name` / `description` / `prompt` are optional to
    stay backwards-compatible with hand-crafted agent definitions.
    """

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]   # If absent → all tools are granted


# ---------------------------------------------------------------------------
# Richer Pydantic model — used internally by the registry and parser
# ---------------------------------------------------------------------------


class SubAgentConfig(BaseModel):
    """Validated, fully-typed representation of a sub-agent definition.

    The markdown parser produces these; the registry stores and validates
    them before converting to the lighter SubAgent TypedDict when needed.
    """

    # --- Identity ----------------------------------------------------------
    name: str = Field(
        ...,
        description="Unique snake_case identifier, e.g. 'research-agent'.",
        pattern=r"^[a-z][a-z0-9-_]*$",
    )
    description: str = Field(
        ...,
        min_length=10,
        description="One-sentence summary shown to the orchestrator when it picks an agent.",
    )

    # --- Behaviour ---------------------------------------------------------
    prompt: str = Field(
        ...,
        min_length=20,
        description="Full system prompt injected into the sub-agent's isolated context.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit allow-list of tool names. "
            "Empty list means 'inherit all tools from the parent graph'."
        ),
    )
    tool_access_policy: ToolAccessPolicy = Field(
        default=ToolAccessPolicy.ALL,
        description="Resolved automatically from `tools` but can be overridden.",
    )

    # --- Meta --------------------------------------------------------------
    status: AgentStatus = Field(
        default=AgentStatus.ACTIVE,
        description="Only ACTIVE agents are registered in the live registry.",
    )
    version: str = Field(
        default="1.0.0",
        description="Semver string; useful when multiple versions of an agent coexist.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form labels for filtering (e.g. ['research', 'web']).",
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Absolute path to the markdown file that defined this agent, if any.",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Hard cap on how many LLM turns the sub-agent may take per invocation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs from the markdown front-matter.",
    )

    # --- Validators --------------------------------------------------------

    @field_validator("name")
    @classmethod
    def name_no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("Agent name must not contain spaces; use hyphens or underscores.")
        return v.lower()

    @field_validator("tools")
    @classmethod
    def deduplicate_tools(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in v:
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @model_validator(mode="after")
    def resolve_tool_access_policy(self) -> "SubAgentConfig":
        """Auto-derive policy from the tools list when not explicitly set."""
        if self.tools:
            self.tool_access_policy = ToolAccessPolicy.EXPLICIT
        else:
            self.tool_access_policy = ToolAccessPolicy.ALL
        return self

    # --- Conversion --------------------------------------------------------

    def to_subagent_dict(self) -> SubAgent:
        """Convert to the lightweight TypedDict consumed by _create_task_tool."""
        base: SubAgent = {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
        }
        if self.tools:
            base["tools"] = list(self.tools)
        return base

    def __repr__(self) -> str:
        return (
            f"<SubAgentConfig name={self.name!r} "
            f"status={self.status.value} "
            f"tools={self.tools or 'ALL'}>"
        )


# ---------------------------------------------------------------------------
# Registry-level models
# ---------------------------------------------------------------------------


class RegistryEntry(BaseModel):
    """A slot in the SubAgentRegistry — wraps a config with runtime state."""

    config: SubAgentConfig
    # populated after the LangChain agent object is built
    agent_instance: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    @property
    def is_active(self) -> bool:
        return self.config.status == AgentStatus.ACTIVE

    @property
    def name(self) -> str:
        return self.config.name


class RegistrySnapshot(BaseModel):
    """Serialisable snapshot of the entire registry (for debugging / persistence)."""

    entries: list[SubAgentConfig]
    total: int = 0
    active: int = 0

    @model_validator(mode="after")
    def compute_counts(self) -> "RegistrySnapshot":
        self.total = len(self.entries)
        self.active = sum(1 for e in self.entries if e.status == AgentStatus.ACTIVE)
        return self


# ---------------------------------------------------------------------------
# Markdown-parser output model
# ---------------------------------------------------------------------------


class ParsedMarkdownAgent(BaseModel):
    """Intermediate representation produced by the markdown parser.

    Fields map 1-to-1 with the supported markdown sections; the parser
    fills defaults for anything that is absent from the file.
    """

    # Required sections (parser raises if missing)
    name: str
    description: str
    prompt: str

    # Optional sections (sensible defaults supplied by parser)
    tools: list[str] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.ACTIVE
    version: str = "1.0.0"
    tags: list[str] = Field(default_factory=list)
    max_iterations: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Populated by the parser — not part of the markdown file itself
    source_file: Optional[str] = None

    def to_config(self) -> SubAgentConfig:
        """Promote to the validated SubAgentConfig."""
        return SubAgentConfig(**self.model_dump())


# ---------------------------------------------------------------------------
# AgentEntry — shared descriptor consumed by StructuredTool.from_function()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentEntry:
    """
    Thin descriptor returned by both BaseRegistry and CustomRegistry.
    The main agent converts these to StructuredTool instances.

    Attributes
    ----------
    name        : unique identifier used as the LangChain tool name
    description : human-readable description forwarded to the LLM
    func        : the agent's call(prompt, context) callable
    module      : the backing module (None for markdown agents)
    """
    name: str
    description: str
    func: Callable[[str, str], str]
    module: ModuleType | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Convenience re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "AgentEntry",
    "AgentStatus",
    "ToolAccessPolicy",
    "SubAgent",
    "SubAgentConfig",
    "RegistryEntry",
    "RegistrySnapshot",
    "ParsedMarkdownAgent",
]
