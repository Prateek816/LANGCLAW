"""
cli.py — Rich-powered REPL components for LangClaw.
"""

from __future__ import annotations

import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.columns import Columns
from rich.rule import Rule
from rich import box
from rich.table import Table

console = Console()

# ── Palette ───────────────────────────────────────────────────────────────────
_ACCENT   = "bright_cyan"
_DIM      = "grey50"
_MUTED    = "grey70"
_OK       = "spring_green3"
_WARN     = "dark_orange"
_ERR      = "red1"
_TITLE    = "bright_white"
_LOGO_COL = "bright_cyan"

# ── Glyphs ────────────────────────────────────────────────────────────────────
_TICK  = "●"
_CROSS = "○"
_ARROW = "›"
_SEP   = "│"
_DOT   = "·"


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _status_glyph(enabled: bool) -> tuple[str, str]:
    """Return (glyph, style) depending on enabled state."""
    return (_TICK, _OK) if enabled else (_CROSS, _ERR)


def _kv(label: str, value: str, value_style: str = _TITLE) -> Text:
    """Single key–value line."""
    t = Text()
    t.append(f"  {_ARROW} ", style=_DIM)
    t.append(f"{label:<20}", style=_MUTED)
    t.append(value, style=value_style)
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────────────────────

_LOGO = r"""
  ██╗      █████╗ ███╗  ██╗ ██████╗  ██████╗██╗      █████╗ ██╗    ██╗
  ██║     ██╔══██╗████╗ ██║██╔════╝ ██╔════╝██║     ██╔══██╗██║    ██║
  ██║     ███████║██╔██╗██║██║  ███╗██║     ██║     ███████║██║ █╗ ██║
  ██║     ██╔══██║██║╚████║██║   ██║██║     ██║     ██╔══██║██║███╗██║
  ███████╗██║  ██║██║  ╚███║╚██████╔╝╚██████╗███████╗██║  ██║╚███╔███╔╝
  ╚══════╝╚═╝  ╚═╝╚═╝   ╚══╝ ╚═════╝  ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
"""


def print_banner(
    agent_name: str,
    provider: str,
    model: str,
    session_id: str,
    max_history: int,
    auto_compaction: bool,
    compaction_threshold: int,
    cron_running: bool,
    tracing_enabled: bool,
) -> None:
    """Print startup banner with agent info."""
    console.print()

    # ── Logo ──────────────────────────────────────────────────────────────────
    logo = Text(_LOGO, style=f"bold {_LOGO_COL}")
    console.print(logo)

    tagline = Text()
    tagline.append(f"  {_DOT * 3}  ", style=_DIM)
    tagline.append("multi-provider  LLM  orchestration  shell", style=_MUTED)
    tagline.append(f"  {_DOT * 3}", style=_DIM)
    console.print(tagline, justify="left")
    console.print()

    # ── Info table ────────────────────────────────────────────────────────────
    table = Table(
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 1),
        show_edge=False,
        expand=False,
    )
    table.add_column(style=_DIM,   no_wrap=True)   # label
    table.add_column(style=_MUTED, no_wrap=True)   # separator
    table.add_column(no_wrap=True)                 # value

    def row(label: str, value: Text | str) -> None:
        table.add_row(f"  {label}", _SEP, value)

    # Agent / Model
    row("agent",   Text(agent_name, style=f"bold {_TITLE}"))

    model_cell = Text()
    model_cell.append(provider, style=f"bold {_ACCENT}")
    model_cell.append("  /  ", style=_DIM)
    model_cell.append(model, style=f"bold {_ACCENT}")
    row("model",   model_cell)

    row("session", Text(session_id, style=_MUTED))
    row("history", Text(f"{max_history} messages", style=_TITLE))

    # Compaction
    g, gs = _status_glyph(auto_compaction)
    compact_cell = Text()
    compact_cell.append(g, style=gs)
    if auto_compaction:
        compact_cell.append(f"  enabled", style=_OK)
        compact_cell.append(f"  [{compaction_threshold:,} tokens]", style=_DIM)
    else:
        compact_cell.append(f"  disabled", style=_ERR)
    row("compaction", compact_cell)

    # Cron
    g, gs = _status_glyph(cron_running)
    cron_cell = Text()
    cron_cell.append(g, style=gs)
    cron_cell.append("  running" if cron_running else "  disabled",
                     style=_OK if cron_running else _ERR)
    row("cron", cron_cell)

    # Tracing
    g, gs = _status_glyph(tracing_enabled)
    trace_cell = Text()
    trace_cell.append(g, style=gs)
    if tracing_enabled:
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        trace_cell.append("  LangSmith", style=_OK)
        trace_cell.append(f"  [{project}]", style=_DIM)
    else:
        trace_cell.append("  disabled", style=_ERR)
    row("tracing", trace_cell)

    # Wrap in panel
    panel = Panel(
        table,
        border_style=_ACCENT,
        box=box.MINIMAL_DOUBLE_HEAD,
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)
    console.print()

    # ── Footer hint ───────────────────────────────────────────────────────────
    hint = Text()
    hint.append("  type ", style=_DIM)
    hint.append("quit", style=f"bold {_ACCENT}")
    hint.append(" or ", style=_DIM)
    hint.append("exit", style=f"bold {_ACCENT}")
    hint.append(" to terminate the session", style=_DIM)
    console.print(hint)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  REPL I/O
# ─────────────────────────────────────────────────────────────────────────────

def print_agent_prompt() -> str:
    """Styled input prompt."""
    console.print(Rule(style=_DIM), end="")
    return Prompt.ask(
        f"\n  [{_ACCENT}]you[/{_ACCENT}]  [{_DIM}]{_ARROW}[/{_DIM}]"
    )


def print_agent_prefix() -> None:
    """Print agent label before streaming output begins."""
    label = Text()
    label.append("\n  agent  ", style=f"bold {_OK}")
    label.append(f"{_ARROW}  ", style=_DIM)
    console.print(label, end="")


def print_exit() -> None:
    """Styled farewell."""
    console.print()
    farewell = Text()
    farewell.append(f"  {_TICK}  ", style=_DIM)
    farewell.append("session terminated", style=_MUTED)
    farewell.append("  —  ", style=_DIM)
    farewell.append("see you next time", style=f"italic {_ACCENT}")
    console.print(farewell)
    console.print()


def print_error(msg: str) -> None:
    """Styled error message."""
    err = Text()
    err.append(f"  {_CROSS} ", style=_ERR)
    err.append("error  ", style=f"bold {_ERR}")
    err.append(_SEP + "  ", style=_DIM)
    err.append(msg, style=_MUTED)
    console.print()
    console.print(
        Panel(err, border_style=_ERR, box=box.MINIMAL, padding=(0, 1), expand=False)
    )
    console.print()