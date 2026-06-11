"""markdown_parser.py — Parse sub-agent definitions from Markdown files.

Each ``.md`` file in the agents directory describes exactly **one** sub-agent.
The expected file format is::

    # Agent Name
    <!-- or: name: research-agent  (front-matter style) -->

    ## Description
    A one-sentence summary of what this agent does.

    ## Prompt
    Full system prompt that will be injected into the sub-agent's
    isolated context window.

    ## Tools
    - web_search
    - read_file
    - python_repl

    <!-- All sections below are optional -->

    ## Status
    active

    ## Version
    1.2.0

    ## Tags
    - research
    - web

    ## Max Iterations
    15

    ## Metadata
    key: value
    another_key: another value

Parsing rules
-------------
* Section headers are **case-insensitive** and may include surrounding
  whitespace.
* ``## Tools`` accepts a bullet list (``- tool_name``) **or** a
  comma-separated inline list.
* ``## Metadata`` accepts ``key: value`` pairs, one per line.
* If ``## Status`` is absent → defaults to ``active``.
* If ``## Version`` is absent → defaults to ``1.0.0``.
* If ``## Tags`` is absent → defaults to ``[]``.
* If ``## Max Iterations`` is absent → defaults to ``10``.
* If the file contains a YAML-style front-matter block (between ``---``
  delimiters), those key-value pairs are merged into ``metadata`` and
  recognised fields (name, status, version, tags, max_iterations) are
  promoted automatically.
* The agent ``name`` is derived (in priority order) from:
  1. ``## Name`` section content,
  2. YAML front-matter ``name:`` key,
  3. The filename stem (spaces → hyphens, lowercased).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from core.subagents.model import AgentStatus, ParsedMarkdownAgent, SubAgentConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Matches ``## Section Title`` with optional leading/trailing whitespace
_SECTION_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)

# YAML-like front-matter block: content between the first two ``---`` fences
_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n\s*---\s*\n", re.DOTALL)

# A single front-matter key-value line: ``key: value``
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-]*):\s*(.*)$")

# Bullet list item: ``- item`` or ``* item``
_BULLET_RE = re.compile(r"^[\-\*]\s+(.+)$")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert arbitrary text to a valid agent name slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Extract YAML-like front-matter and return (kv_dict, remaining_body)."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    fm_block = match.group(1)
    body = raw[match.end():]
    kv: dict[str, str] = {}
    for line in fm_block.splitlines():
        m = _KV_RE.match(line.strip())
        if m:
            kv[m.group(1).lower()] = m.group(2).strip()
    return kv, body


def _split_into_sections(body: str) -> dict[str, str]:
    """Split markdown body into ``{section_title_lower: content}``."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))

    for i, match in enumerate(matches):
        title = match.group(1).lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        sections[title] = content

    return sections


def _parse_list_field(raw: str) -> list[str]:
    """Parse a markdown section that should be a list.

    Supports:
    * Bullet list  (``- item`` or ``* item``)
    * Comma-separated inline (``item1, item2``)
    * One item per line
    """
    items: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            items.append(bullet.group(1).strip())
        elif "," in line:
            items.extend(p.strip() for p in line.split(",") if p.strip())
        else:
            items.append(line)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _parse_metadata_field(raw: str) -> dict[str, str]:
    """Parse ``key: value`` pairs from the Metadata section."""
    result: dict[str, str] = {}
    for line in raw.splitlines():
        m = _KV_RE.match(line.strip())
        if m:
            result[m.group(1)] = m.group(2)
    return result


def _coerce_status(raw: str) -> AgentStatus:
    mapping = {s.value: s for s in AgentStatus}
    cleaned = raw.strip().lower()
    if cleaned not in mapping:
        logger.warning("Unknown status %r; defaulting to 'active'.", raw)
        return AgentStatus.ACTIVE
    return mapping[cleaned]


def _coerce_max_iterations(raw: str, default: int = 10) -> int:
    try:
        val = int(raw.strip())
        return max(1, min(val, 50))   # clamp to [1, 50]
    except (ValueError, TypeError):
        logger.warning("Could not parse max_iterations %r; using %d.", raw, default)
        return default


# ---------------------------------------------------------------------------
# Single-file parser
# ---------------------------------------------------------------------------


def parse_agent_file(path: Path) -> ParsedMarkdownAgent:
    """Parse a single markdown file and return a :class:`ParsedMarkdownAgent`.

    Args:
        path: Absolute or relative path to the ``.md`` file.

    Returns:
        A validated :class:`ParsedMarkdownAgent` instance.

    Raises:
        ValueError: If required sections (description, prompt) are missing.
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Agent definition file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(raw)
    sections = _split_into_sections(body)

    # ---- Resolve name (priority: ## Name > front-matter > filename) --------
    name_raw = (
        sections.get("name")
        or sections.get("agent name")
        or fm.get("name")
        or path.stem
    )
    name = _slugify(name_raw)

    # ---- Required fields ---------------------------------------------------
    description_raw = sections.get("description") or fm.get("description", "")
    if not description_raw.strip():
        raise ValueError(
            f"[{path.name}] Missing required '## Description' section."
        )

    prompt_raw = sections.get("prompt") or sections.get("system prompt") or fm.get("prompt", "")
    if not prompt_raw.strip():
        raise ValueError(
            f"[{path.name}] Missing required '## Prompt' (or '## System Prompt') section."
        )

    # ---- Optional fields ---------------------------------------------------
    tools_raw = sections.get("tools") or sections.get("allowed tools") or fm.get("tools", "")
    tools = _parse_list_field(tools_raw) if tools_raw else []

    status_raw = sections.get("status") or fm.get("status", "active")
    status = _coerce_status(status_raw)

    version = (
        sections.get("version", "").strip()
        or fm.get("version", "1.0.0")
    )
    if not version:
        version = "1.0.0"

    tags_raw = sections.get("tags") or fm.get("tags", "")
    tags = _parse_list_field(tags_raw) if tags_raw else []

    max_iter_raw = (
        sections.get("max iterations", "")
        or sections.get("max_iterations", "")
        or fm.get("max_iterations", "")
        or fm.get("max iterations", "")
    )
    max_iterations = _coerce_max_iterations(max_iter_raw) if max_iter_raw.strip() else 10

    # Merge front-matter remainder into metadata
    metadata_section = sections.get("metadata") or sections.get("meta", "")
    metadata = _parse_metadata_field(metadata_section)
    # Pull any unknown fm keys into metadata
    known_fm_keys = {"name", "description", "prompt", "tools", "status",
                     "version", "tags", "max_iterations", "max iterations"}
    for k, v in fm.items():
        if k not in known_fm_keys:
            metadata.setdefault(k, v)

    return ParsedMarkdownAgent(
        name=name,
        description=description_raw.strip(),
        prompt=prompt_raw.strip(),
        tools=tools,
        status=status,
        version=version,
        tags=tags,
        max_iterations=max_iterations,
        metadata=metadata,
        source_file=str(path.resolve()),
    )


# ---------------------------------------------------------------------------
# Directory-level parser
# ---------------------------------------------------------------------------


def parse_agents_directory(
    directory: str | Path,
    *,
    glob_pattern: str = "**/*.md",
    skip_errors: bool = True,
) -> list[SubAgentConfig]:
    """Parse all markdown agent definitions found in *directory*.

    Args:
        directory: Path to the folder containing ``.md`` agent files.
        glob_pattern: Glob pattern used to discover files (default ``**/*.md``).
        skip_errors: If ``True`` (default), log parse errors and continue;
            if ``False``, re-raise on the first failure.

    Returns:
        A list of validated :class:`SubAgentConfig` instances for every
        file that parsed successfully and whose status is not ``disabled``.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Agents directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    configs: list[SubAgentConfig] = []
    files = sorted(root.glob(glob_pattern))

    if not files:
        logger.warning("No markdown files found in %s with pattern %r", root, glob_pattern)
        return configs

    for md_file in files:
        try:
            parsed = parse_agent_file(md_file)
            if parsed.status == AgentStatus.DISABLED:
                logger.info("Skipping disabled agent in %s", md_file.name)
                continue
            config = parsed.to_config()
            configs.append(config)
            logger.debug("Loaded agent %r from %s", config.name, md_file.name)
        except Exception as exc:
            if skip_errors:
                logger.error("Failed to parse %s: %s", md_file.name, exc)
            else:
                raise

    logger.info(
        "Parsed %d agent(s) from %s (%d file(s) scanned)",
        len(configs),
        root,
        len(files),
    )
    return configs


# ---------------------------------------------------------------------------
# Iterator variant (memory-friendly for large directories)
# ---------------------------------------------------------------------------


def iter_agent_files(
    directory: str | Path,
    *,
    glob_pattern: str = "**/*.md",
    skip_errors: bool = True,
) -> Iterator[SubAgentConfig]:
    """Yield :class:`SubAgentConfig` objects one by one as files are parsed.

    Identical semantics to :func:`parse_agents_directory` but yields
    instead of collecting — useful when the directory is very large.
    """
    root = Path(directory)
    for md_file in sorted(root.glob(glob_pattern)):
        try:
            parsed = parse_agent_file(md_file)
            if parsed.status == AgentStatus.DISABLED:
                continue
            yield parsed.to_config()
        except Exception as exc:
            if skip_errors:
                logger.error("Failed to parse %s: %s", md_file.name, exc)
            else:
                raise


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys

    ap = argparse.ArgumentParser(description="Parse agent markdown files and dump JSON.")
    ap.add_argument("directory", help="Directory containing .md agent definitions.")
    ap.add_argument("--strict", action="store_true", help="Stop on first parse error.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        configs = parse_agents_directory(args.directory, skip_errors=not args.strict)
        print(json.dumps([c.model_dump() for c in configs], indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
