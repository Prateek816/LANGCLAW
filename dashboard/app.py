"""LangClaw Dashboard — FastAPI backend for the web UI.

Serves a single-page app at / and provides REST API at /api/*.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config as cfg

logger = logging.getLogger(__name__)

app = FastAPI(title="LangClaw Dashboard", version="1.0.0")

# ---------------------------------------------------------------------------
# Session manager (lazy-initialized)
# ---------------------------------------------------------------------------

_session_manager = None
_session_store = None


def _get_session_manager():
    global _session_manager, _session_store
    if _session_manager is None:
        from session_manager import SessionManager
        from core.session_store import SessionStore

        _session_store = SessionStore()
        _session_manager = SessionManager(store=_session_store)

        # Import main's factory or create our own
        from core.agent import Agent

        def _factory(session_id: str) -> Agent:
            context_dir = str(cfg.group_context_dir(session_id))
            return Agent(
                context_dir=context_dir,
                session_id=session_id,
                verbose=cfg.get_bool("agent", "verbose", default=False),
                session_store=_session_store,
            )

        _session_manager.set_factory(_factory)
    return _session_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(data: dict) -> None:
    """Atomic write to the config JSON file."""
    config_path = cfg.config_path()
    if config_path is None:
        config_path = cfg.LANGCLAW_HOME / "langclaw.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        dir=str(config_path.parent), suffix=".tmp", prefix="langclaw_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, config_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    cfg.load(force=True)


def _skills_dir() -> str:
    return os.path.join(str(cfg.home()), "context", "skills")


def _subagents_dir() -> str:
    return os.path.join(str(cfg.home()), "context", "subagents")


# ---------------------------------------------------------------------------
# Static serving
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_ui():
    return FileResponse(_STATIC_DIR / "index.html")


# Mount static assets (if any CSS/JS files are added later)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------


@app.get("/api/config")
async def get_config():
    cfg.load(force=True)
    return cfg.as_dict()


@app.get("/api/config/{section}")
async def get_config_section(section: str):
    cfg.load(force=True)
    data = cfg.as_dict()
    if section not in data:
        raise HTTPException(404, f"Section '{section}' not found")
    return {section: data[section]}


@app.put("/api/config/{section}")
async def update_config_section(section: str, request: Request):
    body = await request.json()
    cfg.load(force=True)
    data = cfg.as_dict()
    data[section] = body
    _write_config(data)
    return {"status": "ok", section: body}


# ---------------------------------------------------------------------------
# LLM API
# ---------------------------------------------------------------------------


@app.get("/api/llm/providers")
async def get_providers():
    from core.llm.factory import list_providers
    from core.llm.config import _PROVIDER_DEFAULTS

    return {
        "providers": list_providers(),
        "defaults": _PROVIDER_DEFAULTS,
    }


# ---------------------------------------------------------------------------
# MCP Servers API
# ---------------------------------------------------------------------------


@app.get("/api/mcp/servers")
async def get_mcp_servers():
    cfg.load(force=True)
    return cfg.get("mcp", "servers", default={})


@app.post("/api/mcp/servers")
async def add_mcp_server(request: Request):
    body = await request.json()
    name = body.pop("name", "").strip()
    if not name:
        raise HTTPException(400, "Server name required")

    cfg.load(force=True)
    data = cfg.as_dict()
    mcp = data.setdefault("mcp", {})
    servers = mcp.setdefault("servers", {})

    if name in servers:
        raise HTTPException(409, f"Server '{name}' already exists")

    server_cfg = {"transport": body.get("transport", "stdio")}
    if server_cfg["transport"] == "stdio":
        server_cfg["command"] = body.get("command", "")
        if body.get("args"):
            server_cfg["args"] = (
                body["args"] if isinstance(body["args"], list)
                else [a.strip() for a in str(body["args"]).split(",") if a.strip()]
            )
        if body.get("cwd"):
            server_cfg["cwd"] = body["cwd"]
    else:
        server_cfg["uri"] = body.get("uri", "")

    servers[name] = server_cfg
    _write_config(data)
    return {"status": "ok", "server": {name: server_cfg}}


@app.put("/api/mcp/servers/{name}")
async def update_mcp_server(name: str, request: Request):
    body = await request.json()

    cfg.load(force=True)
    data = cfg.as_dict()
    servers = data.get("mcp", {}).get("servers", {})

    if name not in servers:
        raise HTTPException(404, f"Server '{name}' not found")

    server_cfg = {"transport": body.get("transport", "stdio")}
    if server_cfg["transport"] == "stdio":
        server_cfg["command"] = body.get("command", "")
        if body.get("args"):
            server_cfg["args"] = (
                body["args"] if isinstance(body["args"], list)
                else [a.strip() for a in str(body["args"]).split(",") if a.strip()]
            )
        if body.get("cwd"):
            server_cfg["cwd"] = body["cwd"]
    else:
        server_cfg["uri"] = body.get("uri", "")

    servers[name] = server_cfg
    _write_config(data)
    return {"status": "ok", "server": {name: server_cfg}}


@app.delete("/api/mcp/servers/{name}")
async def delete_mcp_server(name: str):
    cfg.load(force=True)
    data = cfg.as_dict()
    servers = data.get("mcp", {}).get("servers", {})

    if name not in servers:
        raise HTTPException(404, f"Server '{name}' not found")

    del servers[name]
    _write_config(data)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Skills API
# ---------------------------------------------------------------------------


@app.get("/api/skills")
async def list_skills():
    from core.skill import SkillRegistry

    s_dir = _skills_dir()
    if not os.path.isdir(s_dir):
        return []
    registry = SkillRegistry(skills_dirs=[s_dir])
    skills = registry.discover()
    return [
        {
            "name": s.name,
            "description": s.description,
            "path": s.path,
            "category": s.category,
            "emoji": s.emoji,
            "dependencies": s.dependencies,
        }
        for s in skills
    ]


@app.get("/api/skills/{name}")
async def get_skill(name: str):
    from core.skill import SkillRegistry

    s_dir = _skills_dir()
    registry = SkillRegistry(skills_dirs=[s_dir])
    skill = registry.load_skill(name)
    if not skill:
        raise HTTPException(404, f"Skill '{name}' not found")
    return {
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
        "path": skill.metadata.path,
        "category": skill.metadata.category,
        "emoji": skill.metadata.emoji,
        "dependencies": skill.metadata.dependencies,
    }


@app.post("/api/skills")
async def create_skill(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Skill name required")

    description = body.get("description", "").strip()
    instructions = body.get("instructions", "").strip()
    category = body.get("category", "").strip()

    if category:
        skill_path = os.path.join(_skills_dir(), category, name)
    else:
        skill_path = os.path.join(_skills_dir(), name)

    os.makedirs(skill_path, exist_ok=True)

    md_path = os.path.join(skill_path, "SKILL.md")
    if os.path.exists(md_path):
        raise HTTPException(409, f"Skill '{name}' already exists")

    # Build SKILL.md with frontmatter
    lines = ["---"]
    lines.append(f"name: {name}")
    if description:
        lines.append(f"description: >")
        for desc_line in description.split("\n"):
            lines.append(f"  {desc_line}")
    lines.append("---")
    lines.append("")
    if instructions:
        lines.append(instructions)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"status": "ok", "name": name, "path": skill_path}


@app.put("/api/skills/{name}")
async def update_skill(name: str, request: Request):
    body = await request.json()
    instructions = body.get("instructions", "").strip()

    from core.skill import SkillRegistry

    s_dir = _skills_dir()
    registry = SkillRegistry(skills_dirs=[s_dir])
    skill = registry.load_skill(name)
    if not skill:
        raise HTTPException(404, f"Skill '{name}' not found")

    md_path = os.path.join(skill.metadata.path, "SKILL.md")

    # Rebuild with updated instructions, preserving frontmatter
    description = body.get("description", skill.description).strip()
    lines = ["---"]
    lines.append(f"name: {name}")
    if description:
        lines.append(f"description: >")
        for desc_line in description.split("\n"):
            lines.append(f"  {desc_line}")
    if skill.metadata.emoji:
        lines.append(f"metadata:")
        lines.append(f'  emoji: "{skill.metadata.emoji}"')
    if skill.metadata.dependencies:
        deps_str = ", ".join(skill.metadata.dependencies)
        lines.append(f"dependencies: {deps_str}")
    lines.append("---")
    lines.append("")
    lines.append(instructions)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"status": "ok", "name": name}


# ---------------------------------------------------------------------------
# Subagents API
# ---------------------------------------------------------------------------


@app.get("/api/subagents")
async def list_subagents():
    from core.subagents.markdown_parser import parse_agents_directory

    agents_dir = _subagents_dir()
    os.makedirs(agents_dir, exist_ok=True)
    try:
        configs = parse_agents_directory(agents_dir)
    except Exception:
        return []
    return [c.model_dump() for c in configs]


@app.get("/api/subagents/{name}")
async def get_subagent(name: str):
    agents_dir = _subagents_dir()
    # Find the .md file
    md_file = _find_subagent_file(agents_dir, name)
    if not md_file:
        raise HTTPException(404, f"Subagent '{name}' not found")
    content = md_file.read_text(encoding="utf-8")
    return {"name": name, "content": content, "path": str(md_file)}


@app.post("/api/subagents")
async def create_subagent(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Subagent name required")

    slug = re.sub(r"[^\w\-]", "-", name.lower().strip()).strip("-")
    agents_dir = _subagents_dir()
    os.makedirs(agents_dir, exist_ok=True)

    md_path = os.path.join(agents_dir, f"{slug}.md")
    if os.path.exists(md_path):
        raise HTTPException(409, f"Subagent '{name}' already exists")

    content = _build_subagent_markdown(name, body)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {"status": "ok", "name": slug, "path": md_path}


@app.put("/api/subagents/{name}")
async def update_subagent(name: str, request: Request):
    body = await request.json()
    agents_dir = _subagents_dir()

    md_file = _find_subagent_file(agents_dir, name)
    if not md_file:
        raise HTTPException(404, f"Subagent '{name}' not found")

    content = _build_subagent_markdown(name, body)
    md_file.write_text(content, encoding="utf-8")

    return {"status": "ok", "name": name}


@app.delete("/api/subagents/{name}")
async def delete_subagent(name: str):
    agents_dir = _subagents_dir()
    md_file = _find_subagent_file(agents_dir, name)
    if not md_file:
        raise HTTPException(404, f"Subagent '{name}' not found")

    md_file.unlink()
    return {"status": "ok"}


def _find_subagent_file(agents_dir: str, name: str) -> Path | None:
    """Find .md file for a subagent by name."""
    root = Path(agents_dir)
    if not root.exists():
        return None
    # Try direct filename match first
    direct = root / f"{name}.md"
    if direct.exists():
        return direct
    # Scan all .md files and parse names
    from core.subagents.markdown_parser import parse_agent_file

    for md_file in sorted(root.glob("**/*.md")):
        try:
            parsed = parse_agent_file(md_file)
            if parsed.name == name:
                return md_file
        except Exception:
            continue
    return None


def _build_subagent_markdown(name: str, body: dict) -> str:
    """Build a subagent .md file from form data."""
    lines = [f"# {name}", ""]

    description = body.get("description", "").strip()
    if description:
        lines.append("## Description")
        lines.append(description)
        lines.append("")

    prompt = body.get("prompt", "").strip()
    if prompt:
        lines.append("## Prompt")
        lines.append(prompt)
        lines.append("")

    tools = body.get("tools", [])
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]
    if tools:
        lines.append("## Tools")
        for tool in tools:
            lines.append(f"- {tool}")
        lines.append("")

    status = body.get("status", "active").strip()
    if status and status != "active":
        lines.append("## Status")
        lines.append(status)
        lines.append("")

    version = body.get("version", "1.0.0").strip()
    if version and version != "1.0.0":
        lines.append("## Version")
        lines.append(version)
        lines.append("")

    tags = body.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    if tags:
        lines.append("## Tags")
        for tag in tags:
            lines.append(f"- {tag}")
        lines.append("")

    max_iter = body.get("max_iterations", 10)
    if max_iter and int(max_iter) != 10:
        lines.append("## Max Iterations")
        lines.append(str(max_iter))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat API (Streaming SSE)
# ---------------------------------------------------------------------------


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "web:dashboard")

    if not message:
        raise HTTPException(400, "Message required")

    sm = _get_session_manager()

    async def event_stream():
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def token_callback(chunk: str):
            asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)

        def run_chat():
            try:
                agent = sm.get_or_create(session_id)
                agent.chat_stream(message, token_callback=token_callback)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put(f"[ERROR] {e}"), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        thread = threading.Thread(target=run_chat, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            yield f"data: {json.dumps({'token': chunk})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Sessions API
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
async def list_sessions():
    sm = _get_session_manager()
    active = sm.list_sessions()

    # Also list persisted session files
    sessions_dir = os.path.join(str(cfg.home()), "context", "sessions")
    persisted = []
    if os.path.isdir(sessions_dir):
        for f in os.listdir(sessions_dir):
            if f.endswith(".json"):
                persisted.append(f[:-5])  # strip .json

    return {"active": active, "persisted": persisted}


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    sm = _get_session_manager()
    sm.reset(session_id)
    return {"status": "ok", "session_id": session_id}


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    sessions_dir = os.path.join(str(cfg.home()), "context", "sessions")
    safe_id = re.sub(r"[^\w\-]", "_", session_id)
    session_file = os.path.join(sessions_dir, f"{safe_id}.json")

    if not os.path.exists(session_file):
        return {"messages": []}

    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
