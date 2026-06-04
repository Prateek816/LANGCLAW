"""
SessionStore — persist agent conversation history to disk.

Layout
------
  ~/.langclaw/context/sessions/<session_id>.json

Each file contains the agent's message list (list[dict] with role/content keys).
Write-through after each chat turn. Restored on session creation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Characters allowed in session IDs for safe filesystem paths
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-:.]")


def _sanitize_session_id(session_id: str) -> str:
    """Convert session_id to a safe filename component."""
    return _SAFE_ID_RE.sub("_", session_id)


class SessionStore:
    """Persist agent message history to JSON files on disk."""

    def __init__(self, store_dir: str | None = None) -> None:
        if store_dir is None:
            import config as _cfg
            store_dir = os.path.join(str(_cfg.LANGCLAW_HOME), "context", "sessions")
        self._store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def _path_for(self, session_id: str) -> str:
        safe = _sanitize_session_id(session_id)
        return os.path.join(self._store_dir, f"{safe}.json")

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist messages for a session (atomic write)."""
        path = self._path_for(session_id)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=1)
            os.rename(tmp, path)
        except OSError as exc:
            logger.error("[SessionStore] Failed to save session %s: %s", session_id, exc)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def load(self, session_id: str) -> list[dict[str, Any]] | None:
        """Load persisted messages for a session. Returns None if not found."""
        path = self._path_for(session_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("[SessionStore] Session %s: expected list, got %s", session_id, type(data).__name__)
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("[SessionStore] Failed to load session %s: %s", session_id, exc)
            return None

    def delete(self, session_id: str) -> None:
        """Delete persisted session data."""
        path = self._path_for(session_id)
        try:
            os.unlink(path)
            logger.debug("[SessionStore] Deleted session %s", session_id)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.error("[SessionStore] Failed to delete session %s: %s", session_id, exc)
