"""Sessions — a chat is a durable, server-owned object now, not localStorage.

A session holds its project, title and messages under
`<projects_root>/sessions/<id>.json`, keyed by the chat's thread id. The server
being the source of truth means a chat survives a reload, opens in a second tab,
maps a run started in it to the right project, and cannot be lost by clearing a
browser. The per-run link marker (`<run_dir>/session`) stays as it was.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MARKER = "session"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe(session_id: str) -> str:
    return _UNSAFE.sub("_", (session_id or "").strip())[:128]


def mark_session(run_dir: Path, session_id: str) -> None:
    """Record which session a run belongs to. Best-effort — a run with no session
    (a bare CLI run) simply has no marker."""
    sid = (session_id or "").strip()
    if not sid:
        return
    try:
        (run_dir / MARKER).write_text(sid, encoding="utf-8")
    except OSError:
        pass


def session_of(run_dir: Path) -> str:
    try:
        f = run_dir / MARKER
        return f.read_text(encoding="utf-8").strip() if f.is_file() else ""
    except OSError:
        return ""


class SessionStore:
    """Chat sessions on disk: one JSON per session, keyed by id. Messages are the
    AG-UI messages the client holds; stored verbatim and handed back so the thread
    resumes. Single-user, best-effort writes."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_safe(session_id)}.json"

    def get(self, session_id: str) -> dict[str, Any] | None:
        if not _safe(session_id):
            return None
        try:
            return json.loads(self._path(session_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def project_of(self, session_id: str) -> str:
        rec = self.get(session_id)
        return str(rec.get("project")) if rec and rec.get("project") else "default"

    def save(self, session_id: str, *, project: str, title: str, messages: list, updated_at: float) -> None:
        sid = _safe(session_id)
        if not sid:
            return
        rec = {
            "id": session_id,
            "project": project or "default",
            "title": title or "",
            "updated_at": updated_at,
            "messages": messages,
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._path(session_id).write_text(json.dumps(rec), encoding="utf-8")
        except OSError:
            pass

    def list_for(self, project: str) -> list[dict[str, Any]]:
        """Session headers (no messages) for a project, newest first."""
        out: list[dict[str, Any]] = []
        if self.root.is_dir():
            for f in self.root.glob("*.json"):
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if rec.get("project", "default") == project and rec.get("title"):
                    out.append(
                        {
                            "id": rec.get("id", f.stem),
                            "project": rec.get("project", "default"),
                            "title": rec.get("title", ""),
                            "updated_at": rec.get("updated_at", 0),
                        }
                    )
        out.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return out

    def delete(self, session_id: str) -> bool:
        p = self._path(session_id)
        if p.is_file():
            p.unlink()
            return True
        return False
