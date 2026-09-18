"""Sessions — the unit of work that ties a chat to the runs it launched.

A session is client-owned (assistant-ui holds the messages, keyed by its
thread id). The server keeps only the *link*: each run records the session it
belongs to, so "the runs of this chat" and "the chat that started this run" are
both answerable across tabs, reloads and the CLI.

The link is a one-line marker file at the run's root — `<run_dir>/session` —
outside the served workspace. That keeps `RunState` and every existing test
untouched; the session id is just the AG-UI `thread_id` of the chat that ran it
(or whatever the launcher passes).
"""

from __future__ import annotations

import json
from pathlib import Path

MARKER = "session"


class SessionRegistry:
    """A tiny server-side map session_id → project, so a run the orchestrator
    starts in a chat lands in that chat's project. The frontend registers a
    conversation's project when it opens it; a session it never registered
    resolves to `default`. One small JSON file — single-user, best-effort."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def set(self, session_id: str, project: str) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        data = self._load()
        data[sid] = project or "default"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass

    def project_of(self, session_id: str) -> str:
        return self._load().get((session_id or "").strip(), "default")


def mark_session(run_dir: Path, session_id: str) -> None:
    """Record which session a run belongs to. Best-effort — a run without a
    session (a bare CLI run) simply has no marker."""
    sid = (session_id or "").strip()
    if not sid:
        return
    try:
        (run_dir / MARKER).write_text(sid, encoding="utf-8")
    except OSError:
        pass


def session_of(run_dir: Path) -> str:
    """The session a run belongs to, or "" if it belongs to none."""
    try:
        f = run_dir / MARKER
        return f.read_text(encoding="utf-8").strip() if f.is_file() else ""
    except OSError:
        return ""
