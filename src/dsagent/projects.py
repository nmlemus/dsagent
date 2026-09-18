"""Projects — the durable container a run and a session belong to.

A project is a workspace directory plus metadata. It is a generic harness
concept (invariant 1): a project has inputs and runs; that those inputs are
*datasets* is the cartridge's framing, not the harness's.

    <projects_root>/
      <project-id>/
        project.json      # {id, name, goal, cartridge, created_at}
        data/             # project inputs, reusable across sessions and runs
        runs/<run-id>/…   # runs scoped to the project

The `default` project is special: it adopts the server's legacy flat runs
directory in place, so an existing deployment (and every existing test) keeps
working — nothing is moved. Only newly created projects live fully under
`<projects_root>/<id>/`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_JSON = "project.json"
DEFAULT_ID = "default"

_ID_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Project:
    id: str
    name: str
    goal: str = ""
    cartridge: str = ""
    created_at: float = field(default_factory=time.time)

    def dict(self) -> dict:
        return asdict(self)


def _slug(name: str) -> str:
    s = _ID_RE.sub("-", name.strip().lower()).strip("-")
    return s or "project"


def _safe_id(value: str) -> bool:
    return bool(value) and "/" not in value and "\\" not in value and value not in (".", "..")


class ProjectStore:
    """Reads and writes projects under one root, with `default` adopting the
    legacy runs directory. Pure filesystem — no model, no network."""

    def __init__(self, root: Path, default_runs_dir: Path):
        self.root = root
        self.default_runs_dir = default_runs_dir

    # --- locations ---------------------------------------------------------

    def dir(self, project_id: str) -> Path:
        return self.root / project_id

    def runs_dir(self, project_id: str) -> Path:
        """Where a project's runs live. `default` keeps the legacy path in
        place; every other project namespaces under its own directory."""
        if project_id == DEFAULT_ID:
            return self.default_runs_dir
        return self.dir(project_id) / "runs"

    def data_dir(self, project_id: str) -> Path:
        return self.dir(project_id) / "data"

    def locate_run(self, run_id: str) -> Path | None:
        """Find a run wherever it lives — the default (legacy) dir first, then
        each project. This is the seam that makes every run-scoped route
        project-aware: they resolve a run by id without knowing its project."""
        if not _safe_id(run_id):
            return None
        default = self.default_runs_dir / run_id
        if (default / "run.json").is_file():
            return default
        if self.root.is_dir():
            for d in self.root.iterdir():
                if not d.is_dir() or d.name == DEFAULT_ID:
                    continue
                candidate = d / "runs" / run_id
                if (candidate / "run.json").is_file():
                    return candidate
        return None

    def project_of(self, run_dir: Path) -> str:
        """Which project a run directory belongs to (for display/aggregation)."""
        try:
            if run_dir.parent == self.default_runs_dir:
                return DEFAULT_ID
            if run_dir.parent.name == "runs" and run_dir.parent.parent.parent == self.root:
                return run_dir.parent.parent.name
        except (OSError, ValueError):
            pass
        return DEFAULT_ID

    # --- reads -------------------------------------------------------------

    def ensure_default(self) -> Project:
        """The default project always exists; its metadata is written lazily
        without touching the adopted runs directory."""
        meta = self.dir(DEFAULT_ID) / PROJECT_JSON
        if meta.is_file():
            return self._read(DEFAULT_ID) or self._default_project()
        proj = self._default_project()
        self._write(proj)
        return proj

    def _default_project(self) -> Project:
        return Project(id=DEFAULT_ID, name="Default", goal="", cartridge="", created_at=0.0)

    def get(self, project_id: str) -> Project | None:
        if not _safe_id(project_id):
            return None
        if project_id == DEFAULT_ID and not (self.dir(DEFAULT_ID) / PROJECT_JSON).is_file():
            return self._default_project()
        return self._read(project_id)

    def _read(self, project_id: str) -> Project | None:
        meta = self.dir(project_id) / PROJECT_JSON
        if not meta.is_file():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return Project(
            id=str(data.get("id") or project_id),
            name=str(data.get("name") or project_id),
            goal=str(data.get("goal") or ""),
            cartridge=str(data.get("cartridge") or ""),
            created_at=float(data.get("created_at") or 0.0),
        )

    def list(self) -> list[Project]:
        """Default first, then the rest newest-first. A directory without a
        readable `project.json` is skipped, not fatal."""
        projects = [self.ensure_default()]
        others: list[Project] = []
        if self.root.is_dir():
            for d in self.root.iterdir():
                if not d.is_dir() or d.name == DEFAULT_ID:
                    continue
                proj = self._read(d.name)
                if proj:
                    others.append(proj)
        others.sort(key=lambda p: p.created_at, reverse=True)
        return projects + others

    # --- writes ------------------------------------------------------------

    def create(self, name: str, *, goal: str = "", cartridge: str = "") -> Project:
        base = _slug(name)
        pid = base
        n = 2
        while (self.dir(pid) / PROJECT_JSON).is_file() or pid == DEFAULT_ID:
            pid = f"{base}-{n}"
            n += 1
        proj = Project(id=pid, name=name.strip() or pid, goal=goal, cartridge=cartridge)
        self.data_dir(pid).mkdir(parents=True, exist_ok=True)
        self.runs_dir(pid).mkdir(parents=True, exist_ok=True)
        self._write(proj)
        return proj

    def _write(self, proj: Project) -> None:
        meta = self.dir(proj.id) / PROJECT_JSON
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(proj.dict(), indent=2), encoding="utf-8")
