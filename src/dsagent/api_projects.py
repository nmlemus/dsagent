"""HTTP routes for projects. Kept separate from the runs API so the project
layer reads on its own. Pure filesystem via `ProjectStore` — no model."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from dsagent.datasets import delete_dataset, list_datasets, store_dataset
from dsagent.projects import ProjectStore
from dsagent.runs import list_runs, run_dirs


def add_projects_routes(app, store: ProjectStore, sessions: Any = None) -> None:
    def _summary(project_id: str) -> dict[str, Any]:
        proj = store.get(project_id)
        assert proj is not None
        return {**proj.dict(), "runs": len(run_dirs(store.runs_dir(project_id)))}

    @app.get("/projects")
    def projects_route() -> dict[str, Any]:
        return {"projects": [{**p.dict(), "runs": len(run_dirs(store.runs_dir(p.id)))} for p in store.list()]}

    @app.post("/projects")
    async def create_project(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="a project needs a name")
        proj = store.create(name, goal=str(body.get("goal") or ""), cartridge=str(body.get("cartridge") or ""))
        return _summary(proj.id)

    @app.get("/projects/{project_id}")
    def project_route(project_id: str) -> dict[str, Any]:
        if store.get(project_id) is None:
            raise HTTPException(status_code=404, detail="unknown project")
        return _summary(project_id)

    @app.get("/projects/{project_id}/runs")
    def project_runs_route(project_id: str, limit: int = 50, session: str = "") -> dict[str, Any]:
        if store.get(project_id) is None:
            raise HTTPException(status_code=404, detail="unknown project")
        runs = list_runs(store.runs_dir(project_id), limit=limit)
        if session:
            runs = [r for r in runs if r.session == session]
        return {"runs": [s.dict() for s in runs]}

    @app.get("/projects/{project_id}/data")
    def list_data(project_id: str) -> dict[str, Any]:
        if store.get(project_id) is None:
            raise HTTPException(status_code=404, detail="unknown project")
        return {"datasets": list_datasets(store.data_dir(project_id))}

    @app.put("/projects/{project_id}/data/{name}")
    async def upload_data(project_id: str, name: str, request: Request) -> dict[str, Any]:
        """Upload a dataset to a project (raw body, like the run upload). It is
        seeded into every new run of the project."""
        if store.get(project_id) is None:
            raise HTTPException(status_code=404, detail="unknown project")
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="empty upload")
        stored = store_dataset(store.data_dir(project_id), name, payload)
        return {"name": stored, "bytes": len(payload)}

    @app.delete("/projects/{project_id}/data/{name}")
    def remove_data(project_id: str, name: str) -> dict[str, Any]:
        if store.get(project_id) is None:
            raise HTTPException(status_code=404, detail="unknown project")
        if not delete_dataset(store.data_dir(project_id), name):
            raise HTTPException(status_code=404, detail="no such dataset")
        return {"deleted": name}

    @app.post("/sessions")
    async def register_session(request: Request) -> dict[str, Any]:
        """Tell the server which project a chat session belongs to, so a run the
        orchestrator starts in that chat lands in the right project. JSON:
        `{id, project}`. Best-effort; an unregistered session is `default`."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        sid = str((body or {}).get("id") or "").strip()
        project = str((body or {}).get("project") or "default")
        if not sid:
            raise HTTPException(status_code=400, detail="a session needs an id")
        if store.get(project) is None:
            raise HTTPException(status_code=400, detail=f"unknown project {project!r}")
        if sessions is not None:
            sessions.set(sid, project)
        return {"id": sid, "project": project}
