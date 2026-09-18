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

    @app.get("/projects/{project_id}/sessions")
    def list_sessions(project_id: str) -> dict[str, Any]:
        if store.get(project_id) is None:
            raise HTTPException(status_code=404, detail="unknown project")
        return {"sessions": sessions.list_for(project_id) if sessions is not None else []}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        rec = sessions.get(session_id) if sessions is not None else None
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return rec

    @app.put("/sessions/{session_id}")
    async def save_session(session_id: str, request: Request) -> dict[str, Any]:
        """Persist a chat: its project, title and messages. The client saves as
        the thread grows, so a chat is durable and shared across tabs."""
        import time as _time

        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        project = str(body.get("project") or "default")
        if store.get(project) is None:
            raise HTTPException(status_code=400, detail=f"unknown project {project!r}")
        if sessions is not None:
            sessions.save(
                session_id,
                project=project,
                title=str(body.get("title") or ""),
                messages=body.get("messages") or [],
                updated_at=float(body.get("updated_at") or _time.time()),
            )
        return {"id": session_id, "project": project}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        if sessions is None or not sessions.delete(session_id):
            raise HTTPException(status_code=404, detail="unknown session")
        return {"deleted": session_id}
