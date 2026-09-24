// Thin typed client over the `dsagent serve` REST + SSE surface. Same-origin in
// production (the SPA is served by the server); the Vite dev proxy handles :8000.

import type {
  CartridgesResponse,
  EventEnvelope,
  GateDecision,
  Project,
  RunDetail,
  RunSummary,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// The sentence the server put in `detail`, for a screen to show instead of a
// status code. FastAPI's body is `{"detail": "..."}` and `ApiError.message` is
// `"<status> <statusText>: <body>"`, so the detail is what follows the colon.
export function detailOf(e: unknown): string {
  if (!(e instanceof ApiError)) return "";
  const body = e.message.slice(e.message.indexOf(":") + 1).trim();
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    /* not JSON: fall through to the raw body */
  }
  return body;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiError(
      res.status,
      `${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`,
    );
  }
  return res.json() as Promise<T>;
}

export const api = {
  cartridges: () => fetch("/cartridges").then(json<CartridgesResponse>),

  runs: (limit = 50) =>
    fetch(`/runs?limit=${limit}`)
      .then(json<{ runs: RunSummary[] }>)
      .then((r) => r.runs),

  run: (id: string) => fetch(`/runs/${id}`).then(json<RunDetail>),

  // --- projects & sessions ---
  projects: () =>
    fetch("/projects")
      .then(json<{ projects: Project[] }>)
      .then((r) => r.projects),

  createProject: (name: string, goal = "", cartridge = "") =>
    fetch("/projects", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, goal, cartridge }),
    }).then(json<Project>),

  projectRuns: (
    project: string,
    opts: { limit?: number; session?: string } = {},
  ) => {
    const q = new URLSearchParams();
    if (opts.limit) q.set("limit", String(opts.limit));
    if (opts.session) q.set("session", opts.session);
    return fetch(`/projects/${project}/runs?${q}`)
      .then(json<{ runs: RunSummary[] }>)
      .then((r) => r.runs);
  },

  // --- project datasets (uploaded once, seeded into each run) ---
  datasets: (project: string) =>
    fetch(`/projects/${project}/data`)
      .then(json<{ datasets: { name: string; bytes: number }[] }>)
      .then((r) => r.datasets),

  uploadDataset: (project: string, file: File) =>
    fetch(`/projects/${project}/data/${encodeURIComponent(file.name)}`, {
      method: "PUT",
      body: file,
    }).then(json<{ name: string; bytes: number }>),

  deleteDataset: (project: string, name: string) =>
    fetch(`/projects/${project}/data/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }).then(json<{ deleted: string }>),

  // --- sessions (chats live on the server: durable, shared, project-mapped) ---
  sessions: (project: string) =>
    fetch(`/projects/${project}/sessions`)
      .then(
        json<{
          sessions: {
            id: string;
            title: string;
            updated_at: number;
            project: string;
          }[];
        }>,
      )
      .then((r) => r.sessions),

  getSession: (id: string) =>
    fetch(`/sessions/${id}`).then(
      json<{
        id: string;
        project: string;
        title: string;
        updated_at: number;
        messages: unknown[];
      }>,
    ),

  saveSession: (
    id: string,
    body: {
      project: string;
      title: string;
      messages: unknown[];
      updated_at?: number;
    },
    opts: { keepalive?: boolean } = {},
  ) =>
    fetch(`/sessions/${id}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      // `keepalive` lets a save started as the page unloads still complete, so
      // the last turn isn't lost when the user reloads or closes right after it.
      keepalive: opts.keepalive,
    }).then(json<{ id: string; project: string }>),

  deleteSession: (id: string) =>
    fetch(`/sessions/${id}`, { method: "DELETE" }).then(
      json<{ deleted: string }>,
    ),

  // --- project artifacts (files a chat wrote: reports, charts) ---
  projectArtifacts: (project: string) =>
    fetch(`/projects/${project}/artifacts`)
      .then(
        json<{ artifacts: { path: string; bytes: number; mtime: number }[] }>,
      )
      .then((r) => r.artifacts),

  projectFileUrl: (project: string, path: string) =>
    `/projects/${project}/files/${path.split("/").map(encodeURIComponent).join("/")}`,

  projectPreview: (project: string, path: string, rows = 50) =>
    fetch(`/projects/${project}/preview/${path}?rows=${rows}`).then(
      json<{ columns: string[]; rows: unknown[][]; truncated?: boolean }>,
    ),

  create: (
    workflow: string,
    inputs: Record<string, unknown>,
    opts: { project?: string; session?: string } = {},
  ) =>
    fetch("/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        workflow,
        inputs,
        project: opts.project,
        session: opts.session,
      }),
    }).then(
      json<{
        run_id: string;
        project: string;
        inputs: Record<string, unknown>;
      }>,
    ),

  uploadData: (id: string, name: string, file: File) =>
    fetch(
      `/runs/${id}/data/${encodeURIComponent(name)}?filename=${encodeURIComponent(file.name)}`,
      {
        method: "PUT",
        body: file,
      },
    ).then(json<{ stored: string; input: string }>),

  start: (id: string) =>
    fetch(`/runs/${id}/start`, { method: "POST" }).then(
      json<Record<string, unknown>>,
    ),

  stop: (id: string) =>
    fetch(`/runs/${id}/stop`, { method: "POST" }).then(
      json<Record<string, unknown>>,
    ),

  gate: (id: string, decision: GateDecision, note = "", gateId?: string) =>
    fetch(`/runs/${id}/gate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ decision, note, gate_id: gateId }),
    }).then(json<{ accepted: boolean; decision: string }>),

  del: (id: string) =>
    fetch(`/runs/${id}`, { method: "DELETE" }).then(json<{ deleted: string }>),

  preview: (id: string, path: string, rows = 50) =>
    fetch(`/runs/${id}/preview/${path}?rows=${rows}`).then(
      json<{ columns: string[]; rows: unknown[][]; truncated?: boolean }>,
    ),

  fileUrl: (id: string, path: string) => `/runs/${id}/files/${path}`,
  exportUrl: (id: string) => `/runs/${id}/export.html`,
  downloadUrl: (id: string, everything = false) =>
    `/runs/${id}/download${everything ? "?everything=true" : ""}`,
};

// Where a file lives and how to read it, so one viewer serves both a run's
// workspace and a project's chat workspace. The two differ only in their URLs;
// everything a viewer does (fetch text, draw a table, frame a document) is the
// same, and a chat's report deserves the same reading experience as a run's.
export interface FileSource {
  url: (path: string) => string;
  preview: (
    path: string,
    rows?: number,
  ) => Promise<{ columns: string[]; rows: unknown[][]; truncated?: boolean }>;
}

export const runSource = (runId: string): FileSource => ({
  url: (path) => api.fileUrl(runId, path),
  preview: (path, rows) => api.preview(runId, path, rows),
});

export const projectSource = (project: string): FileSource => ({
  url: (path) => api.projectFileUrl(project, path),
  preview: (path, rows) => api.projectPreview(project, path, rows),
});

// Subscribe to a run's event stream. Rebuilds from `events.jsonl` (after=0) then
// follows live — the same endpoint does restore and follow (api.py events_stream).
// Returns an unsubscribe function.
export function subscribeEvents(
  runId: string,
  onEvent: (e: EventEnvelope) => void,
  opts: { onEnd?: () => void; onError?: (err: Event) => void } = {},
): () => void {
  const source = new EventSource(`/runs/${runId}/events`);
  source.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as EventEnvelope);
    } catch {
      /* half-written line; the server drops these, but guard anyway */
    }
  };
  source.addEventListener("end", () => {
    opts.onEnd?.();
    source.close();
  });
  source.onerror = (err) => opts.onError?.(err);
  return () => source.close();
}
