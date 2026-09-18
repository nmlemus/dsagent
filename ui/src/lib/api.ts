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

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  cartridges: () => fetch("/cartridges").then(json<CartridgesResponse>),

  runs: (limit = 50) =>
    fetch(`/runs?limit=${limit}`).then(json<{ runs: RunSummary[] }>).then((r) => r.runs),

  run: (id: string) => fetch(`/runs/${id}`).then(json<RunDetail>),

  // --- projects & sessions ---
  projects: () => fetch("/projects").then(json<{ projects: Project[] }>).then((r) => r.projects),

  createProject: (name: string, goal = "") =>
    fetch("/projects", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, goal }),
    }).then(json<Project>),

  projectRuns: (project: string, opts: { limit?: number; session?: string } = {}) => {
    const q = new URLSearchParams();
    if (opts.limit) q.set("limit", String(opts.limit));
    if (opts.session) q.set("session", opts.session);
    return fetch(`/projects/${project}/runs?${q}`).then(json<{ runs: RunSummary[] }>).then((r) => r.runs);
  },

  // Tell the server which project a chat belongs to, so a run the orchestrator
  // starts in it lands in the right project.
  registerSession: (id: string, project: string) =>
    fetch("/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id, project }),
    }).then(json<{ id: string; project: string }>),

  create: (
    workflow: string,
    inputs: Record<string, unknown>,
    opts: { project?: string; session?: string } = {},
  ) =>
    fetch("/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ workflow, inputs, project: opts.project, session: opts.session }),
    }).then(json<{ run_id: string; project: string; inputs: Record<string, unknown> }>),

  uploadData: (id: string, name: string, file: File) =>
    fetch(`/runs/${id}/data/${encodeURIComponent(name)}?filename=${encodeURIComponent(file.name)}`, {
      method: "PUT",
      body: file,
    }).then(json<{ stored: string; input: string }>),

  start: (id: string) =>
    fetch(`/runs/${id}/start`, { method: "POST" }).then(json<Record<string, unknown>>),

  stop: (id: string) =>
    fetch(`/runs/${id}/stop`, { method: "POST" }).then(json<Record<string, unknown>>),

  gate: (id: string, decision: GateDecision, note = "") =>
    fetch(`/runs/${id}/gate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ decision, note }),
    }).then(json<{ accepted: boolean; decision: string }>),

  del: (id: string) => fetch(`/runs/${id}`, { method: "DELETE" }).then(json<{ deleted: string }>),

  preview: (id: string, path: string, rows = 50) =>
    fetch(`/runs/${id}/preview/${path}?rows=${rows}`).then(
      json<{ columns: string[]; rows: unknown[][]; truncated?: boolean }>,
    ),

  fileUrl: (id: string, path: string) => `/runs/${id}/files/${path}`,
  exportUrl: (id: string) => `/runs/${id}/export.html`,
  downloadUrl: (id: string, everything = false) =>
    `/runs/${id}/download${everything ? "?everything=true" : ""}`,
};

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
