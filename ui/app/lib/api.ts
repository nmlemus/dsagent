/**
 * The runs API, typed. `src/dsagent/api.py` is the other half of every shape here.
 *
 * Everything goes through `/dsa/*`, which `next.config.mjs` rewrites onto
 * `dsagent serve`. Same-origin on purpose: `fetch` is subject to CORS and
 * `<img>`/`<iframe>` are not, and the first canvas showed figures while failing
 * on text because of exactly that. The prefix is `/dsa` rather than `/runs`
 * because `/runs/<id>` is a page in this app — a rewrite on that path would
 * swallow the run screen itself.
 */

export const API = "/dsa";

export type RunStatus = "pending" | "running" | "awaiting_gate" | "done" | "failed";

/** The gate a run is standing at right now — `RunState.gate` in `run.json`. */
export type PendingGate = {
  run_id: string;
  workflow: string;
  step: string;
  persona: string;
  produces: string[];
  prompt: string;
  kind: string;
  asked_at: number;
};

export type RunSummary = {
  run_id: string;
  workflow: string;
  cartridge: string;
  status: RunStatus;
  inputs: Record<string, string>;
  started_at: number | null;
  finished_at: number | null;
  duration: number | null;
  steps_done: number;
  steps_total: number;
  usage: Record<string, number>;
  cost_usd: number | null;
  /** Seconds this run spent waiting for a person, summed over its gates. */
  gate_wait: number;
  awaiting: PendingGate | null;
  /** Whether a process is actually standing behind it, as opposed to having been. */
  live: boolean;
  error: string | null;
};

export type GateRecord = {
  decision: "approve" | "reject";
  note: string;
  ts: number;
  asked_at: number;
};

export type StepRecord = {
  id: string;
  status: "pending" | "running" | "done" | "failed";
  gate: GateRecord | null;
  started_at: number | null;
  finished_at: number | null;
  output: string;
  error: string;
  tool_calls: Record<string, number>;
  usage: Record<string, number>;
  skills_read: string[];
};

export type WorkflowStep = {
  id: string;
  persona: string;
  env: string;
  needs: string[];
  produces: string[];
  gate: { kind: string; prompt: string | null } | null;
};

export type WorkflowInput = {
  type: string;
  required: boolean;
  default: string | null;
  options: string[] | null;
};

export type WorkflowShape = {
  name: string;
  cartridge: string;
  description: string;
  env: string;
  inputs: Record<string, WorkflowInput>;
  personas: string[];
  steps: WorkflowStep[];
};

export type RunDetail = RunSummary & {
  steps: Record<string, StepRecord>;
  workflow_shape: WorkflowShape | null;
  driver_error: string | null;
};

export type Catalogue = {
  cartridges: { name: string; version: string; description: string }[];
  workflows: WorkflowShape[];
  /** In replay mode, the one workflow the recording can play. */
  replay: string | null;
};

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    cache: "no-store",
    ...init,
    headers: init?.body ? { "content-type": "application/json", ...init?.headers } : init?.headers,
  });
  if (!response.ok) {
    throw new ApiError(await describe(response), response.status);
  }
  return (await response.json()) as T;
}

/**
 * An error a person can read.
 *
 * The API answers a refusal with `{"detail": "..."}`; anything else is shown as
 * the status line. What must never reach the screen is a stack trace from a
 * minified bundle, which is what the M2.2 error toast did (M2.2.1, item 6).
 */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function describe(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    /* not JSON; the status line is what we have */
  }
  return `${response.status} ${response.statusText}`;
}

export const listRuns = () => json<{ runs: RunSummary[] }>("/runs").then((b) => b.runs);

export const getRun = (runId: string) => json<RunDetail>(`/runs/${encodeURIComponent(runId)}`);

export const getCatalogue = () => json<Catalogue>("/cartridges");

export const createRun = (workflow: string, inputs: Record<string, string>) =>
  json<{ run_id: string; inputs: Record<string, string> }>("/runs", {
    method: "POST",
    body: JSON.stringify({ workflow, inputs }),
  });

/** Send the operator's file straight up as the body — see `upload_input`. */
export const uploadInput = (runId: string, input: string, file: File) =>
  json<{ input: string; value: string; bytes: number }>(
    `/runs/${encodeURIComponent(runId)}/data/${encodeURIComponent(input)}` +
      `?filename=${encodeURIComponent(file.name)}`,
    { method: "PUT", body: file, headers: { "content-type": "application/octet-stream" } },
  );

export const startRun = (runId: string) =>
  json<{ started: boolean; resumed?: boolean; reason?: string }>(
    `/runs/${encodeURIComponent(runId)}/start`,
    { method: "POST" },
  );

export const answerGate = (runId: string, decision: "approve" | "reject", note = "") =>
  json<{ accepted: boolean; decision: string }>(`/runs/${encodeURIComponent(runId)}/gate`, {
    method: "POST",
    body: JSON.stringify({ decision, note }),
  });

/** A workspace file's URL. Relative, so images, iframes and fetch all agree. */
export function fileUrl(runId: string, path: string): string {
  const segments = path.split("/").map(encodeURIComponent).join("/");
  return `${API}/runs/${encodeURIComponent(runId)}/files/${segments}`;
}

export const eventsUrl = (runId: string) => `${API}/runs/${encodeURIComponent(runId)}/events`;
