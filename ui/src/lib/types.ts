// Shapes returned by `dsagent serve` (src/dsagent/api.py, runs.py). Only the
// fields the UI reads are typed; the server may send more.

export type RunStatus =
  | "pending"
  | "running"
  | "awaiting_gate"
  | "done"
  | "failed"
  | "stopped";

export interface InputSpec {
  type: string;
  required: boolean;
  default: unknown;
  options: string[] | null;
  guess: unknown;
}

export interface GateSpec {
  kind: "human" | "auto";
  prompt: string;
}

export interface StepShape {
  id: string;
  persona: string;
  env: string;
  needs: string[];
  produces: string[];
  section: string;
  gate: GateSpec | null;
}

export interface WorkflowShape {
  name: string;
  cartridge: string;
  description: string;
  env: string;
  title_input: string | null;
  data_input: string | null;
  inputs: Record<string, InputSpec>;
  personas: string[];
  steps: StepShape[];
}

export interface CartridgeInfo {
  name: string;
  version: string;
  description: string;
}

export interface Project {
  id: string;
  name: string;
  goal: string;
  cartridge: string;
  created_at: number;
  runs: number;
}

export interface CartridgesResponse {
  cartridges: CartridgeInfo[];
  workflows: WorkflowShape[];
  replay: string | null;
}

export interface AwaitingGate {
  step: string;
  prompt: string;
  produces?: string[];
  asked_at?: number;
  kind?: string;
}

export interface RunSummary {
  run_id: string;
  workflow: string;
  cartridge: string;
  status: RunStatus;
  inputs: Record<string, unknown>;
  started_at: number | null;
  finished_at: number | null;
  duration: number | null;
  steps_done: number;
  steps_total: number;
  usage: Record<string, number>;
  cost_usd: number | null;
  gate_wait: number;
  awaiting: AwaitingGate | null;
  session?: string;
}

export interface RunDetail extends RunSummary {
  steps: Record<string, StepRecord>;
  workflow_shape: WorkflowShape | null;
  plan: unknown;
  driver_error: string | null;
}

export interface StepRecord {
  status?: string;
  gate?: {
    kind: string;
    prompt: string;
    asked_at?: number;
    decision?: string | null;
    note?: string;
    decided_at?: number;
  } | null;
  [k: string]: unknown;
}

// --- Live events (docs/ui-slice.md §3). Each SSE `data:` is one of these. ---

export interface EventEnvelope<V = Record<string, unknown>> {
  index: number;
  name: string;
  value: V;
}

export interface StepEvent {
  run_id: string;
  workflow: string;
  step: string;
  persona: string;
  env: string;
  index: number;
  total: number;
  status: "started" | "done" | "failed" | "awaiting_gate";
  needs: string[];
  produces: string[];
  produces_matched: Record<string, string[]>;
  gate: {
    kind: string;
    prompt: string;
    asked_at?: number;
    decision?: string | null;
    note?: string;
    decided_at?: number;
  } | null;
  ts: number;
  error: string | null;
}

export interface ToolEvent {
  run_id: string;
  step: string;
  persona: string;
  tool: string;
  tool_call_id: string;
  phase: "started" | "finished";
  args_preview: Record<string, unknown>;
  ts: number;
}

export interface FileEvent {
  run_id: string;
  step: string;
  path: string;
  kind: "deliverable" | "working";
  change: "created" | "modified";
  size: number;
  mtime: number;
  ts: number;
}

export interface NoteEvent {
  run_id: string;
  step: string;
  persona: string;
  text: string;
  ts: number;
}

export type GateDecision = "approve" | "reject";
