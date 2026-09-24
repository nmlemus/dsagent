// Shapes returned by `dsagent serve` (src/dsagent/api.py, runs.py). Only the
// fields the UI reads are typed; the server may send more.

export type RunStatus =
  "pending" | "running" | "awaiting_gate" | "done" | "failed" | "stopped" | "interrupted";

export interface InputSpec {
  type: string;
  required: boolean;
  default: unknown;
  options: string[] | null;
  guess: unknown;
  // Presentation, from workflow.yaml. `null` means "show the input's name".
  label: string | null;
  placeholder: string | null;
  help: string | null;
}

export interface GateSpec {
  kind: "human" | "auto";
  prompt: string;
}

export interface StepShape {
  id: string;
  title: string | null;
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
  title: string | null;
  // What the user is told the workflow does; `description` is for the model.
  summary: string | null;
  icon: string | null;
  badge: string | null;
  env: string;
  title_input: string | null;
  data_input: string | null;
  inputs: Record<string, InputSpec>;
  personas: string[];
  steps: StepShape[];
}

// --- The experience: every word the UI shows inside its fixed structure,
// declared by the cartridge (`cartridge.yaml: experience`). Mirrors
// `dsagent.cartridge.models.Experience`; the server always sends it complete.

export interface Noun {
  singular: string;
  plural: string;
}

export interface Experience {
  version: number;
  title: string;
  tagline: string;
  nouns: { run: Noun; resource: Noun; deliverable: Noun };
  nav: {
    overview: string;
    chat: string;
    runs: string;
    resources: string;
    deliverables: string;
  };
  home: {
    eyebrow: string;
    headline: string;
    intro: string;
    stages: string[];
    steps: { title: string; text: string }[];
  };
  assistant: {
    headline: string;
    intro: string;
    placeholder: string;
    suggestions: string[];
    instructions: string;
  };
  resources: { intro: string; formats: string[] };
}

export interface PersonaInfo {
  name: string;
  role: string;
  description: string;
}

export interface CartridgeInfo {
  name: string;
  version: string;
  description: string;
  experience: Experience;
  personas: PersonaInfo[];
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
  gate_id?: string;
  attempt?: number;
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
