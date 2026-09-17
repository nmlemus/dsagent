/**
 * A run screen, reduced from its event log.
 *
 * The same reducer handles a run that finished last week and a run that is three
 * seconds old: both arrive as `dsagent.*` events in order, the first as a
 * backlog and the second one at a time. That is what makes a reload, a second
 * tab and a CLI-started run the same screen (`docs/ui-product.md` §4.2), and it
 * is why nothing here reads the live AG-UI stream the way the M2.2 slice did.
 *
 * Event payloads: `docs/ui-slice.md` §3 and its M2.5 amendment.
 */

export type RunEvent = {
  index: number;
  name: "dsagent.step" | "dsagent.tool" | "dsagent.file" | "dsagent.note";
  value: Record<string, any>;
};

export type StepStatus = "started" | "done" | "failed" | "awaiting_gate";

/** A gate as it rides on a step event: pending while `decision` is null. */
export type GateOnStep = {
  kind: string;
  prompt: string;
  asked_at: number | null;
  decision: "approve" | "reject" | null;
  note: string;
  decided_at: number | null;
};

export type RunFile = {
  step: string;
  path: string;
  kind: "deliverable" | "working";
  change: "created" | "modified";
  size: number;
  mtime: number;
};

export type Note = { step: string; persona: string; text: string; ts: number };

export type StepRow = {
  step: string;
  persona: string;
  index: number;
  total: number;
  status: StepStatus;
  /** Declared `produces`, patterns included — the promise. */
  produces: string[];
  /** Each promise mapped to the files that have actually landed under it. */
  matched: Record<string, string[]>;
  startedAt: number;
  endedAt: number | null;
  error: string | null;
  tools: Record<string, number>;
  /** How many tool calls are still in flight — the "is anything happening" signal. */
  running: number;
  gate: GateOnStep | null;
  notes: Note[];
};

export type RunView = {
  steps: StepRow[];
  files: RunFile[];
  /** The step whose gate is waiting for an answer, if any. */
  gateStep: StepRow | null;
  /** The file the canvas should be showing, unless the operator pinned one. */
  focus: string | null;
  /** The last event index applied — the cursor a reconnect resumes from. */
  cursor: number;
};

export const emptyRun: RunView = { steps: [], files: [], gateStep: null, focus: null, cursor: 0 };

export function reduce(view: RunView, event: RunEvent): RunView {
  const next =
    event.name === "dsagent.step"
      ? applyStep(view, event.value)
      : event.name === "dsagent.file"
        ? applyFile(view, event.value)
        : event.name === "dsagent.tool"
          ? applyTool(view, event.value)
          : applyNote(view, event.value);
  return { ...next, cursor: Math.max(view.cursor, event.index ?? 0) };
}

export const reduceAll = (view: RunView, events: RunEvent[]): RunView =>
  events.reduce(reduce, view);

function applyStep(view: RunView, e: Record<string, any>): RunView {
  const steps = upsert(view.steps, e.step, (row) => {
    const base: StepRow = row ?? {
      step: e.step,
      persona: e.persona,
      index: e.index,
      total: e.total,
      status: e.status,
      produces: e.produces ?? [],
      matched: e.produces_matched ?? {},
      startedAt: e.ts,
      endedAt: null,
      error: null,
      tools: {},
      running: 0,
      gate: null,
      notes: [],
    };
    return {
      ...base,
      status: e.status,
      produces: e.produces ?? base.produces,
      // `started` carries empty lists by design; keeping what we had is what
      // stops a re-entered step from blanking its own history.
      matched: e.status === "started" ? base.matched : (e.produces_matched ?? base.matched),
      startedAt: e.status === "started" ? e.ts : base.startedAt,
      endedAt: e.status === "started" ? null : e.ts,
      error: e.error ?? null,
      // A pending gate replaces an answered one only forwards: the run is asking
      // again, which is what a resume does.
      gate: (e.gate as GateOnStep) ?? base.gate,
      running: e.status === "started" ? base.running : 0,
    };
  });

  const view2 = { ...view, steps };
  const waiting = steps.find((s) => s.status === "awaiting_gate" && s.gate?.decision == null);
  const focus =
    e.status === "done" || e.status === "awaiting_gate"
      ? (lastMatched(e) ?? view.focus)
      : view.focus;
  return { ...view2, gateStep: waiting ?? null, focus };
}

function applyFile(view: RunView, e: Record<string, any>): RunView {
  const file: RunFile = {
    step: e.step, path: e.path, kind: e.kind, change: e.change, size: e.size, mtime: e.mtime,
  };
  const i = view.files.findIndex((f) => f.path === file.path);
  // Append-only: four figures landing in twenty seconds must not reorder the
  // list under the reader's eye (run 001, observation 2).
  const files = i === -1 ? [...view.files, file] : view.files.map((f, n) => (n === i ? file : f));
  return { ...view, files };
}

function applyTool(view: RunView, e: Record<string, any>): RunView {
  const started = e.phase === "started";
  const steps = upsert(view.steps, e.step, (row) =>
    row
      ? {
          ...row,
          tools: started ? { ...row.tools, [e.tool]: (row.tools[e.tool] ?? 0) + 1 } : row.tools,
          running: Math.max(0, row.running + (started ? 1 : -1)),
        }
      : row,
  );
  return { ...view, steps };
}

function applyNote(view: RunView, e: Record<string, any>): RunView {
  const note: Note = { step: e.step, persona: e.persona, text: e.text, ts: e.ts };
  const steps = upsert(view.steps, e.step, (row) =>
    row ? { ...row, notes: [...row.notes, note] } : row,
  );
  return { ...view, steps };
}

/** Update the row for `step`, inserting in DAG order when it is new. */
function upsert(
  rows: StepRow[],
  step: string,
  make: (row: StepRow | undefined) => StepRow | undefined,
): StepRow[] {
  const i = rows.findIndex((r) => r.step === step);
  const updated = make(i === -1 ? undefined : rows[i]);
  if (!updated) return rows;
  if (i === -1) return [...rows, updated].sort((a, b) => a.index - b.index);
  const next = [...rows];
  next[i] = updated;
  return next;
}

/**
 * The step's last file that actually exists, for the focus move.
 *
 * `produces` is the promise and may be a glob, so it is not necessarily a path;
 * `produces_matched` is what landed. Reading the matched side is what lets a step
 * whose contract is `artifacts/figures/*.png` focus an actual figure. Focus moves
 * only when a step ends — never on a file event, or the four-figure burst in
 * `analyze` would drag the pane around four times (run 001, observation 3).
 */
function lastMatched(step: Record<string, any>): string | null {
  const declared: string[] = step.produces ?? [];
  const matched: Record<string, string[]> = step.produces_matched ?? {};
  for (let i = declared.length - 1; i >= 0; i--) {
    const hits = matched[declared[i]] ?? [];
    if (hits.length > 0) return hits[hits.length - 1];
  }
  return null;
}

/** Deliverables first, then working files; each group in the order it landed. */
export function grouped(files: RunFile[]): { label: string; files: RunFile[] }[] {
  return [
    { label: "Deliverables", files: files.filter((f) => f.kind === "deliverable") },
    { label: "Working files", files: files.filter((f) => f.kind !== "deliverable") },
  ].filter((group) => group.files.length > 0);
}
