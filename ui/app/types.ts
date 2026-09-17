/** `dsagent.file` — `docs/ui-slice.md` §3. */
export type RunFile = {
  run_id: string;
  step: string;
  path: string;
  kind: "deliverable" | "working";
  change: "created" | "modified";
  size: number;
  mtime: number;
};

export type StepStatus = "started" | "done" | "failed" | "awaiting_gate";

/** One row of the DAG panel, accumulated from `dsagent.step` and `dsagent.tool`. */
export type StepRow = {
  step: string;
  persona: string;
  index: number;
  total: number;
  status: StepStatus;
  /** Declared `produces`, patterns included — the promise. */
  produces: string[];
  /** Each promise mapped to what actually landed. */
  matched: Record<string, string[]>;
  startedAt: number;
  endedAt: number | null;
  error: string | null;
  tools: Record<string, number>;
};

export function reduceStep(rows: StepRow[], e: Record<string, any>): StepRow[] {
  const i = rows.findIndex((r) => r.step === e.step);
  const base: StepRow =
    i === -1
      ? {
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
        }
      : rows[i];

  const row: StepRow = {
    ...base,
    status: e.status,
    produces: e.produces ?? base.produces,
    // `started` carries empty lists by design; keep what we had rather than
    // blanking a re-entered step's history.
    matched: e.status === "started" ? base.matched : (e.produces_matched ?? base.matched),
    endedAt: e.status === "started" ? null : e.ts,
    error: e.error ?? null,
  };
  if (i === -1) return [...rows, row].sort((a, b) => a.index - b.index);
  const next = [...rows];
  next[i] = row;
  return next;
}

export function reduceTool(rows: StepRow[], e: Record<string, any>): StepRow[] {
  if (e.phase !== "started") return rows;
  const i = rows.findIndex((r) => r.step === e.step);
  if (i === -1) return rows;
  const row = rows[i];
  const next = [...rows];
  next[i] = { ...row, tools: { ...row.tools, [e.tool]: (row.tools[e.tool] ?? 0) + 1 } };
  return next;
}
