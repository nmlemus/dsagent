"use client";

import type { RunDetail } from "../lib/api";
import { duration, money, tokens } from "../lib/format";
import type { RunView } from "../lib/run-state";

import { FileView } from "./file-view";

/**
 * Where raw lives — and it is never the default.
 *
 * Progressive disclosure is the pattern the research is unambiguous about: a
 * summary, then the step, then its tool calls, then the bytes. Hiding reasoning
 * entirely and dumping a full trace are both listed as anti-patterns, and this
 * is the third option — everything is reachable in one click from the thing it
 * belongs to, and nothing arrives unasked.
 */
export type Detail = { kind: "file" | "step" | "spec"; id: string } | null;

export function Drawer({
  runId,
  detail,
  view,
  run,
  onClose,
}: {
  runId: string;
  detail: Detail;
  view: RunView;
  /** The run record, for the per-step telemetry the event log does not carry. */
  run: RunDetail | null;
  onClose: () => void;
}) {
  if (!detail) return null;
  return (
    <aside className="drawer" aria-label="Detail">
      <header className="drawer-head">
        <b className="drawer-title">
          {detail.kind === "file"
            ? detail.id
            : detail.kind === "spec"
              ? `Vega-Lite · ${detail.id}`
              : `Step · ${detail.id}`}
        </b>
        <button type="button" className="btn btn-quiet btn-small" onClick={onClose}>
          Close
        </button>
      </header>
      {detail.kind === "file" ? (
        <FileView runId={runId} path={detail.id} />
      ) : detail.kind === "spec" ? (
        <SpecDetail id={detail.id} view={view} />
      ) : (
        <StepDetail id={detail.id} view={view} run={run} />
      )}
    </aside>
  );
}

/**
 * A chart's spec, as the persona wrote it.
 *
 * This is the answer to "hidden code is a complaint" (research part A, listed as
 * table stakes): the thing on screen is a document, and its source is one click
 * away, unedited.
 */
function SpecDetail({ id, view }: { id: string; view: RunView }) {
  const card = view.cards.find((c) => c.chartId === id);
  if (!card) return <p className="view-note dim">No such chart.</p>;
  return (
    <div className="drawer-body">
      <dl className="drawer-facts">
        <Fact label="chart_id" value={card.chartId} mono />
        <Fact label="version" value={String(card.version)} mono />
        <Fact label="by" value={card.persona} />
        <Fact label="data" value={card.dataRef} mono />
        <Fact label="rows" value={card.rows == null ? "—" : String(card.rows)} mono />
      </dl>
      <Block title="Specification">
        <pre className="drawer-pre">{JSON.stringify(card.spec, null, 2)}</pre>
      </Block>
    </div>
  );
}

/** One step, all the way down: what it cost, what it called, what it said. */
function StepDetail({ id, view, run }: { id: string; view: RunView; run: RunDetail | null }) {
  const step = view.steps.find((s) => s.step === id);
  if (!step) return <p className="view-note dim">This step has not started.</p>;
  const record = run?.steps?.[id];
  const calls = Object.entries(step.tools).sort((a, b) => b[1] - a[1]);
  const cards = view.cards.filter((c) => c.step === id);

  return (
    <div className="drawer-body">
      <dl className="drawer-facts">
        <Fact label="persona" value={step.persona} />
        <Fact label="status" value={step.status} />
        <Fact
          label="elapsed"
          value={step.endedAt ? duration(step.endedAt - step.startedAt) : "running"}
        />
        <Fact label="model" value={record?.model || "—"} mono />
        <Fact label="tokens" value={tokens(record?.usage?.input_tokens)} mono />
        <Fact label="cost" value={money(record?.cost_usd)} mono />
      </dl>

      <Block title="Tool calls">
        {calls.length === 0 ? (
          <p className="dim">none yet</p>
        ) : (
          <pre className="drawer-pre">
            {calls.map(([tool, n]) => `${tool.padEnd(18)} × ${n}`).join("\n")}
          </pre>
        )}
      </Block>

      {cards.length > 0 && (
        <Block title="Emitted">
          <pre className="drawer-pre">
            {cards.map((c) => `${c.kind.padEnd(6)} ${c.chartId} v${c.version}`).join("\n")}
          </pre>
        </Block>
      )}

      {step.notes.length > 0 && (
        <Block title={`${step.persona}’s working notes`}>
          {step.notes.map((note, i) => (
            <p key={i} className="drawer-note">
              {note.text}
            </p>
          ))}
        </Block>
      )}

      {step.error && (
        <Block title="Error">
          <p className="drawer-error">{step.error}</p>
        </Block>
      )}
    </div>
  );
}

function Fact({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="drawer-fact">
      <dt>{label}</dt>
      <dd className={mono ? "mono" : undefined}>{value}</dd>
    </div>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="drawer-block">
      <h3 className="drawer-block-head">{title}</h3>
      {children}
    </section>
  );
}
