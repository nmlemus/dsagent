"use client";

import { useState } from "react";

import type { RunDetail } from "../lib/api";
import { duration, initial, money, percent, tokens } from "../lib/format";
import type { StepRow } from "../lib/run-state";
import { useClock } from "../lib/use-run";
import { GateCard, GateVerdict } from "./gate-card";

/**
 * The progress region: the whole DAG across the top, one step open below it.
 *
 * A horizontal stepper rather than a list, for one reason above taste: the gate
 * card must never need a scroll. A vertical stack showed two of four steps in a
 * 900 px window, which put the one thing an operator has to answer below the
 * fold. Across the top, four steps fit at any height and the open step is the
 * interesting one — the gate if a gate is waiting, otherwise whatever is running.
 *
 * The declared `produces` are drawn as promises with a tick before the files
 * exist. That is the product's thesis on screen: a workflow is a contract, and
 * the runner verifies it on disk.
 */
export function Progress({
  detail,
  steps,
  onOpen,
  onDecide,
  deciding,
  onResume,
  resuming,
}: {
  detail: RunDetail | null;
  steps: StepRow[];
  onOpen: (path: string) => void;
  onDecide: (decision: "approve" | "reject", note: string) => void;
  deciding: boolean;
  onResume: () => void;
  resuming: boolean;
}) {
  const [picked, setPicked] = useState<string | null>(null);
  const attention = focusOf(steps);
  // A click holds until the run's own focus moves on, which is the next moment
  // there is something better to look at.
  const open = steps.find((s) => s.step === picked) ?? attention;
  // The whole DAG, always: the steps that have happened, in the places the
  // workflow declares for them, and the ones still to come drawn as waiting. A
  // stepper that only shows what has run makes a failed run look complete.
  const declared = detail?.workflow_shape?.steps ?? [];
  const order = declared.length > 0 ? declared.map((s) => s.id) : steps.map((s) => s.step);
  const byId = new Map(steps.map((s) => [s.step, s]));

  return (
    <section className="progress" aria-label="Run progress">
      <RunHeader detail={detail} steps={steps} />
      <Stalled detail={detail} steps={steps} onResume={onResume} resuming={resuming} />

      <ol className="stepper">
        {order.map((id) => {
          const row = byId.get(id);
          if (row) {
            return (
              <Chip
                key={id}
                step={row}
                open={id === open?.step}
                onPick={() => setPicked(id === picked ? null : id)}
              />
            );
          }
          const spec = declared.find((s) => s.id === id);
          return (
            <li key={id} className="chip is-pending">
              <span className="chip-button" aria-disabled="true">
                <span className="chip-mark">{initial(spec?.persona ?? "?")}</span>
                <span className="chip-name">{id}</span>
                <span className="chip-note dim">waiting</span>
              </span>
            </li>
          );
        })}
      </ol>

      {open && (
        <StepDetail step={open} onOpen={onOpen} onDecide={onDecide} deciding={deciding} />
      )}
    </section>
  );
}

/**
 * What to say, and offer, when a run has stopped and will not start itself.
 *
 * Three ways a run stands still: a step failed, a gate was sent back, or the
 * process that was driving it is gone. All three are one sentence and one button
 * — the button being `POST /runs/{id}/start`, which is the runner's own resume:
 * finished steps are skipped, the failed one is re-run, the gate is asked again.
 */
function Stalled({
  detail,
  steps,
  onResume,
  resuming,
}: {
  detail: RunDetail | null;
  steps: StepRow[];
  onResume: () => void;
  resuming: boolean;
}) {
  if (!detail) return null;
  const pending = steps.some((s) => s.status === "awaiting_gate" && s.gate?.decision == null);
  if (pending) return null;

  const failed = steps.find((s) => s.status === "failed");
  const sentBack = steps.find((s) => s.gate?.decision === "reject");
  const stale = detail.status === "running" && !detail.live;

  if (detail.status === "failed" || failed) {
    return (
      <Banner
        tone="failed"
        what={
          failed
            ? `${failed.persona} could not finish ${failed.step}. Nothing after it ran.`
            : (detail.error ?? detail.driver_error ?? "The run stopped with an error.")
        }
        action={failed ? `Retry from ${failed.step}` : "Try again"}
        onAct={onResume}
        busy={resuming}
      />
    );
  }
  if (detail.status === "awaiting_gate" && sentBack) {
    return (
      <Banner
        tone="waiting"
        what={
          sentBack.gate?.note
            ? `You sent ${sentBack.step} back: “${sentBack.gate.note}”`
            : `You sent ${sentBack.step} back.`
        }
        action="Resume and reopen the gate"
        onAct={onResume}
        busy={resuming}
      />
    );
  }
  if (stale) {
    return (
      <Banner
        tone="failed"
        what="Nothing is driving this run — the server that started it is gone."
        action="Pick it up from here"
        onAct={onResume}
        busy={resuming}
      />
    );
  }
  return null;
}

function Banner({
  tone,
  what,
  action,
  onAct,
  busy,
}: {
  tone: "failed" | "waiting";
  what: string;
  action: string;
  onAct: () => void;
  busy: boolean;
}) {
  return (
    <div className={`banner is-${tone}`}>
      <p>{what}</p>
      <button className="btn btn-small" onClick={onAct} disabled={busy}>
        {busy ? "Starting…" : action}
      </button>
    </div>
  );
}

/** The step worth looking at: a waiting gate, else what is running, else the last. */
export function focusOf(steps: StepRow[]): StepRow | undefined {
  return (
    steps.find((s) => s.status === "awaiting_gate" && s.gate?.decision == null) ??
    steps.find((s) => s.status === "started") ??
    steps.find((s) => s.status === "failed") ??
    steps[steps.length - 1]
  );
}

function RunHeader({ detail, steps }: { detail: RunDetail | null; steps: StepRow[] }) {
  const usage = detail?.usage ?? {};
  const input = usage.input_tokens ?? 0;
  const cached = usage.cache_read ?? 0;
  const running = steps.some((s) => s.status === "started");

  return (
    <header className="run-header">
      <div className="run-id">
        <h1 className="run-title serif">{detail?.workflow ?? "Run"}</h1>
        <span className="mono dim">{detail?.run_id}</span>
      </div>
      <dl className="metrics">
        <Metric label="Elapsed" value={duration(detail?.duration ?? null)} live={running} />
        <Metric
          label="Waited for you"
          value={duration(detail?.gate_wait ?? 0)}
          hint="Time this run spent standing at a gate"
        />
        <Metric
          label="Tokens"
          value={tokens(input)}
          hint={`${input.toLocaleString()} in · ${(usage.output_tokens ?? 0).toLocaleString()} out`}
        />
        <Metric
          label="Cached"
          value={percent(cached, input)}
          hint="Share of input read from cache"
        />
        <Metric label="Cost" value={money(detail?.cost_usd)} />
      </dl>
    </header>
  );
}

function Metric({
  label,
  value,
  hint,
  live,
}: {
  label: string;
  value: string;
  hint?: string;
  live?: boolean;
}) {
  return (
    <div className="metric" title={hint}>
      <dt>{label}</dt>
      <dd className={live ? "mono is-live" : "mono"}>{value}</dd>
    </div>
  );
}

function Chip({ step, open, onPick }: { step: StepRow; open: boolean; onPick: () => void }) {
  const waiting = step.status === "awaiting_gate" && step.gate?.decision == null;
  const met = Object.values(step.matched).filter((hits) => hits.length > 0).length;

  return (
    <li className={`chip is-${step.status}${waiting ? " is-waiting" : ""}${open ? " is-open" : ""}`}>
      <button onClick={onPick} className="chip-button">
        <span className="chip-mark" title={step.persona}>
          {initial(step.persona)}
        </span>
        <span className="chip-name">{step.step}</span>
        <span className="chip-note">
          {waiting ? (
            "needs you"
          ) : step.status === "started" ? (
            <Elapsed step={step} />
          ) : step.status === "failed" ? (
            "failed"
          ) : (
            `${met}/${step.produces.length} delivered`
          )}
        </span>
      </button>
    </li>
  );
}

function StepDetail({
  step,
  onOpen,
  onDecide,
  deciding,
}: {
  step: StepRow;
  onOpen: (path: string) => void;
  onDecide: (decision: "approve" | "reject", note: string) => void;
  deciding: boolean;
}) {
  const [notesOpen, setNotesOpen] = useState(false);
  const pending = step.status === "awaiting_gate" && step.gate?.decision == null;
  const calls = Object.entries(step.tools).sort((a, b) => b[1] - a[1]);
  const total = calls.reduce((n, [, count]) => n + count, 0);

  return (
    <article className={`detail is-${step.status}`}>
      <div className="detail-head">
        <h2 className="detail-name">{step.step}</h2>
        <span className="dim">{step.persona}</span>
        <Activity step={step} calls={total} />
        <span className="detail-elapsed mono dim">
          <Elapsed step={step} />
        </span>
      </div>

      <ul className="promises">
        {step.produces.map((entry) => {
          const hits = step.matched[entry] ?? [];
          return (
            <li key={entry} className={hits.length ? "is-met" : "is-unmet"}>
              {hits.length === 0 ? (
                <span className="promise mono">{entry}</span>
              ) : (
                hits.map((path) => (
                  <button key={path} className="promise mono" onClick={() => onOpen(path)}>
                    {path}
                  </button>
                ))
              )}
            </li>
          );
        })}
      </ul>

      {calls.length > 0 && (
        <div className="tools">
          {calls.map(([tool, n]) => (
            <span key={tool} className="tool mono">
              {tool} <b>{n}</b>
            </span>
          ))}
        </div>
      )}

      {step.notes.length > 0 && (
        <div className="notes">
          <button className="btn-quiet btn-small" onClick={() => setNotesOpen((v) => !v)}>
            {notesOpen ? "Hide" : "Show"} what {step.persona} said · {step.notes.length}
          </button>
          {notesOpen && (
            <div className="notes-body">
              {step.notes.map((note, i) => (
                <p key={i}>{note.text}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {step.error && <StepError step={step} />}

      {pending ? (
        <GateCard step={step} onDecide={onDecide} busy={deciding} />
      ) : (
        <GateVerdict step={step} />
      )}
    </article>
  );
}

/**
 * A failure in words, not a stack trace.
 *
 * The runner's own failure is already a sentence — "step 'analyze' did not
 * produce: artifacts/findings.md" — so the job here is to say who was working
 * and what was missing, and to leave the raw text available for whoever wants it
 * (M2.2.1, item 6).
 */
function StepError({ step }: { step: StepRow }) {
  const [raw, setRaw] = useState(false);
  const missing = step.produces.filter((entry) => (step.matched[entry] ?? []).length === 0);
  return (
    <div className="step-error">
      <p className="step-error-what">
        <b>{step.persona}</b> could not finish <b>{step.step}</b>
        {missing.length > 0 && (
          <>
            {" "}
            — nothing was written for{" "}
            {missing.map((entry) => (
              <code key={entry} className="mono">
                {entry}
              </code>
            ))}
          </>
        )}
        .
      </p>
      <button className="btn-quiet btn-small" onClick={() => setRaw((v) => !v)}>
        {raw ? "Hide" : "Show"} what the runner reported
      </button>
      {raw && <pre className="step-error-raw">{step.error}</pre>}
    </div>
  );
}

/**
 * What the persona is doing right now, in the minutes before a file appears —
 * run 001's first observation, and why an empty canvas is never a blank pane.
 */
function Activity({ step, calls }: { step: StepRow; calls: number }) {
  if (step.status !== "started") {
    return <span className="detail-activity dim">{calls} tool calls</span>;
  }
  return (
    <span className="detail-activity">
      <span className="working" aria-hidden="true" />
      {step.running > 0 ? "working" : "thinking"} · {calls} tool calls so far
    </span>
  );
}

/** Ticks while the step runs, freezes when it ends. */
function Elapsed({ step }: { step: StepRow }) {
  const now = useClock(step.status === "started");
  const end = step.endedAt ?? now;
  return <>{duration(Math.max(0, end - step.startedAt))}</>;
}
