"use client";

import type { RunDetail, StepRecord } from "../lib/api";
import { stopRun } from "../lib/api";
import { duration, initial, money, percent, tokens } from "../lib/format";
import type { StepRow } from "../lib/run-state";
import { useClock } from "../lib/use-run";

import { useState } from "react";

/**
 * The team rail: who is working, on what, what it costs, and where it stopped.
 *
 * It is the *activity*, deliberately separate from the document beside it — the
 * one consensus in the research on agentic layouts (part C: "separate the
 * conversation from the activity tracker"). It is also the only dark surface in
 * the product: the document is a document, and the thing watching it work is
 * not competing with it for the reader's eye.
 *
 * Nothing here is a percentage bar. Named phases with elapsed time say what is
 * happening; a bar filling to 62 % says nothing that is true.
 */
export function Rail({
  detail,
  steps,
  openStep,
  onOpenStep,
  onChanged,
  children,
}: {
  detail: RunDetail | null;
  steps: StepRow[];
  openStep: string | null;
  onOpenStep: (step: string | null) => void;
  /** Re-read the run after something on the rail changed it. */
  onChanged: () => void;
  /** The conversation, mounted by the page so the rail does not pull it in. */
  children?: React.ReactNode;
}) {
  const live = detail?.status === "running" || detail?.status === "awaiting_gate";
  const now = useClock(Boolean(live));
  const elapsed =
    detail?.duration ?? (detail?.started_at ? now - detail.started_at : null);

  return (
    <aside className="rail">
      <header className="rail-head">
        <p className="rail-eyebrow">
          {detail?.workflow ?? "run"} · <span className="mono">{detail?.run_id ?? ""}</span>
        </p>
        <h1 className="rail-title">{title(detail)}</h1>
        <p className="rail-sub">{dataset(detail)}</p>
        <p className="rail-status">
          <RunPill detail={detail} />
          <span className="mono rail-elapsed">{duration(elapsed)}</span>
          {detail?.live && <Stop runId={detail.run_id} onStopped={onChanged} />}
        </p>
        <Metrics detail={detail} />
      </header>

      <div className="rail-steps">
        {steps.length === 0 && <p className="rail-empty">The run is about to start.</p>}
        {steps.map((step, i) => (
          <StepItem
            key={step.step}
            step={step}
            record={detail?.steps?.[step.step] ?? null}
            open={openStep === step.step}
            onToggle={() => onOpenStep(openStep === step.step ? null : step.step)}
            next={steps[i + 1]}
            now={now}
          />
        ))}
      </div>

      {children}
    </aside>
  );
}

/** The run's state as one word, the same vocabulary as the home ledger. */
function RunPill({ detail }: { detail: RunDetail | null }) {
  const status = detail?.status ?? "pending";
  const stale = !detail?.live && (status === "running" || status === "awaiting_gate");
  const label =
    status === "awaiting_gate" && detail?.awaiting
      ? "needs you"
      : stale
        ? "interrupted"
        : status === "awaiting_gate"
          ? "sent back"
          : status;
  return (
    <span className={`pill is-${stale ? "stale" : status}`}>
      <i className="pill-dot" />
      {label}
    </span>
  );
}

/**
 * Stop, as a first-class button rather than a thing you do by closing the tab.
 *
 * It takes effect between steps: the step in flight finishes, because a persona
 * holding a kernel and half a written file cannot be ended anywhere else without
 * leaving a workspace nothing can describe. That is also what makes "stopping
 * never costs more than what already ran" (§1.7) true rather than nearly true,
 * and the button says which it is before you press it.
 */
function Stop({ runId, onStopped }: { runId: string; onStopped: () => void }) {
  const [asked, setAsked] = useState(false);
  return (
    <button
      type="button"
      className="rail-stop"
      disabled={asked}
      title="The step running now finishes; the run stops before the next one"
      onClick={() => {
        setAsked(true);
        void stopRun(runId).then(onStopped).catch(() => setAsked(false));
      }}
    >
      {asked ? "stopping…" : "Stop"}
    </button>
  );
}

/**
 * Cost, tokens, cached share and the human wait — the four numbers §1.3 asks for.
 *
 * Cost sits first because that is the complaint the research found loudest about
 * every product in this category: spend that is unattributable and arrives late.
 */
function Metrics({ detail }: { detail: RunDetail | null }) {
  const usage = detail?.usage ?? {};
  const input = usage.input_tokens ?? 0;
  const cached = (usage.cache_read ?? 0) + (usage.cache_creation ?? 0);
  const total = input + (usage.output_tokens ?? 0);
  return (
    <dl className="rail-metrics">
      <Metric label="cost" value={money(detail?.cost_usd)} />
      <Metric label="tokens" value={tokens(total)} title={total ? `${total} tokens` : undefined} />
      <Metric label="cached" value={input ? percent(cached, input) : "—"} />
      <Metric label="waited" value={detail?.gate_wait ? duration(detail.gate_wait) : "—"} />
    </dl>
  );
}

function Metric({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="rail-metric">
      <dt>{label}</dt>
      <dd className="mono" title={title}>
        {value}
      </dd>
    </div>
  );
}

/**
 * One step: its mark, its name, its persona, and how long it has been at it.
 *
 * The running step and the one waiting for a decision are the two that open
 * themselves — those are the two a person opened the page to look at. The rest
 * open on a click, and clicking the row is also how the drawer is pointed at the
 * step's raw detail.
 */
function StepItem({
  step,
  record,
  open,
  onToggle,
  next,
  now,
}: {
  step: StepRow;
  /** The run's own record of this step — cost and tokens are not events. */
  record: StepRecord | null;
  open: boolean;
  onToggle: () => void;
  next: StepRow | undefined;
  now: number;
}) {
  const waiting = step.status === "awaiting_gate" && !step.gate?.decision;
  const running = step.status === "started";
  const state = waiting ? "waiting" : running ? "running" : step.status;
  const elapsed = running ? now - step.startedAt : (step.endedAt ?? 0) - step.startedAt;
  // A failed step opens itself. It is the only thing on the screen that matters
  // once a run stops, and "click to find out why" is not an answer.
  const shown = open || running || waiting || step.status === "failed";

  return (
    <>
      <div className={`rail-step is-${state}${shown ? " is-open" : ""}`}>
        <button type="button" className="rail-step-row" onClick={onToggle} aria-expanded={shown}>
          <span className="rail-mark">{mark(state, step.index + 1)}</span>
          <span className="rail-step-name">{step.step}</span>
          <span className="mono rail-step-elapsed">
            {step.startedAt ? duration(elapsed) : ""}
          </span>
        </button>
        <p className="rail-who">
          <span className="rail-av">{initial(step.persona)}</span>
          {step.persona}
          {running && " · working"}
          {waiting && " · waiting for you"}
        </p>
        {shown && <StepDetail step={step} record={record} />}
      </div>
      {step.status === "done" && next && <HandOff step={step} to={next.persona} />}
    </>
  );
}

/** What the step promised, what it has called, what it thought, and what it cost. */
function StepDetail({ step, record }: { step: StepRow; record: StepRecord | null }) {
  const calls = Object.entries(step.tools).sort((a, b) => b[1] - a[1]);
  const note = step.notes.at(-1);
  return (
    <div className="rail-detail">
      {step.produces.map((entry) => {
        const hits = step.matched[entry] ?? [];
        const state = hits.length > 0 ? "ok" : step.status === "failed" ? "bad" : "waiting";
        return (
          <p key={entry} className={`rail-promise is-${state}`}>
            <i>{state === "ok" ? "✓" : state === "bad" ? "✗" : "○"}</i>
            <span className="mono">{entry}</span>
          </p>
        );
      })}
      {calls.length > 0 && (
        <p className="rail-tools mono">
          {calls.map(([tool, n]) => (
            <span key={tool}>
              {tool} <b>{n}</b>
            </span>
          ))}
        </p>
      )}
      {/* The persona's own working note, folded away. Hiding reasoning entirely
          and dumping the whole trace are both listed as anti-patterns in the
          research; a summary that opens is the third option. */}
      {note && (
        <details className="rail-narration">
          <summary>{step.persona} — working note</summary>
          <p>{note.text}</p>
        </details>
      )}
      {(record?.cost_usd != null || record?.usage?.input_tokens) && (
        <p className="rail-cost mono">
          {money(record?.cost_usd ?? null)} ·{" "}
          {tokens((record?.usage?.input_tokens ?? 0) + (record?.usage?.output_tokens ?? 0))} tokens
        </p>
      )}
      {step.error && <p className="rail-error">{step.error}</p>}
    </div>
  );
}

/**
 * What one persona handed the next.
 *
 * A hand-off is the thing no product in the research showed at all, and it is
 * the cheapest way to make a team read as a team rather than as one model with
 * four names: this is the moment Marie's work becomes what Noel starts from.
 */
function HandOff({ step, to }: { step: StepRow; to: string }) {
  const n = Object.values(step.matched).reduce((sum, hits) => sum + hits.length, 0);
  if (!n) return null;
  return (
    <p className="rail-handoff">
      {n} {n === 1 ? "artifact" : "artifacts"} to {to}
    </p>
  );
}

function mark(state: string, index: number): string {
  if (state === "done") return "✓";
  if (state === "failed") return "✗";
  if (state === "waiting") return "?";
  if (state === "running") return "";
  return String(index);
}

function title(detail: RunDetail | null): string {
  const question = detail?.inputs?.question;
  if (question) return question.length > 90 ? `${question.slice(0, 88)}…` : question;
  return detail?.workflow ?? "Run";
}

function dataset(detail: RunDetail | null): string {
  const path = detail?.inputs?.data_path;
  return path ? path.slice(path.lastIndexOf("/") + 1) : "no dataset";
}
