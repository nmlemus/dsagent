"use client";

import { useState } from "react";

import type { RunDetail } from "../lib/api";
import { duration, initial, money, percent, tokens } from "../lib/format";
import type { StepRow } from "../lib/run-state";
import { useClock } from "../lib/use-run";
import { GateCard, GateVerdict } from "./gate-card";

/**
 * The progress region: what the team is doing, step by step.
 *
 * The declared `produces` of each step are drawn as promises with a tick, before
 * the files exist — that is the product's own thesis on screen (a workflow is a
 * contract the runner verifies on disk), and it is the one place this screen
 * spends any visual weight.
 */
export function Progress({
  runId,
  detail,
  steps,
  onOpen,
  onDecide,
  deciding,
}: {
  runId: string;
  detail: RunDetail | null;
  steps: StepRow[];
  onOpen: (path: string) => void;
  onDecide: (decision: "approve" | "reject", note: string) => void;
  deciding: boolean;
}) {
  return (
    <section className="progress" aria-label="Run progress">
      <RunHeader detail={detail} steps={steps} />
      <div className="stepper">
        {steps.map((step) => (
          <StepCard
            key={step.step}
            runId={runId}
            step={step}
            onOpen={onOpen}
            onDecide={onDecide}
            deciding={deciding}
          />
        ))}
        {steps.length === 0 && <StepsPending detail={detail} />}
      </div>
    </section>
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
        <Metric label="Waited for you" value={duration(detail?.gate_wait ?? 0)} />
        <Metric
          label="Tokens"
          value={tokens(input)}
          hint={`${input.toLocaleString()} in · ${(usage.output_tokens ?? 0).toLocaleString()} out`}
        />
        <Metric label="Cached" value={percent(cached, input)} hint="Share of input read from cache" />
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

function StepCard({
  runId,
  step,
  onOpen,
  onDecide,
  deciding,
}: {
  runId: string;
  step: StepRow;
  onOpen: (path: string) => void;
  onDecide: (decision: "approve" | "reject", note: string) => void;
  deciding: boolean;
}) {
  const [open, setOpen] = useState(false);
  const pending = step.status === "awaiting_gate" && step.gate?.decision == null;
  const calls = Object.entries(step.tools).sort((a, b) => b[1] - a[1]);
  const total = calls.reduce((n, [, count]) => n + count, 0);

  return (
    <article className={`step is-${step.status}${pending ? " is-waiting" : ""}`}>
      <div className="step-spine">
        <span className="step-mark" title={step.persona}>
          {initial(step.persona)}
        </span>
      </div>

      <div className="step-body">
        <div className="step-head">
          <h3 className="step-name">{step.step}</h3>
          <span className="step-persona dim">{step.persona}</span>
          <Elapsed step={step} />
        </div>

        <Activity step={step} calls={total} />

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

        {step.notes.length > 0 && (
          <div className="notes">
            <button className="btn-quiet btn-small" onClick={() => setOpen((v) => !v)}>
              {open ? "Hide" : "Show"} what {step.persona} said · {step.notes.length}
            </button>
            {open && (
              <div className="notes-body">
                {step.notes.map((note, i) => (
                  <p key={i}>{note.text}</p>
                ))}
              </div>
            )}
          </div>
        )}

        {step.error && <div className="step-error">{step.error}</div>}

        {pending ? (
          <GateCard runId={runId} step={step} onDecide={onDecide} busy={deciding} />
        ) : (
          <GateVerdict step={step} />
        )}
      </div>
    </article>
  );
}

/**
 * What the persona is doing right now, in the first ninety seconds when no file
 * has appeared yet — run 001's first observation, and the reason an empty canvas
 * is not allowed to be a blank pane.
 */
function Activity({ step, calls }: { step: StepRow; calls: number }) {
  if (step.status !== "started") {
    return (
      <p className="step-activity dim">
        {calls > 0 ? `${calls} tool calls` : "no tool calls"}
      </p>
    );
  }
  const busy = step.running > 0;
  return (
    <p className="step-activity">
      <span className="working" aria-hidden="true" />
      {busy ? `working · ${calls} tool calls so far` : `thinking · ${calls} tool calls so far`}
    </p>
  );
}

/** Ticks while the step runs, freezes when it ends. */
function Elapsed({ step }: { step: StepRow }) {
  const now = useClock(step.status === "started");
  const end = step.endedAt ?? now;
  return <span className="step-elapsed mono">{duration(Math.max(0, end - step.startedAt))}</span>;
}

/** Before the first event: the DAG as declared, so the screen is never blank. */
function StepsPending({ detail }: { detail: RunDetail | null }) {
  const steps = detail?.workflow_shape?.steps ?? [];
  if (steps.length === 0) return null;
  return (
    <>
      {steps.map((step) => (
        <article key={step.id} className="step is-pending">
          <div className="step-spine">
            <span className="step-mark">{initial(step.persona)}</span>
          </div>
          <div className="step-body">
            <div className="step-head">
              <h3 className="step-name">{step.id}</h3>
              <span className="step-persona dim">{step.persona}</span>
            </div>
            <p className="step-activity dim">waiting to start</p>
            <ul className="promises">
              {step.produces.map((entry) => (
                <li key={entry} className="is-unmet">
                  <span className="promise mono">{entry}</span>
                </li>
              ))}
            </ul>
          </div>
        </article>
      ))}
    </>
  );
}
