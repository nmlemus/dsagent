"use client";

import { useEffect, useState } from "react";

import { useRun } from "./run-context";
import type { StepRow } from "./types";

export function DagPanel({ onOpen }: { onOpen: (path: string) => void }) {
  const { steps } = useRun();
  if (steps.length === 0) return null;
  return (
    <div className="dag">
      {steps.map((row) => (
        <StepLine key={row.step} row={row} onOpen={onOpen} />
      ))}
    </div>
  );
}

const MARK: Record<StepRow["status"], string> = {
  started: "◐",
  done: "●",
  failed: "✕",
  awaiting_gate: "⏸",
};

function StepLine({ row, onOpen }: { row: StepRow; onOpen: (path: string) => void }) {
  const { narrationFor } = useRun();
  const [open, setOpen] = useState(false);
  const narration = narrationFor(row.step);
  const tools = Object.entries(row.tools).sort((a, b) => b[1] - a[1]);

  return (
    <div className={`dag-step is-${row.status}`}>
      <div className="dag-head">
        <span className="dag-mark">{MARK[row.status]}</span>
        <span className="dag-name">{row.step}</span>
        <span className="dag-persona">{row.persona}</span>
        <Elapsed row={row} />
      </div>

      <ul className="dag-produces">
        {row.produces.map((entry) => {
          const hits = row.matched[entry] ?? [];
          return (
            <li key={entry} className={hits.length ? "is-met" : "is-unmet"}>
              <span className="dag-tick">{hits.length ? "✓" : "○"}</span>
              {hits.length === 0 ? (
                // Nothing to link to yet. A pattern shows as written, which is
                // the promise: `artifacts/figures/*.png` is not a file.
                <span className="dag-promise">{entry}</span>
              ) : (
                hits.map((path) => (
                  <button key={path} className="dag-file" onClick={() => onOpen(path)}>
                    {path}
                  </button>
                ))
              )}
            </li>
          );
        })}
      </ul>

      {tools.length > 0 && (
        <div className="dag-tools">
          {tools.map(([name, n]) => (
            <span key={name} className="dag-tool">
              {name} <b>{n}</b>
            </span>
          ))}
        </div>
      )}

      {narration.length > 0 && (
        <div className="dag-log">
          <button className="dag-log-toggle" onClick={() => setOpen((v) => !v)}>
            {open ? "▾" : "▸"} {row.persona} {row.status === "started" ? "is working" : "worked"} —{" "}
            {narration.length} note{narration.length === 1 ? "" : "s"}
          </button>
          {open && (
            <div className="dag-log-body">
              {narration.map((n, i) => (
                <p key={i}>{n.text}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {row.error && <div className="dag-error">{row.error}</div>}
    </div>
  );
}

/** Ticks while the step runs, freezes when it ends. */
function Elapsed({ row }: { row: StepRow }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  const live = row.status === "started";

  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(id);
  }, [live]);

  const end = row.endedAt ?? now;
  const seconds = Math.max(0, Math.round(end - row.startedAt));
  return <span className="dag-elapsed">{format(seconds)}</span>;
}

function format(s: number): string {
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}
