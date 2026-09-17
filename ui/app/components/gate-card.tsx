"use client";

import { useState } from "react";

import { fileUrl } from "../lib/api";
import type { StepRow } from "../lib/run-state";
import { basename, duration } from "../lib/format";
import { useClock } from "../lib/use-run";

/**
 * The decision the run is waiting for, at the step that is waiting.
 *
 * It renders from the event log rather than from a live interrupt, which is what
 * lets it appear in a second tab, survive a reload, and still be answerable after
 * the server has been restarted (§7.5, §7.6, §7.11). Answering posts to
 * `POST /runs/{id}/gate`; the run is what holds the question, not this browser.
 */
export function GateCard({
  runId,
  step,
  onDecide,
  busy,
}: {
  runId: string;
  step: StepRow;
  onDecide: (decision: "approve" | "reject", note: string) => void;
  busy: boolean;
}) {
  const [note, setNote] = useState("");
  const now = useClock(true);
  const gate = step.gate;
  if (!gate) return null;

  // The wait is the number that says whether this gate earns its cost — run 003
  // spent 37 seconds here and nothing on the screen said so (M2.2.1, item 5).
  const waiting = gate.asked_at ? now - gate.asked_at : 0;
  const files = Object.values(step.matched).flat();

  return (
    <div className="gate">
      <div className="gate-head">
        <span className="gate-mark" aria-hidden="true" />
        <div>
          <div className="gate-title">{step.persona} needs a decision</div>
          <div className="gate-meta mono">
            {step.step} · waiting {duration(waiting)}
          </div>
        </div>
      </div>

      <p className="gate-message">{gate.prompt}</p>

      {files.length > 0 && (
        <ul className="gate-files">
          {files.map((path) => (
            <li key={path}>
              <a href={fileUrl(runId, path)} target="_blank" rel="noreferrer" className="mono">
                {basename(path)}
              </a>
            </li>
          ))}
        </ul>
      )}

      <label className="gate-note">
        <span className="dim">Note — sent with a rejection, kept in the run’s history</span>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={2}
          placeholder="What needs changing?"
          disabled={busy}
        />
      </label>

      <div className="gate-actions">
        <button
          className="btn btn-primary"
          onClick={() => onDecide("approve", note.trim())}
          disabled={busy}
        >
          Approve and continue
        </button>
        <button className="btn" onClick={() => onDecide("reject", note.trim())} disabled={busy}>
          Send back
        </button>
      </div>
    </div>
  );
}

/** What a rejected gate leaves behind, so the run can be resumed knowingly. */
export function GateVerdict({ step }: { step: StepRow }) {
  const gate = step.gate;
  if (!gate?.decision) return null;
  const waited =
    gate.decided_at && gate.asked_at ? duration(gate.decided_at - gate.asked_at) : null;

  return (
    <div className={`verdict is-${gate.decision}`}>
      <span className="verdict-label">
        {gate.decision === "approve" ? "Approved" : "Sent back"}
        {waited ? ` after ${waited}` : ""}
      </span>
      {gate.note && <p className="verdict-note">“{gate.note}”</p>}
    </div>
  );
}
