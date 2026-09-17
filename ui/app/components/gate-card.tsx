"use client";

import { useState } from "react";

import type { StepRow } from "../lib/run-state";
import { duration } from "../lib/format";
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
  step,
  onDecide,
  busy,
}: {
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
        {/* One line, growing when it is written in: a note is optional on an
            approval and the reason for a rejection, and neither deserves to push
            the buttons off the bottom of the panel. */}
        <input
          className="gate-note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Add a note — kept in the run’s history"
          disabled={busy}
          aria-label="Note to send with the decision"
        />
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
