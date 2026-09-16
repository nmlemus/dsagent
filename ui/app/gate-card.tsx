"use client";

import { useState } from "react";

/**
 * The payload `interrupt()` sends for a human gate — `dsagent/serve.py`'s
 * `gate_payload`, and `docs/ui-slice.md` §3.
 */
export type GatePayload = {
  reason: string;
  message: string;
  run_id: string;
  workflow: string;
  step: string;
  persona: string;
  produces: string[];
};

export const GATE_REASON = "dsagent.gate";

/**
 * Where `dsagent serve` is, as the *browser* sees it.
 *
 * Not `DSAGENT_URL`: that one is read server-side by the CopilotKit route, and
 * may well be a hostname only the Next process can resolve. These links open in
 * the user's own tab, so they need a public origin.
 */
const FILES_ORIGIN =
  process.env.NEXT_PUBLIC_DSAGENT_ORIGIN ?? "http://localhost:8000";

export function fileUrl(runId: string, path: string): string {
  const segments = path.split("/").map(encodeURIComponent).join("/");
  return `${FILES_ORIGIN}/runs/${encodeURIComponent(runId)}/files/${segments}`;
}

/**
 * The legacy `on_interrupt` event delivers its value as a **JSON string**, not an
 * object.
 *
 * This is the whole reason an `enabled` predicate written as
 * `event.value?.reason === "dsagent.gate"` silently never fires: `value` is
 * `"{\"reason\": \"dsagent.gate\", ...}"`, so `.reason` is `undefined` and the
 * card never renders while the run sits at a gate forever. Verified against
 * `copilotkit.LangGraphAGUIAgent` on the default settings.
 */
function asObject(value: unknown): Record<string, any> | null {
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" ? (value as Record<string, any>) : null;
}

/** True when this interrupt is one of ours. Use as `useInterrupt`'s `enabled`. */
export function isGate(eventValue: unknown): boolean {
  return asObject(eventValue)?.reason === GATE_REASON;
}

/**
 * Pull our payload out of whichever interrupt flow delivered it.
 *
 * Legacy `on_interrupt` puts the LangGraph interrupt value in `event.value` (as
 * that JSON string). The standard flow — `emit_interrupt_outcome=True` — puts an
 * AG-UI `Interrupt` there instead, whose lifted fields are `reason` / `message` /
 * `response_schema` and whose remaining keys survive under
 * `metadata.langgraph.raw`. Reading both means the card does not care which one
 * the bridge is configured for.
 */
export function gateOf(
  eventValue: unknown,
  interrupt: unknown,
): GatePayload | null {
  const value = asObject(eventValue);
  for (const candidate of [
    value,
    asObject(value?.metadata?.langgraph?.raw),
    asObject((interrupt as Record<string, any> | null)?.metadata?.langgraph?.raw),
  ]) {
    if (candidate?.reason === GATE_REASON && typeof candidate.step === "string") {
      return candidate as GatePayload;
    }
  }
  return null;
}

export function GateCard({
  gate,
  onDecision,
}: {
  gate: GatePayload;
  onDecision: (payload: { decision: "approve" | "reject"; note?: string }) => void;
}) {
  const [note, setNote] = useState("");
  const [sent, setSent] = useState<"approve" | "reject" | null>(null);

  const decide = (decision: "approve" | "reject") => {
    setSent(decision);
    onDecision(decision === "reject" && note.trim()
      ? { decision, note: note.trim() }
      : { decision });
  };

  return (
    <div className="gate">
      <div className="gate-head">
        <span className="gate-badge">Gate</span>
        <span className="gate-step">{gate.step}</span>
        <span className="gate-persona">{gate.persona}</span>
      </div>

      <p className="gate-message">{gate.message}</p>

      {gate.produces.length > 0 && (
        <>
          <div className="gate-label">Produced by this step</div>
          <ul className="gate-files">
            {gate.produces.map((path) => (
              <li key={path}>
                <a
                  href={fileUrl(gate.run_id, path)}
                  target="_blank"
                  rel="noreferrer"
                >
                  {path}
                </a>
              </li>
            ))}
          </ul>
        </>
      )}

      <textarea
        className="gate-note"
        placeholder="Note (sent with a rejection)"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        disabled={sent !== null}
        rows={2}
      />

      <div className="gate-actions">
        <button
          className="gate-approve"
          onClick={() => decide("approve")}
          disabled={sent !== null}
        >
          {sent === "approve" ? "Approved" : "Approve"}
        </button>
        <button
          className="gate-reject"
          onClick={() => decide("reject")}
          disabled={sent !== null}
        >
          {sent === "reject" ? "Rejected" : "Reject"}
        </button>
      </div>

      <div className="gate-run">{gate.workflow} · {gate.run_id}</div>
    </div>
  );
}
