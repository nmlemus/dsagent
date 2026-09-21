import { useState } from "react";
import { api } from "../lib/api";
import type { AwaitingGate, GateDecision } from "../lib/types";

interface Props {
  runId: string;
  gate: AwaitingGate;
  // The step's display title from the workflow shape; the id otherwise.
  stepTitle?: string;
  onAnswered: () => void;
}

export function GateCard({ runId, gate, stepTitle, onAnswered }: Props) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<GateDecision | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function answer(decision: GateDecision) {
    setBusy(decision);
    setError(null);
    try {
      await api.gate(runId, decision, note);
      onAnswered();
    } catch (e) {
      setError(String(e));
      setBusy(null);
    }
  }

  return (
    <div className="gate-card">
      <h4>Revisión · {stepTitle || gate.step}</h4>
      <p style={{ margin: 0 }}>{gate.prompt}</p>
      {gate.produces && gate.produces.length > 0 && (
        <div className="produces">
          {gate.produces.map((p) => (
            <span className="chip" key={p}>
              {p}
            </span>
          ))}
        </div>
      )}
      {error && (
        <div className="err-banner" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}
      <textarea
        rows={2}
        aria-label="Comentario para la revisión"
        placeholder="Añade contexto o indica qué debería cambiar…"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        style={{ marginTop: 10 }}
      />
      <div className="gate-actions">
        <button
          className="btn approve"
          disabled={busy !== null}
          onClick={() => answer("approve")}
        >
          {busy === "approve" ? "Aprobando…" : "Aprobar y continuar"}
        </button>
        <button
          className="btn reject"
          disabled={busy !== null}
          onClick={() => answer("reject")}
        >
          {busy === "reject" ? "Enviando…" : "Solicitar cambios"}
        </button>
      </div>
    </div>
  );
}
