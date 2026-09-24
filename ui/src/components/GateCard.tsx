import { useState } from "react";
import { api } from "../lib/api";
import { basename } from "../lib/format";
import { useExperience } from "../lib/experience";
import type { AwaitingGate, GateDecision } from "../lib/types";
import { Icon } from "./Icon";

// The one screen where the product asks the user for something. It names the
// stage, quotes what the step asked, lists what that step declares it produces,
// and offers exactly two answers. The state it carries is told by an icon and a
// word as well as by colour, and its accent is the review state — not the brand.

interface Props {
  runId: string;
  gate: AwaitingGate;
  // The step's display title from the workflow shape; the id otherwise.
  stepTitle?: string;
  onAnswered: () => void;
}

export function GateCard({ runId, gate, stepTitle, onAnswered }: Props) {
  const { nouns } = useExperience();
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<GateDecision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const produces = gate.produces ?? [];

  async function answer(decision: GateDecision) {
    setBusy(decision);
    setError(null);
    try {
      await api.gate(runId, decision, note, gate.gate_id);
      onAnswered();
    } catch (e) {
      setError(String(e));
      setBusy(null);
    }
  }

  return (
    <section className="gate-card" aria-labelledby="gate-title">
      <p className="gate-kind">
        <Icon name="clock" size={15} />
        Esperando tu revisión
      </p>
      <h3 id="gate-title">{stepTitle || gate.step}</h3>
      <p className="gate-prompt">{gate.prompt}</p>
      {produces.length > 0 && (
        <div className="gate-produces">
          <span className="gate-produces-label">Lo que produce este paso</span>
          <ul>
            {produces.map((path) => (
              <li key={path}>
                <Icon name="file" size={15} />
                <span title={path}>{basename(path)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {error && (
        <div className="err-banner" role="alert">
          {error}
        </div>
      )}
      <label className="gate-note">
        <span>
          Comentario <span className="faint">Opcional</span>
        </span>
        <textarea
          rows={2}
          placeholder="Añade contexto o indica qué debería cambiar…"
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
      </label>
      <div className="gate-actions">
        <button
          className="btn primary"
          disabled={busy !== null}
          onClick={() => answer("approve")}
        >
          <Icon name="check" size={16} />
          {busy === "approve" ? "Aprobando…" : "Aprobar y continuar"}
        </button>
        <button
          className="btn"
          disabled={busy !== null}
          onClick={() => answer("reject")}
        >
          <Icon name="refresh" size={16} />
          {busy === "reject" ? "Enviando…" : "Solicitar cambios"}
        </button>
      </div>
      <p className="gate-foot">
        Tu respuesta decide si este {nouns.run.singular} continúa.
      </p>
    </section>
  );
}
