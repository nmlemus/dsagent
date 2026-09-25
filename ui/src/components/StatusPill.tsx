import type { RunStatus } from "../lib/types";

const LABEL: Record<string, string> = {
  pending: "Por iniciar",
  running: "En curso",
  awaiting_gate: "Por revisar",
  done: "Completado",
  failed: "Con errores",
  stopped: "Detenido",
  interrupted: "Interrumpido",
  cancelling: "Cancelando",
  cleanup_required: "Requiere limpieza",
};

export function StatusPill({ status }: { status: RunStatus | string }) {
  return (
    <span className={`pill s-${status}`}>
      <span className="dot-s" />
      {LABEL[status] ?? status}
    </span>
  );
}
