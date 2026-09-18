import type { RunStatus } from "../lib/types";

const LABEL: Record<string, string> = {
  pending: "pending",
  running: "running",
  awaiting_gate: "at gate",
  done: "done",
  failed: "failed",
  stopped: "stopped",
};

export function StatusPill({ status }: { status: RunStatus | string }) {
  return (
    <span className={`pill s-${status}`}>
      <span className="dot-s" />
      {LABEL[status] ?? status}
    </span>
  );
}
