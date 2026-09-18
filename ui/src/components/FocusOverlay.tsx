import { useEffect } from "react";
import { api } from "../lib/api";
import { basename } from "../lib/format";
import { FileViewer } from "./FileViewer";

export type FocusTarget = { kind: "file"; path: string };

// One artifact at full workspace size. The 460px canvas is for glancing; this is
// for reading a table, chart or document. (The full HTML report opens in its own
// browser tab instead — see FileViewer.) Esc or ✕ closes it.
export function FocusOverlay({
  runId,
  target,
  onClose,
}: {
  runId: string;
  target: FocusTarget;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="focus-overlay" role="dialog" aria-label={basename(target.path)}>
      <div className="col-head">
        <span className="row" style={{ gap: 8 }}>
          <span className="mono">{basename(target.path)}</span>
          <a className="btn ghost sm" href={api.fileUrl(runId, target.path)} target="_blank" rel="noreferrer">
            Open raw ↗
          </a>
        </span>
        <button className="icon-btn" onClick={onClose} aria-label="Close" title="Close (Esc)">
          ✕
        </button>
      </div>
      <div className="focus-body">
        <FileViewer runId={runId} path={target.path} full />
      </div>
    </div>
  );
}
