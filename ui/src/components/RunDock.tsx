import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { basename, bytes, cost, duration } from "../lib/format";
import { focusDeliverable, useRunStream } from "../lib/useRunStream";
import type { RunDetail, StepEvent, StepShape } from "../lib/types";
import { FileViewer } from "./FileViewer";
import { FocusOverlay, type FocusTarget } from "./FocusOverlay";
import { GateCard } from "./GateCard";
import { StatusPill } from "./StatusPill";

// The run, docked on the right of the chat: a compact DAG, the gate, and the
// artifacts. The chat stays the home; this opens when a run is selected or
// started. Reuses the run event stream and the file viewers.
export function RunDock({ runId, onClose }: { runId: string; onClose: () => void }) {
  const stream = useRunStream(runId);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [pinned, setPinned] = useState(false);
  const [focus, setFocus] = useState<FocusTarget | null>(null);

  const refresh = useCallback(() => {
    api.run(runId).then(setDetail).catch(() => {});
  }, [runId]);

  useEffect(() => {
    setDetail(null);
    setSelected(null);
    setPinned(false);
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [runId, refresh]);

  useEffect(() => {
    if (stream.lastIndex > 0) refresh();
  }, [stream.lastIndex, refresh]);

  const focusPath = useMemo(() => focusDeliverable(stream.steps), [stream.steps]);
  useEffect(() => {
    if (!pinned && focusPath) setSelected(focusPath);
  }, [focusPath, pinned]);

  const shapeSteps: StepShape[] = detail?.workflow_shape?.steps ?? [];
  const files = [...stream.files].sort((a, b) => a.path.localeCompare(b.path));
  const stopped = detail && ["done", "failed", "stopped"].includes(detail.status);

  return (
    <div className="col rundock">
      <div className="col-head">
        <div style={{ minWidth: 0 }}>
          <div className="mono" style={{ overflow: "hidden", textOverflow: "ellipsis", fontSize: 12 }}>
            {runId}
          </div>
          {detail && (
            <div className="sub">
              {detail.steps_done}/{detail.steps_total} · {cost(detail.cost_usd)} · {duration(detail.duration)}
            </div>
          )}
        </div>
        <div className="row" style={{ gap: 6 }}>
          {detail && <StatusPill status={detail.status} />}
          {detail && !stopped && (
            <button className="btn danger sm" onClick={() => api.stop(runId).then(refresh).catch(() => {})}>
              Stop
            </button>
          )}
          <button className="icon-btn" onClick={onClose} title="Close" aria-label="Close run">
            ✕
          </button>
        </div>
      </div>

      <div className="col-body">
        {detail?.status === "awaiting_gate" && detail.awaiting && (
          <GateCard runId={runId} gate={detail.awaiting} onAnswered={refresh} />
        )}

        {/* compact DAG */}
        <div className="side-label">Steps</div>
        <div className="dag-compact">
          {shapeSteps.map((shape) => {
            const live: StepEvent | undefined = stream.steps[shape.id];
            const status = live?.status ?? "pending";
            return (
              <div key={shape.id} className="step-compact">
                <span className={`dot-s ${status}`} />
                <span className="mono">{shape.id}</span>
                <span className="faint" style={{ fontSize: 11 }}>@{shape.persona}</span>
                {shape.gate && <span className="chip" style={{ padding: "0 5px" }}>gate</span>}
              </div>
            );
          })}
          {shapeSteps.length === 0 && <div className="side-empty">Loading run…</div>}
        </div>

        {/* artifacts */}
        <div className="row spread" style={{ margin: "14px 0 6px" }}>
          <span className="side-label" style={{ margin: 0 }}>Artifacts</span>
          <span className="row" style={{ gap: 6 }}>
            <a className="btn ghost sm" href={api.exportUrl(runId)} target="_blank" rel="noreferrer">
              Report ↗
            </a>
            <a className="btn ghost sm" href={api.downloadUrl(runId)} target="_blank" rel="noreferrer">
              ↓ Zip
            </a>
          </span>
        </div>

        {files.length === 0 ? (
          <div className="side-empty">Files appear here as the run writes them.</div>
        ) : (
          <div className="file-tree">
            {files.map((f) => (
              <button
                key={f.path}
                className={`file-item ${f.kind} ${selected === f.path ? "active" : ""}`}
                onClick={() => {
                  setSelected(f.path);
                  setPinned(true);
                }}
                title={f.path}
              >
                <span className="dot-s" style={{ background: f.kind === "deliverable" ? "var(--ok)" : "var(--fg-faint)" }} />
                <span className="path">{f.path}</span>
                <span className="size">{bytes(f.size)}</span>
              </button>
            ))}
          </div>
        )}

        {selected && (
          <div style={{ marginTop: 12 }}>
            <div className="row spread section-label" style={{ alignItems: "center" }}>
              <span>{basename(selected)}</span>
              <button className="icon-btn" onClick={() => setFocus({ kind: "file", path: selected })} title="Maximize">
                ⤢
              </button>
            </div>
            <FileViewer runId={runId} path={selected} />
          </div>
        )}
      </div>

      {focus && <FocusOverlay runId={runId} target={focus} onClose={() => setFocus(null)} />}
    </div>
  );
}
