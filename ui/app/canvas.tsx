"use client";

import { useInterrupt } from "@copilotkit/react-core/v2";

import { FileView } from "./file-view";
import { GateCard, gateOf, isGate } from "./gate-card";
import { type RunFile, useRunFiles } from "./use-run-files";

export function Canvas() {
  const { files, focused, focus } = useRunFiles();

  const gate = useInterrupt({
    enabled: (event) => isGate(event.value),
    renderInChat: false,
    render: ({ event, interrupt, resolve }) => {
      const payload = gateOf(event?.value, interrupt);
      if (!payload) return <></>;
      return <GateCard gate={payload} onDecision={(d) => resolve(d)} />;
    },
  });

  const current = files.find((f) => f.path === focused) ?? null;

  return (
    <section className="pane-canvas">
      <header>Canvas</header>

      {/* The gate sits above the files rather than replacing them: the decision
          it asks for is usually about a file in the list below. */}
      {gate && <div className="canvas-gate">{gate}</div>}

      {files.length === 0 && !gate && (
        <div className="empty">Artifacts from a run will appear here.</div>
      )}

      {files.length > 0 && (
        <div className="canvas-split">
          <FileList files={files} focused={focused} onFocus={focus} />
          <div className="canvas-view">
            {current ? (
              <FileView runId={current.run_id} path={current.path} />
            ) : (
              <div className="view-note">Pick a file.</div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function FileList({
  files,
  focused,
  onFocus,
}: {
  files: RunFile[];
  focused: string | null;
  onFocus: (path: string) => void;
}) {
  // Declared deliverables first, then working files — run 001 wrote two files
  // nobody meant to deliver, and a canvas that weights them the same shows a
  // scratch CSV as loudly as the report.
  const deliverables = files.filter((f) => f.kind === "deliverable");
  const working = files.filter((f) => f.kind !== "deliverable");

  return (
    <nav className="canvas-files">
      <Group label="Deliverables" files={deliverables} focused={focused} onFocus={onFocus} />
      <Group label="Working files" files={working} focused={focused} onFocus={onFocus} />
    </nav>
  );
}

function Group({
  label,
  files,
  focused,
  onFocus,
}: {
  label: string;
  files: RunFile[];
  focused: string | null;
  onFocus: (path: string) => void;
}) {
  if (files.length === 0) return null;
  return (
    <>
      <div className="files-label">{label}</div>
      <ul>
        {files.map((f) => (
          <li key={f.path}>
            <button
              className={f.path === focused ? "file is-focused" : "file"}
              onClick={() => onFocus(f.path)}
              title={`${f.path} · ${f.step}`}
            >
              <span className="file-path">{f.path}</span>
              <span className="file-meta">
                {f.step} · {humanSize(f.size)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}

function humanSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
