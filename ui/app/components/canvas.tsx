"use client";

import { fileUrl } from "../lib/api";
import { basename, fileSize } from "../lib/format";
import { type RunFile, grouped } from "../lib/run-state";
import { FileView } from "./file-view";

/** The directory part of a workspace path, kept so two `findings.md` differ. */
const dirOf = (path: string): string =>
  path.includes("/") ? path.slice(0, path.lastIndexOf("/") + 1) : "";

/**
 * The canvas: the files a run wrote, as it writes them.
 *
 * Deliverables first, working files after — run 001 wrote two files nobody meant
 * to deliver, and a list that weights them the same shows a scratch CSV as loudly
 * as the report. The list is append-only and focus moves only when a step ends,
 * so a burst of four figures does not drag the pane around.
 */
export function Canvas({
  runId,
  files,
  focused,
  onFocus,
  waiting,
}: {
  runId: string;
  files: RunFile[];
  focused: string | null;
  onFocus: (path: string) => void;
  waiting: React.ReactNode;
}) {
  const current = files.find((f) => f.path === focused) ?? null;

  if (files.length === 0) {
    return (
      <section className="canvas is-empty" aria-label="Files">
        {waiting}
      </section>
    );
  }

  return (
    <section className="canvas" aria-label="Files">
      <nav className="canvas-files">
        {grouped(files).map((group) => (
          <div key={group.label} className="file-group">
            <div className="file-group-label">{group.label}</div>
            <ul>
              {group.files.map((file) => (
                <li key={file.path}>
                  <button
                    className={file.path === focused ? "file is-focused" : "file"}
                    onClick={() => onFocus(file.path)}
                    title={`${file.path} · written by ${file.step}`}
                  >
                    <span className="file-name mono">
                      <span className="file-dir">{dirOf(file.path)}</span>
                      {basename(file.path)}
                    </span>
                    <span className="file-meta">
                      {file.step} · {fileSize(file.size)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <div className="canvas-view">
        {current ? (
          <>
            <header className="view-head">
              <span className="mono">{current.path}</span>
              <a
                className="btn btn-small"
                href={fileUrl(runId, current.path)}
                download={basename(current.path)}
              >
                Download
              </a>
            </header>
            <FileView runId={runId} path={current.path} />
          </>
        ) : (
          <p className="view-note dim">Pick a file.</p>
        )}
      </div>
    </section>
  );
}
