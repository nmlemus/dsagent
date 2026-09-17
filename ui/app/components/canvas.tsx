"use client";

import { API, fileUrl } from "../lib/api";
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
 *
 * On a finished run the report can take the whole pane: at that point the file
 * list is history and the deliverable is what anyone came for (§2.4).
 */
export function Canvas({
  runId,
  files,
  focused,
  onFocus,
  waiting,
  finished,
  reportFull,
  onReportFull,
}: {
  runId: string;
  files: RunFile[];
  focused: string | null;
  onFocus: (path: string) => void;
  waiting: React.ReactNode;
  finished: boolean;
  reportFull: boolean;
  onReportFull: (full: boolean) => void;
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
    <section className={reportFull ? "canvas is-full" : "canvas"} aria-label="Files">
      {!reportFull && (
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
          {finished && (
            <a
              className="btn btn-small download-all"
              href={`${API}/runs/${encodeURIComponent(runId)}/download`}
            >
              Download all as zip
            </a>
          )}
        </nav>
      )}

      <div className="canvas-view">
        {current ? (
          <>
            <header className="view-head">
              <span className="mono">{current.path}</span>
              <div className="view-actions">
                {finished && (
                  <button className="btn btn-small" onClick={() => onReportFull(!reportFull)}>
                    {reportFull ? "Show files" : "Report"}
                  </button>
                )}
                <a
                  className="btn btn-small"
                  href={fileUrl(runId, current.path)}
                  download={basename(current.path)}
                >
                  Download
                </a>
              </div>
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
