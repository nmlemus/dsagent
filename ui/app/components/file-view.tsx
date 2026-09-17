"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { API, fileUrl } from "../lib/api";

type Kind = "html" | "markdown" | "image" | "table" | "sheet" | "text" | "opaque";

/** What to do with a workspace path, by extension. */
export function kindOf(path: string): Kind {
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  if (ext === "html" || ext === "htm") return "html";
  if (ext === "md" || ext === "markdown") return "markdown";
  if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) return "image";
  if (ext === "csv" || ext === "tsv") return "table";
  if (["parquet", "pq"].includes(ext)) return "sheet";
  if (["txt", "json", "yaml", "yml", "log", "py"].includes(ext)) return "text";
  return "opaque";
}

export function FileView({ runId, path }: { runId: string; path: string }) {
  const url = fileUrl(runId, path);
  const kind = kindOf(path);

  if (kind === "html") return <HtmlFile runId={runId} path={path} />;

  if (kind === "image") {
    // eslint-disable-next-line @next/next/no-img-element -- served by dsagent, not Next
    return <img className="view-image" src={url} alt={path} />;
  }

  if (kind === "sheet") return <Sheet runId={runId} path={path} />;

  if (kind === "opaque") {
    return (
      <div className="view-note">
        <p>
          No preview for <code className="mono">{path}</code>.
        </p>
        <p>
          <a href={url} target="_blank" rel="noreferrer">
            Open it directly
          </a>{" "}
          or download it.
        </p>
      </div>
    );
  }

  return <TextFile url={url} path={path} kind={kind} />;
}

/**
 * An HTML artifact, in a frame that cannot do anything.
 *
 * `sandbox=""` is an opaque origin with no scripts: run 002's report is
 * self-contained and needs nothing more. A Plotly dashboard does, so scripts can
 * be turned on deliberately — and when they are, the frame is pointed at
 * `/runs-x/…`, which is the same bytes with a `Content-Security-Policy` that
 * sandboxes them server-side too. `allow-scripts` never travels with
 * `allow-same-origin`; together they are not a sandbox at all (§4.5).
 */
function HtmlFile({ runId, path }: { runId: string; path: string }) {
  const [scripts, setScripts] = useState(false);
  const src = scripts
    ? `${API}/runs-x/${encodeURIComponent(runId)}/files/${path
        .split("/")
        .map(encodeURIComponent)
        .join("/")}`
    : fileUrl(runId, path);

  return (
    <>
      <iframe
        key={src}
        className="view-frame"
        src={src}
        sandbox={scripts ? "allow-scripts" : ""}
        title={path}
      />
      <div className="view-foot">
        {scripts ? (
          <>
            <span className="dim">Scripts are running in an isolated frame.</span>
            <button className="btn btn-small" onClick={() => setScripts(false)}>
              Turn scripts off
            </button>
          </>
        ) : (
          <>
            <span className="dim">Shown without scripts.</span>
            <button className="btn btn-small" onClick={() => setScripts(true)}>
              Run scripts
            </button>
          </>
        )}
      </div>
    </>
  );
}

/**
 * A parquet file, previewed by the server.
 *
 * No browser reads parquet, so M2.2's canvas could only link out of itself. The
 * server has pandas — the cartridge's own kernel env requires it — and 200 rows
 * is what §4.4 asks for.
 */
function Sheet({ runId, path }: { runId: string; path: string }) {
  const [state, setState] = useState<{
    path: string;
    data?: { columns: string[]; rows: unknown[][]; total_rows: number; shown_rows: number };
    error?: string;
  }>({ path: "" });

  useEffect(() => {
    let live = true;
    const url = `${API}/runs/${encodeURIComponent(runId)}/preview/${path
      .split("/")
      .map(encodeURIComponent)
      .join("/")}`;
    fetch(url)
      .then(async (r) => {
        const body = await r.json();
        if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
        return body;
      })
      .then((data) => live && setState({ path, data }))
      .catch((e) => live && setState({ path, error: String(e.message ?? e) }));
    return () => {
      live = false;
    };
  }, [runId, path]);

  const current = state.path === path ? state : null;
  if (!current) return <p className="view-note dim">Reading {path}…</p>;
  if (current.error) {
    return (
      <div className="view-note">
        <p>Could not preview this file: {current.error}</p>
        <p className="dim">Download it to open in something that reads it.</p>
      </div>
    );
  }

  const { columns, rows, total_rows, shown_rows } = current.data!;
  return (
    <div className="view-table">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, r) => (
            <tr key={r}>
              {cells.map((cell, i) => (
                <td key={i} className={cell === null ? "is-null" : undefined}>
                  {cell === null ? "null" : String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {total_rows > shown_rows && (
        <p className="view-note dim">
          {shown_rows.toLocaleString()} of {total_rows.toLocaleString()} rows.
        </p>
      )}
    </div>
  );
}

function TextFile({ url, path, kind }: { url: string; path: string; kind: Kind }) {
  const { text, error } = useText(url);

  if (error) {
    return (
      <div className="view-note">
        Could not load {path}: {error}
      </div>
    );
  }
  if (text === null) return <div className="view-note dim">Loading {path}…</div>;

  if (kind === "markdown") {
    return (
      <div className="view-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>
    );
  }
  if (kind === "table") {
    return <Table text={text} delimiter={path.endsWith(".tsv") ? "\t" : ","} />;
  }
  return <pre className="view-text">{text}</pre>;
}

/**
 * One file's text, keyed by its URL.
 *
 * The URL is part of the state rather than a thing an effect resets, so
 * switching files shows "loading" by *derivation* — a result whose `url` is not
 * the one being asked for is simply not this file's — instead of by a setState
 * on the way into the effect, which costs a cascading render.
 */
function useText(url: string) {
  const [loaded, setLoaded] = useState<{ url: string; text?: string; error?: string }>({
    url: "",
  });

  useEffect(() => {
    let live = true;
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((text) => live && setLoaded({ url, text }))
      .catch((e) => live && setLoaded({ url, error: String(e.message ?? e) }));
    return () => {
      live = false;
    };
  }, [url]);

  const current = loaded.url === url ? loaded : null;
  return { text: current?.text ?? null, error: current?.error ?? null };
}

const MAX_ROWS = 200;

/**
 * A CSV preview, not a CSV parser.
 *
 * Splits on the delimiter and does not understand quoted fields containing it.
 * That is deliberate: this pane exists to let someone glance at a working file
 * mid-run, and reading a quoted comma wrong here misleads nobody into a decision,
 * because the deliverable is the report.
 */
function Table({ text, delimiter }: { text: string; delimiter: string }) {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return <div className="view-note">Empty file.</div>;

  const header = lines[0].split(delimiter);
  const rows = lines.slice(1, 1 + MAX_ROWS).map((l) => l.split(delimiter));
  const hidden = Math.max(0, lines.length - 1 - rows.length);

  return (
    <div className="view-table">
      <table>
        <thead>
          <tr>
            {header.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, r) => (
            <tr key={r}>
              {cells.map((c, i) => (
                <td key={i}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {hidden > 0 && (
        <p className="view-note dim">{hidden.toLocaleString()} more rows not shown.</p>
      )}
    </div>
  );
}
