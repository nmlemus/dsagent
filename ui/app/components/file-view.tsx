"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { fileUrl } from "../lib/api";

type Kind = "html" | "markdown" | "image" | "table" | "text" | "opaque";

/** What to do with a workspace path, by extension. */
export function kindOf(path: string): Kind {
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  if (ext === "html" || ext === "htm") return "html";
  if (ext === "md" || ext === "markdown") return "markdown";
  if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) return "image";
  if (ext === "csv" || ext === "tsv") return "table";
  if (["txt", "json", "yaml", "yml", "log", "py"].includes(ext)) return "text";
  return "opaque"; // .parquet and friends — see below
}

export function FileView({ runId, path }: { runId: string; path: string }) {
  const url = fileUrl(runId, path);
  const kind = kindOf(path);

  if (kind === "html") {
    // Empty sandbox: no scripts, no same-origin, no forms. Run 002's report is
    // self-contained and needs none of them. A Plotly dashboard will, and that
    // wants `allow-scripts` plus an origin of its own — deferred, not decided here.
    return <iframe className="view-frame" src={url} sandbox="" title={path} />;
  }

  if (kind === "image") {
    // eslint-disable-next-line @next/next/no-img-element -- served by dsagent, not Next
    return <img className="view-image" src={url} alt={path} />;
  }

  if (kind === "opaque") {
    return (
      <div className="view-note">
        <p>
          No preview for <code>{path}</code>.
        </p>
        <p>
          <a href={url} target="_blank" rel="noreferrer">
            Open it directly
          </a>
          . Parquet needs a reader in the browser; the canvas does not ship one yet.
        </p>
      </div>
    );
  }

  return <TextFile url={url} path={path} kind={kind} />;
}

function TextFile({ url, path, kind }: { url: string; path: string; kind: Kind }) {
  const { text, error } = useText(url);

  if (error) return <div className="view-note">Could not load {path}: {error}</div>;
  if (text === null) return <div className="view-note">Loading {path}…</div>;

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
 * on the way into the effect, which costs a cascading render and is what the
 * React lint rule is pointing at.
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
 * That is deliberate for now: this pane exists to let someone glance at a
 * working file mid-run, and a real parser is a dependency to take when something
 * actually depends on being right — reading a quoted comma wrong here misleads
 * nobody into a decision, because the deliverable is the report.
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
        <div className="view-note">
          {hidden.toLocaleString()} more rows not shown.
        </div>
      )}
    </div>
  );
}
