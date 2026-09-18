import { lazy, Suspense, useEffect, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { api } from "../lib/api";
import { ext } from "../lib/format";
import { isVegaLite } from "../lib/vega-detect";

// The vega runtime is heavy; load it only when a chart is actually shown.
const ChartView = lazy(() => import("./ChartView"));

const IMG = new Set(["png", "jpg", "jpeg", "gif", "svg", "webp", "avif"]);
const TABLE = new Set(["csv", "tsv", "parquet"]);
const TEXT = new Set(["txt", "log", "py", "yaml", "yml", "toml", "sh"]);

export function FileViewer({
  runId,
  path,
  full = false,
}: {
  runId: string;
  path: string;
  full?: boolean;
}) {
  const kind = ext(path);
  const vc = full ? "viewer full" : "viewer";

  if (IMG.has(kind)) {
    return (
      <div className={vc}>
        <img src={api.fileUrl(runId, path)} alt={path} />
      </div>
    );
  }
  if (kind === "html") {
    // The living report is a full interactive document (vegaEmbed charts). An
    // in-app iframe can't paint it reliably (a Chrome bug where an iframe under a
    // composited page loads but never paints — see the decisions log), and a
    // report deserves the whole window anyway. So we open it as its own page.
    const url = api.fileUrl(runId, path);
    return (
      <div className="viewer" style={{ padding: 24, textAlign: "center" }}>
        <p className="muted" style={{ marginTop: 0 }}>
          This is a full HTML report with interactive charts.
        </p>
        <a className="btn primary" href={url} target="_blank" rel="noreferrer">
          Open report in a new tab ↗
        </a>
      </div>
    );
  }
  if (TABLE.has(kind)) return <TablePreview runId={runId} path={path} full={full} />;
  if (kind === "md") return <Markdown runId={runId} path={path} full={full} />;
  if (kind === "json") return <JsonFile runId={runId} path={path} />;
  if (TEXT.has(kind) || kind === "") return <TextFile runId={runId} path={path} />;

  return (
    <div className="viewer">
      <div className="empty">
        No inline viewer for “.{kind}”.{" "}
        <a href={api.fileUrl(runId, path)} target="_blank" rel="noreferrer">
          Open raw
        </a>
      </div>
    </div>
  );
}

function useText(runId: string, path: string) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setText(null);
    setError(null);
    fetch(api.fileUrl(runId, path))
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${r.status}`))))
      .then((t) => live && setText(t))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [runId, path]);
  return { text, error };
}

function Markdown({ runId, path, full }: { runId: string; path: string; full?: boolean }) {
  const { text, error } = useText(runId, path);
  if (error) return <div className="viewer"><div className="empty">{error}</div></div>;
  if (text === null) return <div className="viewer"><div className="empty">Loading…</div></div>;
  const html = DOMPurify.sanitize(marked.parse(text, { async: false }) as string);
  return (
    <div className={full ? "viewer full" : "viewer"}>
      <div className="md" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

function JsonFile({ runId, path }: { runId: string; path: string }) {
  const { text, error } = useText(runId, path);
  if (error) return <div className="viewer"><div className="empty">{error}</div></div>;
  if (text === null) return <div className="viewer"><div className="empty">Loading…</div></div>;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return <div className="viewer"><pre>{text}</pre></div>;
  }
  // A Vega-Lite spec renders as an interactive chart (vega-lite v5, matching the
  // server's validation grammar); any other JSON is shown pretty-printed.
  if (isVegaLite(parsed))
    return (
      <Suspense fallback={<div className="viewer"><div className="empty">Loading chart…</div></div>}>
        <ChartView spec={parsed} />
      </Suspense>
    );
  return (
    <div className="viewer">
      <pre>{JSON.stringify(parsed, null, 2)}</pre>
    </div>
  );
}

function TextFile({ runId, path }: { runId: string; path: string }) {
  const { text, error } = useText(runId, path);
  if (error) return <div className="viewer"><div className="empty">{error}</div></div>;
  return (
    <div className="viewer">
      <pre>{text ?? "Loading…"}</pre>
    </div>
  );
}

function TablePreview({ runId, path, full }: { runId: string; path: string; full?: boolean }) {
  const [data, setData] = useState<{ columns: string[]; rows: unknown[][]; truncated?: boolean } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    api
      .preview(runId, path)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [runId, path]);

  if (error) return <div className="viewer"><div className="empty">{error}</div></div>;
  if (!data) return <div className="viewer"><div className="empty">Loading preview…</div></div>;
  return (
    <div className="viewer" style={{ overflowX: "auto", maxHeight: full ? "none" : "60vh" }}>
      <table className="preview">
        <thead>
          <tr>
            {data.columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell == null ? "" : String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.truncated && <div className="faint" style={{ padding: 8 }}>Preview truncated.</div>}
    </div>
  );
}
