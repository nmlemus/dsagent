import { lazy, Suspense, useEffect, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import type { FileSource } from "../lib/api";
import { ext } from "../lib/format";
import { isVegaLite } from "../lib/vega-detect";

// One viewer for every file the product shows, wherever it lives: a run's
// workspace or a project's chat workspace. It is given a `source` (how to build
// a URL and how to ask for a table preview) rather than a run id, so a report a
// persona wrote in a conversation reads exactly like one a run produced.

// The vega runtime is heavy; load it only when a chart is actually shown.
const ChartView = lazy(() => import("./ChartView"));

const IMG = new Set(["png", "jpg", "jpeg", "gif", "svg", "webp", "avif"]);
const TABLE = new Set(["csv", "tsv", "parquet"]);
const TEXT = new Set(["txt", "log", "py", "yaml", "yml", "toml", "sh"]);

export function FileViewer({
  source,
  path,
  full = false,
}: {
  source: FileSource;
  path: string;
  full?: boolean;
}) {
  const kind = ext(path);
  const vc = full ? "viewer full" : "viewer";

  if (IMG.has(kind)) {
    return (
      <div className={vc}>
        <img src={source.url(path)} alt={path} />
      </div>
    );
  }
  if (kind === "html") {
    // The living report is a full interactive document (vegaEmbed charts). An
    // in-app iframe can't paint it reliably (a Chrome bug where an iframe under a
    // composited page loads but never paints — see the decisions log), and a
    // report deserves the whole window anyway. So we open it as its own page.
    return (
      <div className="viewer" style={{ padding: 24, textAlign: "center" }}>
        <p className="muted" style={{ marginTop: 0 }}>
          Documento completo con contenido interactivo.
        </p>
        <a
          className="btn primary"
          href={source.url(path)}
          target="_blank"
          rel="noreferrer"
        >
          Abrir documento completo ↗
        </a>
      </div>
    );
  }
  if (TABLE.has(kind))
    return <TablePreview source={source} path={path} full={full} />;
  if (kind === "md")
    return <Markdown source={source} path={path} full={full} />;
  if (kind === "json") return <JsonFile source={source} path={path} />;
  if (TEXT.has(kind) || kind === "")
    return <TextFile source={source} path={path} />;

  return (
    <div className="viewer">
      <div className="empty">
        Vista previa no disponible para “.{kind}”.{" "}
        <a href={source.url(path)} target="_blank" rel="noreferrer">
          Abrir archivo
        </a>
      </div>
    </div>
  );
}

function useText(source: FileSource, path: string) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const url = source.url(path);
  useEffect(() => {
    let live = true;
    setText(null);
    setError(null);
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${r.status}`))))
      .then((t) => live && setText(t))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [url]);
  return { text, error };
}

function Markdown({
  source,
  path,
  full,
}: {
  source: FileSource;
  path: string;
  full?: boolean;
}) {
  const { text, error } = useText(source, path);
  if (error)
    return (
      <div className="viewer">
        <div className="empty">{error}</div>
      </div>
    );
  if (text === null)
    return (
      <div className="viewer">
        <div className="empty">Cargando…</div>
      </div>
    );
  const html = DOMPurify.sanitize(
    marked.parse(text, { async: false }) as string,
  );
  return (
    <div className={full ? "viewer full" : "viewer"}>
      <div className="md" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

function JsonFile({ source, path }: { source: FileSource; path: string }) {
  const { text, error } = useText(source, path);
  if (error)
    return (
      <div className="viewer">
        <div className="empty">{error}</div>
      </div>
    );
  if (text === null)
    return (
      <div className="viewer">
        <div className="empty">Cargando…</div>
      </div>
    );
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return (
      <div className="viewer">
        <pre>{text}</pre>
      </div>
    );
  }
  // A Vega-Lite spec renders as an interactive chart (vega-lite v5, matching the
  // server's validation grammar); any other JSON is shown pretty-printed.
  if (isVegaLite(parsed))
    return (
      <Suspense
        fallback={
          <div className="viewer">
            <div className="empty">Preparando gráfico…</div>
          </div>
        }
      >
        <ChartView spec={parsed} />
      </Suspense>
    );
  return (
    <div className="viewer">
      <pre>{JSON.stringify(parsed, null, 2)}</pre>
    </div>
  );
}

function TextFile({ source, path }: { source: FileSource; path: string }) {
  const { text, error } = useText(source, path);
  if (error)
    return (
      <div className="viewer">
        <div className="empty">{error}</div>
      </div>
    );
  return (
    <div className="viewer">
      <pre>{text ?? "Cargando…"}</pre>
    </div>
  );
}

function TablePreview({
  source,
  path,
  full,
}: {
  source: FileSource;
  path: string;
  full?: boolean;
}) {
  const [data, setData] = useState<{
    columns: string[];
    rows: unknown[][];
    truncated?: boolean;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    source
      .preview(path)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
    // `source` is rebuilt on each render by its owner; the pair that identifies
    // the request is the URL and the path.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source.url(path), path]);

  if (error)
    return (
      <div className="viewer">
        <div className="empty">{error}</div>
      </div>
    );
  if (!data)
    return (
      <div className="viewer">
        <div className="empty">Preparando vista previa…</div>
      </div>
    );
  return (
    <div
      className="viewer"
      style={{ overflowX: "auto", maxHeight: full ? "none" : "60vh" }}
    >
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
      {data.truncated && (
        <div className="faint" style={{ padding: 8 }}>
          Vista previa parcial.
        </div>
      )}
    </div>
  );
}
