"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { API } from "../lib/api";
import type { Card } from "../lib/run-state";

import { useAsk, useSelection } from "./ask";

/**
 * A chart and a table the reader can actually use.
 *
 * This is the thing the milestone exists for. A PNG is a decision nobody
 * downstream can revisit — the axes are baked, the numbers are gone, and the
 * reader cannot hover a point, zoom a range, or ask what a category holds. A
 * `show_chart` emission is a Vega-Lite spec plus a reference to the rows, so the
 * card can do all of that, and the spec it renders is the one the run recorded:
 * open "Spec" and you are looking at what the persona actually wrote.
 *
 * Vega is bundled from `node_modules` and loaded on demand. Never a CDN: client
 * data is on this screen, and a script request is a request to somebody else's
 * server with this page's URL on it.
 */

const MAX_ROWS = 2000;
/** The preview endpoint's own ceiling. A card draws an aggregate — five rows,
    or forty-eight — because the persona was told to aggregate first; this is the
    guard for the day one of them references the raw dataset. */

type Row = Record<string, unknown>;

/**
 * The Vega config every chart is drawn with — the Aiuda system, from the server.
 *
 * Fetched rather than kept here, because the exported report has to look like
 * the screen and two hand-kept palettes drift the first time one is edited.
 * `src/dsagent/export.py` holds the values; `GET /chart-theme` serves them.
 * Fetched once for the page, not once per card.
 */
let themeRequest: Promise<Record<string, unknown>> | null = null;

function chartTheme(): Promise<Record<string, unknown>> {
  themeRequest ??= fetch(`${API}/chart-theme`)
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}));
  return themeRequest;
}

/** The card's rows, fetched once from the run's own preview endpoint. */
function useRows(card: Card): { rows: Row[] | null; error: string | null } {
  const [state, setState] = useState<{ ref: string; rows?: Row[]; error?: string }>({ ref: "" });

  const { dataRef, dataUrl, version } = card;
  useEffect(() => {
    let live = true;
    const url =
      `${API}/runs/${encodeURIComponent(runOf({ dataUrl } as Card))}/preview/` +
      `${dataRef.split("/").map(encodeURIComponent).join("/")}?rows=${MAX_ROWS}`;
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((body) => live && setState({ ref: dataRef, rows: shape(body) }))
      .catch((e) => live && setState({ ref: dataRef, error: String(e.message ?? e) }));
    return () => {
      live = false;
    };
    // Not `[card]`: the reducer builds a fresh object every time it runs, and
    // during a replay it runs ten times a second — which had every visible card
    // refetching its rows and re-embedding Vega at 10 Hz. What the rows depend
    // on is the file, the run, and which version of the chart asked for them.
  }, [dataRef, dataUrl, version]);

  const current = state.ref === card.dataRef ? state : null;
  return { rows: current?.rows ?? null, error: current?.error ?? null };
}

/** The run a card's data lives in — the tool wrote the URL, so read it back. */
function runOf(card: Pick<Card, "dataUrl">): string {
  const match = /^\/runs\/([^/]+)\/preview\//.exec(card.dataUrl);
  return match ? decodeURIComponent(match[1]) : "";
}

/**
 * `/preview` answers columns and rows-as-arrays; Vega wants objects.
 *
 * Delimited text arrives as strings, because that is what a CSV holds. A
 * quantitative encoding over strings draws a chart that is silently wrong — the
 * axis sorts lexically and the bars are the wrong height — so a column whose
 * every value parses as a number becomes numbers here. Anything else is left
 * exactly as it came: Vega parses ISO dates itself when the encoding says
 * `temporal`, and guessing at dates is how a column of version numbers becomes
 * a timeline.
 */
function shape(body: { columns: string[]; rows: unknown[][] }): Row[] {
  const numeric = body.columns.map((_, i) => body.rows.every((row) => isNumber(row[i])));
  return body.rows.map((row) => {
    const out: Row = {};
    body.columns.forEach((name, i) => {
      const value = row[i];
      out[name] = numeric[i] && typeof value === "string" ? Number(value) : value;
    });
    return out;
  });
}

function isNumber(value: unknown): boolean {
  if (typeof value === "number") return true;
  if (typeof value !== "string" || value.trim() === "") return false;
  return Number.isFinite(Number(value));
}

// ---------------------------------------------------------------- chart ----

export function ChartCard({
  card,
  onOpen,
}: {
  card: Card;
  onOpen: (what: { kind: "spec" | "step"; id: string }) => void;
}) {
  const { rows, error } = useRows(card);
  const ask = useAsk();
  const { selection: shared, select } = useSelection();
  const host = useRef<HTMLDivElement>(null);
  const [mark, setMark] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const width = useWidth(host);
  // The selection belongs to the conversation, not to this card: it is the
  // reader pointing at part of a picture, and it has to ride with whatever they
  // say next. What is shown here is this card's own, if it is the current one.
  const selection = shared?.chartId === card.chartId ? shared.text : null;

  // The spec on screen: the run's, with the reader's own mark if they changed
  // it, and a width in pixels. The edit is local and mechanical — swapping
  // `mark` is the one change Vega-Lite lets you make without knowing what the
  // chart means.
  const spec = useMemo(
    () => laidOut(withMark(untitled(card.spec), mark), width),
    [card.spec, mark, width],
  );
  const alternatives = useMemo(() => marks(card.spec), [card.spec]);

  useEffect(() => {
    if (!host.current || !rows || !spec || !width) return;
    let live = true;
    let view: { finalize: () => void } | null = null;

    void (async () => {
      try {
        const [{ default: embed }, config] = await Promise.all([
          import("vega-embed"),
          chartTheme(),
        ]);
        if (!live || !host.current) return;
        // Into a fresh element, never into the host itself. Two embeds racing on
        // one node is how a layered spec came back as `Duplicate signal name:
        // "zoom_tuple"`: the second call starts while the first is still
        // compiling, and both write their signals into the same view. Vega's
        // own `finalize` cannot help — there is nothing to finalize yet.
        const surface = document.createElement("div");
        // A definite width *before* vega measures it. Personas write
        // `"width": "container"`, which is the right thing to write, and
        // vega-embed resolves it by measuring this element — which is 0 until
        // layout has run, and 0 is what the first SVG came out as: forty-one
        // marks drawn perfectly at zero pixels. Handing it the width we measured
        // ourselves leaves the spec alone, which matters, because a discrete
        // axis ignores an explicit `width` and lands on an infinite one instead.
        surface.style.width = `${width}px`;
        host.current.replaceChildren(surface);
        const result = await embed(surface, spec as never, {
          actions: false,
          renderer: "svg",
          tooltip: { theme: "light" },
          config: config as never,
        });
        if (!live) {
          result.finalize();
          return;
        }
        view = result;
        await result.view.insert("table", rows).runAsync();
        // Then measure again. A band scale takes its width from the data, and
        // the view was laid out before the rows arrived: with zero categories a
        // discrete axis is zero wide, and `autosize: fit` honours that — the
        // legend drew at full size beside a plot squeezed to nothing. The
        // continuous-axis charts on the same screen were fine, which is what
        // made it look like a spec problem rather than an ordering one.
        await result.view.resize().runAsync();
        listen(result.view, selections(spec), rows, (text) =>
          select(text ? { chartId: card.chartId, title: card.title, text } : null),
        );
      } catch (e) {
        if (live) setFailed(String((e as Error).message ?? e));
      }
    })();

    return () => {
      live = false;
      view?.finalize();
    };
  }, [rows, spec, width, card.chartId, card.title, select]);

  return (
    <figure className="card">
      <figcaption className="card-head">
        <b>{card.title}</b>
        <span className="card-tools">
          {alternatives.length > 1 && (
            <select
              aria-label="Chart type"
              value={mark ?? alternatives[0]}
              onChange={(e) => setMark(e.target.value)}
            >
              {alternatives.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          )}
          <button type="button" onClick={() => onOpen({ kind: "spec", id: card.chartId })}>
            Spec
          </button>
          <button type="button" onClick={() => onOpen({ kind: "step", id: card.step })}>
            How
          </button>
          {ask && (
            <button type="button" className="card-ask" onClick={() => setAsking((a) => !a)}>
              Ask {drewIt(card)} to change this
            </button>
          )}
        </span>
      </figcaption>

      <div className="card-body">
        {error && <p className="view-note">Could not read {card.dataRef}: {error}</p>}
        {failed && <p className="view-note">This chart could not be drawn: {failed}</p>}
        {!rows && !error && <div className="card-skeleton" />}
        <div className="vega" ref={host} />
      </div>

      {/* What should change has to be said. The button used to send "please
          change it", which is a request nobody can act on — the model has no way
          to guess what the reader wants different, and answered with nothing.
          This is the sentence that makes it a request. */}
      {asking && ask && (
        <form
          className="card-asking"
          onSubmit={(e) => {
            e.preventDefault();
            const wanted = new FormData(e.currentTarget).get("wanted");
            if (!String(wanted ?? "").trim()) return;
            ask(change(card, String(wanted), selection));
            setAsking(false);
          }}
        >
          <input
            name="wanted"
            autoFocus
            placeholder={`What should ${drewIt(card)} change? e.g. “add a trend line”`}
            aria-label={`What should ${drewIt(card)} change about ${card.title}?`}
          />
          <button type="submit" className="btn btn-small btn-primary">
            Ask
          </button>
        </form>
      )}

      {selection && <p className="card-selection mono">{selection}</p>}

      <p className="card-foot">
        Vega-Lite{card.version > 1 ? ` · v${card.version}` : ""} · data{" "}
        <span className="mono">{card.dataRef}</span>
        {card.rows != null && ` · ${card.rows} rows`} · hover for values
      </p>
    </figure>
  );
}

/**
 * The selection parameters this spec declares, wherever it declares them.
 *
 * Read off the spec rather than off the view. `view.getState()` lists the
 * signals it considers *state*, and a selection is not among them — enumerating
 * that way attached no listener at all, and a brush the reader dragged reported
 * nothing. The persona wrote the params; they are the authority on what there is
 * to listen to.
 */
function selections(spec: Record<string, unknown>): string[] {
  const own = (spec.params ?? []) as { name?: string; select?: unknown }[];
  const layers = ((spec.layer ?? []) as Record<string, unknown>[]).flatMap(
    (layer) => (layer.params ?? []) as { name?: string; select?: unknown }[],
  );
  return [...own, ...layers]
    .filter((param) => param.select && param.name)
    .map((param) => param.name as string);
}

/**
 * How wide the chart may draw, measured rather than declared.
 *
 * Personas write `"width": "container"`, which is the right thing to write — a
 * chart in a document should be as wide as the document. But vega-embed resolves
 * it by measuring the element at embed time, and the first measurement lands
 * before layout: the SVG came out `width="0"` with all 41 marks inside it,
 * drawing perfectly at zero pixels. Measuring here, and re-measuring when the
 * pane changes size, is deterministic; the drawer opening is exactly when a
 * chart has to be redrawn narrower.
 */
function useWidth(host: React.RefObject<HTMLDivElement | null>): number {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const measure = () => setWidth(Math.round(element.clientWidth));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [host]);

  return width;
}

/**
 * The card owns the width; the spec only says it wants the container's.
 *
 * `"width": "container"` is what a persona should write and what the skill asks
 * for — a chart in a document is as wide as the document. Resolving it is the
 * renderer's job, and vega-embed does it by measuring, which fails twice over: it
 * measures before layout (`width="0"`), and on a *discrete* axis Vega-Lite sizes
 * from steps instead, so "fit" squeezes the plot to nothing while the legend
 * still draws. Both were visible on the same screen.
 *
 * So the number is substituted here, with `autosize: fit` stated rather than
 * inferred. Nothing else in the spec is touched: this is layout, not meaning.
 */
function laidOut(
  spec: Record<string, unknown> | null,
  width: number,
): Record<string, unknown> | null {
  if (!spec || !width || spec.width !== "container") return spec;
  return {
    ...spec,
    width,
    autosize: spec.autosize ?? { type: "fit", contains: "padding" },
  };
}

/**
 * A brush the reader drags becomes context for the next question.
 *
 * Not a filter on the chart — the chart already shows what it shows — but an
 * answer to "which part of this are you asking about?", which is the thing a
 * person points at with a finger and cannot type.
 *
 * Watched through the selection's **store dataset**, not its signal. A signal
 * declared inside a layer belongs to that layer's group, so
 * `view.addSignalListener(name)` on a layered spec raises "unrecognized signal"
 * — which is why the first version drew the brush rectangle perfectly and
 * reported nothing at all. Vega-Lite hoists every selection's `_store` to the
 * top level whatever the spec's shape, so that is what this listens to.
 */
function listen(
  view: {
    addDataListener: (name: string, fn: (n: string, v: unknown) => void) => void;
  },
  names: string[],
  rows: Row[],
  report: (text: string | null) => void,
): void {
  for (const name of names) {
    try {
      view.addDataListener(`${name}_store`, (_data, value) => {
        const entries = (value ?? []) as {
          fields?: { field?: string }[];
          values?: unknown[];
        }[];
        const first = entries[0];
        if (!first?.fields || !first.values) return report(null);
        const described = first.fields
          .map((field, i) => ({ field: field.field, span: first.values?.[i] }))
          .filter((pair) => pair.field && Array.isArray(pair.span) && pair.span.length === 2)
          .map(
            (pair) =>
              `${pair.field} ${format((pair.span as unknown[])[0])} → ` +
              `${format((pair.span as unknown[])[1])}`,
          );
        if (described.length === 0) return report(null);
        report(`selected ${described.join(", ")} · of ${rows.length} rows · sent as context`);
      });
    } catch {
      /* a spec whose param is not a selection has no store; that is not an error */
    }
  }
}

function format(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  const asDate = typeof value === "number" || value instanceof Date ? new Date(value as number) : null;
  if (asDate && !Number.isNaN(asDate.valueOf())) return asDate.toISOString().slice(0, 10);
  return String(value);
}

/**
 * Who to name on the button: the persona who drew the chart.
 *
 * A chart amended from the conversation carries the orchestrator as its author
 * for one version — it is who called the tool — but the reader is looking at
 * noel's chart and should be asking noel. A revision inherits the original's
 * persona now; this covers the ones recorded before it did.
 */
function drewIt(card: Card): string {
  return card.persona && card.persona !== "orchestrator" ? card.persona : "the team";
}

/** The spec without its own title: the card header above it already says it. */
function untitled(spec: Record<string, unknown> | null): Record<string, unknown> | null {
  if (!spec || !("title" in spec)) return spec;
  const { title: _dropped, ...rest } = spec;
  return rest;
}

/**
 * What to say when the reader asks for a different chart.
 *
 * Deliberately specific. The orchestrator has one tool that can change a chart
 * in a finished run — `amend_chart` — and it needs the run, the id, and the file
 * the chart draws from; naming all three is the difference between a new version
 * appearing on the page and a paragraph explaining what one would look like.
 */
function change(card: Card, wanted: string, selection: string | null): string {
  const where = selection ? ` The reader has ${selection.replace(" · sent as context", "")}.` : "";
  return (
    `Change the chart "${card.title}" in run ${runOf(card)}: ${wanted.trim()}\n\n` +
    `Its chart_id is "${card.chartId}" and it draws from ${card.dataRef}.` +
    ` Call read_chart to get the current spec, make that change, and call` +
    ` amend_chart with the same chart_id so the new version replaces this chart` +
    ` rather than adding another.${where}` +
    ` If it needs numbers nobody has computed, say so instead of guessing.`
  );
}

/**
 * The marks this chart could sensibly be.
 *
 * Only for a single-mark spec: a layered chart's marks belong to its layers, and
 * swapping them from a toolbar would be the UI deciding what the chart means.
 */
function marks(spec: Record<string, unknown> | null): string[] {
  if (!spec || spec.layer || spec.facet || spec.concat) return [];
  const current = typeof spec.mark === "string" ? spec.mark : (spec.mark as { type?: string })?.type;
  if (!current) return [];
  const family: Record<string, string[]> = {
    bar: ["bar", "point", "line"],
    line: ["line", "area", "bar", "point"],
    area: ["area", "line", "bar"],
    point: ["point", "bar", "line"],
    circle: ["circle", "point", "bar"],
    tick: ["tick", "point", "bar"],
  };
  return family[current] ?? [current];
}

/**
 * The spec with a different mark, and nothing else touched.
 *
 * The mark's own options travel with its type — `cornerRadiusEnd` means nothing
 * on a line, `point: true` nothing on a bar — so they are replaced rather than
 * merged. A temporal x-axis gets `yearmonth` when it becomes a bar, because
 * one bar per millisecond is not a chart.
 */
function withMark(
  spec: Record<string, unknown> | null,
  mark: string | null,
): Record<string, unknown> | null {
  if (!spec) return null;
  if (!mark) return spec;
  const current = typeof spec.mark === "string" ? spec.mark : (spec.mark as { type?: string })?.type;
  if (mark === current) return spec;

  const shaped: Record<string, unknown> =
    mark === "line"
      ? { type: "line", point: true, strokeWidth: 2 }
      : mark === "area"
        ? { type: "area", line: true, opacity: 0.35 }
        : mark === "bar"
          ? { type: "bar", cornerRadiusEnd: 3 }
          : { type: mark };

  const encoding = { ...((spec.encoding as Record<string, Record<string, unknown>>) ?? {}) };
  const x = encoding.x ? { ...encoding.x } : null;
  if (x && mark === "bar" && x.type === "temporal" && !x.timeUnit) {
    // One bar per millisecond is not a chart. Only ever *added*, and only when
    // the persona left the axis ungrouped: deleting a `timeUnit` they wrote
    // would change what the chart aggregates, which is not what "draw this as
    // points instead" asked for.
    x.timeUnit = "yearmonth";
  }
  if (x) encoding.x = x;
  return { ...spec, mark: shaped, encoding };
}

// ---------------------------------------------------------------- table ----

export function TableCard({ card }: { card: Card }) {
  const { rows, error } = useRows(card);
  const ask = useAsk();
  const [sort, setSort] = useState<{ column: string; desc: boolean } | null>(null);
  const [filter, setFilter] = useState("");
  const [pivot, setPivot] = useState(false);

  const columns = card.columns.length > 0 ? card.columns : Object.keys(rows?.[0] ?? {});
  const shown = useMemo(() => {
    let out = rows ?? [];
    if (filter.trim()) {
      const needle = filter.trim().toLowerCase();
      out = out.filter((row) =>
        columns.some((c) => String(row[c] ?? "").toLowerCase().includes(needle)),
      );
    }
    if (sort) {
      const { column, desc } = sort;
      out = [...out].sort((a, b) => compare(a[column], b[column]) * (desc ? -1 : 1));
    }
    return out;
  }, [rows, columns, filter, sort]);

  const toggle = useCallback(
    (column: string) =>
      setSort((s) => (s?.column === column ? { column, desc: !s.desc } : { column, desc: false })),
    [],
  );

  return (
    <figure className="card">
      <figcaption className="card-head">
        <b>{card.title}</b>
        <span className="card-tools">
          <input
            className="card-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter…"
            aria-label={`Filter ${card.title}`}
          />
          <button type="button" onClick={() => setPivot((p) => !p)}>
            {pivot ? "Table" : "Pivot"}
          </button>
          {ask && (
            <button
              type="button"
              className="card-ask"
              onClick={() =>
                ask(`About the table "${card.title}" (${card.dataRef}): what stands out in it?`)
              }
            >
              Ask {card.persona}
            </button>
          )}
        </span>
      </figcaption>

      <div className="card-body is-flush">
        {error && <p className="view-note">Could not read {card.dataRef}: {error}</p>}
        {!rows && !error && <div className="card-skeleton" />}
        {rows && pivot && <Pivot rows={rows} columns={columns} />}
        {rows && !pivot && (
          <div className="card-scroll">
            <table className="grid">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c}>
                      <button type="button" onClick={() => toggle(c)}>
                        {c}
                        {sort?.column === c ? (sort.desc ? " ↓" : " ↑") : ""}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {shown.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c} className={typeof row[c] === "number" ? "num" : undefined}>
                        {String(row[c] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <p className="card-foot">
        <span className="mono">{card.dataRef}</span>
        {card.rows != null && ` · ${card.rows} rows`}
        {shown.length !== (rows?.length ?? 0) && ` · ${shown.length} shown`} · click a column to
        sort
      </p>
    </figure>
  );
}

/**
 * Perspective's own controls, in this product's clothes.
 *
 * Its "configure" button is the one thing that shows through, and it lives in a
 * shadow root with no `part` and no variable of its own — so the tokens go in as
 * a stylesheet the shadow root adopts. The values are read off the page rather
 * than repeated here: `tokens.css` stays the only place they are decided.
 */
function dress(viewer: HTMLElement): void {
  const root = viewer.shadowRoot;
  if (!root || typeof CSSStyleSheet === "undefined") return;
  const token = (name: string, fallback: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

  const sheet = new CSSStyleSheet();
  sheet.replaceSync(`
    #settings_button {
      font-family: ${token("--ui", "system-ui")};
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .02em;
      color: ${token("--ink-dim", "#4c5670")};
      border: 1px solid ${token("--rule", "rgba(20,40,80,.12)")};
      border-radius: ${token("--radius", "3px")};
      background: ${token("--surface", "#fffdf9")};
      padding: 3px 9px;
    }
    #settings_button:hover {
      color: ${token("--accent", "#e8440a")};
      border-color: ${token("--accent", "#e8440a")};
    }
  `);
  try {
    root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet];
  } catch {
    /* an engine without constructable stylesheets keeps Perspective's own */
  }
}

/**
 * The column worth grouping by: the one with repeats and the fewest of them.
 *
 * A pivot exists to collapse rows together. A column whose every value is
 * distinct collapses nothing, and a table where *no* column repeats — a profile,
 * one row per column — has nothing to group at all, so it opens flat.
 */
function groupable(rows: Row[], columns: string[]): string | null {
  const counted = columns
    .map((name) => ({ name, distinct: new Set(rows.map((r) => String(r[name] ?? ""))).size }))
    .filter((c) => c.distinct > 1 && c.distinct < rows.length)
    .sort((a, b) => a.distinct - b.distinct);
  return counted[0]?.name ?? null;
}

function compare(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a ?? "").localeCompare(String(b ?? ""), undefined, { numeric: true });
}

/**
 * Pivot, loaded only when it is asked for.
 *
 * Perspective is a WASM engine of several megabytes — the price of the only
 * open-source pivot table that does group-by, filter and expressions in the
 * browser. Nobody pays it until they press the button, and a build without it
 * says so rather than showing an empty pane.
 */
function Pivot({ rows, columns }: { rows: Row[]; columns: string[] }) {
  const host = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    let viewer: HTMLElement | null = null;

    void (async () => {
      try {
        // The *inline* builds: Perspective's engine is WebAssembly, and the
        // default entry points fetch their `.wasm` beside themselves — which
        // this app does not serve, and which nothing here may fetch from a CDN.
        // The inline bundles carry the WebAssembly inside the JavaScript, so a
        // pivot needs no asset route and no network. They cost ~7 MB, which is
        // why nothing loads them until this button is pressed.
        const perspective = (
          await import("@finos/perspective/dist/esm/perspective.inline.js")
        ).default;
        await import("@finos/perspective-viewer/dist/esm/perspective-viewer.inline.js");
        await import("@finos/perspective-viewer-datagrid");
        // Perspective's own light theme, so the pivot is not the one dark,
        // unstyled control on a cream page. Loaded with the engine, not with the
        // app: nobody who never pivots downloads it.
        await import("@finos/perspective-viewer/dist/css/pro.css");
        if (!live || !host.current) return;
        const worker = await perspective.worker();
        const table = await worker.table(rows as never);
        viewer = document.createElement("perspective-viewer");
        viewer.setAttribute("theme", "Pro Light");
        host.current.replaceChildren(viewer);
        dress(viewer);
        await (viewer as never as { load: (t: unknown) => Promise<void> }).load(table);
        // Group by the first column that repeats, not simply the first one. A
        // profile table's first column is one row per column name, so grouping
        // by it produces a count of 1 against every row — a pivot that pivots
        // nothing, which is what the demo screenshot showed.
        const by = groupable(rows, columns);
        await (viewer as never as { restore: (c: unknown) => Promise<void> }).restore({
          group_by: by ? [by] : [],
          columns: columns.filter((c) => c !== by),
          aggregates: {},
        });
      } catch (e) {
        if (live) setFailed(String((e as Error).message ?? e));
      }
    })();

    return () => {
      live = false;
      viewer?.remove();
    };
  }, [rows, columns]);

  if (failed) {
    return (
      <p className="view-note">
        Pivot needs Perspective, which is not available in this build: {failed}
      </p>
    );
  }
  return <div className="pivot" ref={host} />;
}
