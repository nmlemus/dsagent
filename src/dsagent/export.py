"""The run as one file somebody else can open.

A report that stops being interactive the moment it leaves the product is a
screenshot with extra steps. This builds a single self-contained HTML document —
the run's own sections, its charts as **live Vega-Lite views**, and the rows they
draw, snapshotted — that works from a file:// URL, offline, with no CDN and no
server behind it.

The JavaScript is `vl-convert`'s bundle, extracted once and reused for every
chart in the document. That is why it is a megabyte: a chart you can still hover,
zoom and read the axes of, on a laptop with no network, is worth a megabyte.

Nothing here knows what a chart *means*, and nothing here writes prose. Every
word in the export was written by a persona; every number was read off a file the
run produced.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from dsagent.tabular import TableUnreadable, read_table

SNAPSHOT_ROWS = 5000
"""How many rows of each chart's table travel with the export.

The whole point of `data_ref` is that rows do not ride along; an export has no
server to fetch them from, so here they must. Five thousand is far above what a
persona's aggregate holds and far below what makes a file unopenable."""

CHART_SCRIPT = re.compile(r"<script type=\"text/javascript\">(.*?)</script>", re.DOTALL)


def vega_bundle(vl_version: str) -> str:
    """The Vega/Vega-Lite/vega-embed bundle, as one script body.

    `vl-convert` will produce a self-contained document *per chart*; a report has
    several, and nesting documents is not a thing. So one probe document is
    built and its first script — the libraries, which end by assigning
    `window.vegaEmbed` — is lifted out and reused.
    """
    import vl_convert as vlc

    probe = {
        "$schema": f"https://vega.github.io/schema/vega-lite/{vl_version[:2]}.json",
        "data": {"values": []},
        "mark": "point",
        "encoding": {},
    }
    document = vlc.vegalite_to_html(probe, vl_version=vl_version, bundle=True)
    first = CHART_SCRIPT.search(document)
    if first is None:  # pragma: no cover — vl-convert always emits one
        raise RuntimeError("vl-convert produced no script to bundle")
    return first.group(1)


def snapshot(workspace: Path, data_ref: str) -> list[dict[str, Any]]:
    """The rows a chart draws, as objects, read where the run left them."""
    target = workspace / data_ref
    try:
        columns, rows, _ = read_table(target, SNAPSHOT_ROWS)
    except (TableUnreadable, OSError):
        return []
    return [dict(zip(columns, row)) for row in rows]


def markdown_to_html(text: str) -> str:
    """Markdown as HTML, through the renderer the cartridge already installs.

    Falls back to preformatted text rather than failing the export: a report you
    can read as plain text beats a download that 500s because a package that
    renders headings is missing.
    """
    try:
        import markdown
    except ImportError:
        return f"<pre>{html.escape(text)}</pre>"
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def export_html(run_dir: Path, state: dict[str, Any], *, vl_version: str) -> str:
    """One document: what the run was asked, what it found, and the charts.

    Sections come from the steps in order, each showing the markdown artifact it
    produced — the same rule the live document follows, so the export is the
    screen rather than a second telling of it.
    """
    workspace = run_dir / "workspace"
    charts = list(state.get("charts", {}).values())
    inputs = state.get("inputs") or {}
    title = str(inputs.get("question") or state.get("workflow") or run_dir.name)

    mounts: list[str] = []
    body: list[str] = []
    for index, step in enumerate(state.get("steps", {}).values()):
        section = _section(workspace, step, [c for c in charts if c.get("step") == step.get("id")],
                           index, mounts)
        if section:
            body.append(section)

    return DOCUMENT.format(
        title=html.escape(title),
        run_id=html.escape(run_dir.name),
        meta=html.escape(_meta(state, inputs)),
        bundle=vega_bundle(vl_version),
        body="\n".join(body),
        mounts="\n".join(mounts),
    )


def _meta(state: dict[str, Any], inputs: dict[str, Any]) -> str:
    people = sorted({s.get("id", "") for s in state.get("steps", {}).values()})
    cost = sum(s.get("cost_usd") or 0 for s in state.get("steps", {}).values())
    data = str(inputs.get("data_path") or "")
    bits = [state.get("workflow", ""), data, f"{len(people)} steps"]
    if cost:
        bits.append(f"${cost:.2f}")
    return " · ".join(b for b in bits if b)


def _section(workspace: Path, step: dict[str, Any], charts: list[dict[str, Any]],
             index: int, mounts: list[str]) -> str:
    """One step: its heading, the file it wrote, and the charts it emitted."""
    prose = ""
    for record in step.get("files", []):
        path = record.get("path", "")
        if path.endswith(".md"):
            try:
                prose = markdown_to_html((workspace / path).read_text(encoding="utf-8"))
            except OSError:
                prose = ""
            break

    figures = []
    for chart in charts:
        if chart.get("kind") != "chart" or not chart.get("spec"):
            continue
        mount = f"chart-{len(mounts)}"
        rows = snapshot(workspace, chart.get("data_ref", ""))
        mounts.append(MOUNT.format(
            mount=mount,
            spec=json.dumps(chart["spec"]),
            rows=json.dumps(rows),
        ))
        figures.append(FIGURE.format(
            mount=mount,
            title=html.escape(str(chart.get("title", ""))),
            source=html.escape(str(chart.get("data_ref", ""))),
            rows=len(rows),
        ))

    if not prose and not figures:
        return ""
    heading = html.escape(str(step.get("id", f"step {index + 1}")))
    return f'<section><h2>{index + 1}. {heading}</h2>{prose}{"".join(figures)}</section>'


FIGURE = """
<figure class="chart">
  <figcaption>{title}</figcaption>
  <div id="{mount}"></div>
  <p class="source">Vega-Lite · {source} · {rows} rows travelling with this file</p>
</figure>
"""

MOUNT = """
{{
  const spec = {spec};
  const rows = {rows};
  const host = document.getElementById('{mount}');
  // The same two traps the run screen fell into, and the same two answers:
  // `"container"` is resolved by measuring, which happens before layout, so the
  // width is substituted here; and a band scale takes its width from the data,
  // so the view is measured again once it is drawn. Without either, the plot
  // collapses to nothing beside a full-size legend.
  const withData = {{
    ...spec,
    data: {{values: rows}},
    width: Math.max(320, host.clientWidth - 24),
    autosize: spec.autosize || {{type: 'fit', contains: 'padding'}},
  }};
  vegaEmbed(host, withData, {{actions: false, renderer: 'svg', tooltip: true}})
    .then((result) => result.view.resize().runAsync())
    .catch(console.error);
}}
"""

DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: #faf8f4; color: #142850;
         font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 820px; margin: 0 auto; padding: 40px 24px 80px; }}
  h1 {{ font-size: 32px; line-height: 1.15; margin: 0 0 6px; }}
  h2 {{ font-size: 24px; margin: 40px 0 12px; font-weight: 400; }}
  h3 {{ font-size: 16px; margin: 24px 0 8px; }}
  p, li {{ color: rgba(20,40,80,.78); }}
  .meta {{ color: rgba(20,40,80,.45); font-size: 13px; margin: 0 0 32px; }}
  code, .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  table {{ border-collapse: collapse; font-size: 13px; display: block; overflow-x: auto; }}
  th, td {{ border-bottom: 1px solid rgba(20,40,80,.12); padding: 6px 10px; text-align: left; }}
  th {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
        color: rgba(20,40,80,.45); }}
  figure.chart {{ margin: 20px 0 28px; background: #fffdf9;
                  border: 1px solid rgba(20,40,80,.12); border-radius: 6px; overflow: hidden; }}
  figure.chart figcaption {{ padding: 10px 14px; border-bottom: 1px solid rgba(20,40,80,.12);
                             font-weight: 600; font-size: 14px; }}
  figure.chart > div {{ padding: 10px 12px; }}
  .source {{ margin: 0; padding: 6px 14px; border-top: 1px solid rgba(20,40,80,.12);
             background: #f4f1ea; font-size: 11.5px; color: rgba(20,40,80,.45);
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .vega-embed summary {{ display: none; }}
</style>
<script type="text/javascript">{bundle}</script>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="meta">{meta} · <span class="mono">{run_id}</span></p>
{body}
</main>
<script type="text/javascript">
{mounts}
</script>
</body>
</html>
"""
