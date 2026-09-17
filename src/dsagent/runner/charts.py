"""Charts and tables a persona *emits*, rather than draws.

A persona that writes a PNG has made a decision nobody downstream can revisit:
the axes are baked, the numbers are gone, and the reader cannot hover, zoom or
ask what a point is. These two tools replace that with a **specification plus a
reference to the data** — a Vega-Lite object the browser renders live, the report
embeds, and a later step can re-emit as a second version.

The harness knows nothing about what a chart *means*. It knows that a spec must
satisfy the Vega-Lite schema, that `data_ref` must name a readable table inside
the run's workspace, and that the pair is worth recording as an event. Which
fields go on which axis, and whether a bar is the right mark, belong to the
persona and to the cartridge's skills.

**Data is never inlined by the model.** `spec["data"]` is rewritten to
`{"name": "table"}` and the rows come from the file at `data_ref`, read by the
browser through the run's preview endpoint. A model that pastes 1,400 rows into
a tool call is a model paying to retype a file it already wrote, and getting
some of the numbers wrong on the way.

**Validation is a loop, not a verdict** (research part B, the VegaChat result:
schema-validate then repair converges). A spec that fails the schema comes back
to the persona as the validator's own message, and it tries again. After
`MAX_ATTEMPTS` the tool stops asking and says so, which is the point at which a
PNG fallback is the honest answer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from dsagent.tabular import TableUnreadable, read_table

CHART_EVENT = "dsagent.chart"
"""A chart or table entered the run's document. One event per emission, so a
second version of the same `chart_id` is a second event and the log keeps both."""

MAX_ATTEMPTS = 3
"""How many times one chart may fail the schema before the tool stops asking.

Three is the number in the spec (§1.4). It is a budget for *repair*, not for
work: a persona that cannot produce a valid spec in three tries is being told
something it does not understand, and a fourth round of the same message costs
tokens without changing that."""

PREVIEW_ROWS = 5
"""How much of the referenced table the tool shows back to the persona.

Enough to confirm the file holds what it thinks it holds — a column renamed by a
`groupby` is the usual mistake — and not so much that the rows it deliberately
did not inline arrive anyway."""


@dataclass
class ChartRecord:
    """One chart or table, as the run records it."""

    chart_id: str
    kind: str
    """`chart` or `table` — the two tools, and the two cards the UI has."""
    title: str
    data_ref: str
    """Workspace-relative path of the table the card reads its rows from."""
    step: str
    persona: str
    section: str = ""
    data_url: str = ""
    """Where a reader fetches those rows. The run's own preview endpoint, so the
    browser, the exported report and a second tab all read one file — and the
    model never had to carry the rows through a tool call."""
    spec: dict[str, Any] | None = None
    """The Vega-Lite spec. `None` for a table, which has no grammar to validate."""
    columns: list[str] = field(default_factory=list)
    rows: int | None = None
    version: int = 1
    """Bumped when the same `chart_id` is emitted again — a revision, in place."""
    ts: float = 0.0

    def as_event(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "kind": self.kind,
            "title": self.title,
            "data_ref": self.data_ref,
            "data_url": self.data_url,
            "step": self.step,
            "persona": self.persona,
            "section": self.section,
            "spec": self.spec,
            "columns": self.columns,
            "rows": self.rows,
            "version": self.version,
        }


@dataclass
class StepContext:
    """Which step is emitting, so a card lands in the right part of the document."""

    step: str = ""
    persona: str = ""
    section: str = ""


VEGA_LITE = "v5.21"
"""The Vega-Lite the smoke test compiles against.

It must be the line the **browser** renders, not the newest one available:
`ui/package.json` pins `vega-lite@5`, and a spec validated against v6 that only
v6 accepts is a spec the reader cannot see. Move the two together or neither.
"""


def validate_spec(spec: Any) -> str:
    """Check a Vega-Lite spec. Empty string when it is valid, else what is wrong.

    Two checks, because one is not enough — and the second one is here because a
    real run proved it.

    **The schema**, through altair: it carries the published Vega-Lite JSON
    Schema and builds the spec into its object model, so a mark that does not
    exist and a channel that is not a channel fail here rather than in the
    reader's browser.

    **The compiler**, through `vl-convert`: the spec is rendered once, headless,
    with no rows and no network. A spec can satisfy the schema and still not be a
    chart — run 1 of this milestone emitted two layered specs with a selection
    `param` at the top level, which Vega-Lite pushes into *every* layer, and
    Vega then refuses with `Duplicate signal name: "zoom_tuple"`. Both passed the
    schema. Both drew nothing. The persona only finds that out if the tool tells
    it, which is what this returns.

    A server without either package skips that check rather than refusing to
    record the chart: the alternative makes the run's document depend on how the
    harness was installed.
    """
    if not isinstance(spec, dict):
        return "spec must be a JSON object (a Vega-Lite specification)"
    if not (spec.get("mark") or spec.get("layer") or spec.get("encoding") or spec.get("spec")):
        return "spec has no `mark`, `layer` or `encoding`: that is not a Vega-Lite chart"
    return _schema_error(spec) or _render_error(spec)


def _schema_error(spec: dict[str, Any]) -> str:
    try:
        import altair as alt
    except ImportError:
        return ""
    try:
        alt.Chart.from_dict(dict(spec))
    except Exception as e:  # noqa: BLE001 — altair raises its own error type; any
        # failure to build the chart is a spec the browser would not render either
        return _first_lines(str(e))
    return ""


def _render_error(spec: dict[str, Any]) -> str:
    """Draw it once, with nothing in it, and see whether Vega will have it."""
    try:
        import vl_convert as vlc
    except ImportError:
        return ""
    try:
        # No rows: the question is whether the *spec* parses, and the rows live
        # in a file the browser fetches. No base URLs either — validating a
        # persona's spec is not a reason for this server to make a request.
        vlc.vegalite_to_svg(
            {**spec, "data": {"values": []}}, vl_version=VEGA_LITE, allowed_base_urls=[]
        )
    except Exception as e:  # noqa: BLE001 — whatever it raises, the chart does not draw
        return _compiler_message(str(e))
    return ""


def _compiler_message(message: str) -> str:
    """The compiler's complaint, without its JavaScript stack.

    `vl-convert` runs Vega in an embedded JS engine, so a failure arrives with
    ten frames of `vega-parser` in it. The first two lines say what is wrong; the
    rest is a stack trace through a library the persona cannot edit.
    """
    lines = [line for line in message.splitlines() if line.strip()]
    useful = [line for line in lines if not line.strip().startswith("at ")]
    return "\n".join(useful[:4]) or _first_lines(message)


def _first_lines(message: str, keep: int = 12) -> str:
    """The head of a validator message.

    Altair prints the offending sub-schema in full, which for a chart is hundreds
    of lines of Vega-Lite grammar. The first lines name the field and the
    expected values, which is what a persona needs to fix it; the rest is the
    whole schema being read back to a model that is paying by the token.
    """
    lines = [line for line in message.splitlines() if line.strip()]
    head = lines[:keep]
    if len(lines) > keep:
        head.append(f"… ({len(lines) - keep} more lines of schema omitted)")
    return "\n".join(head)


def slug(text: str) -> str:
    """A chart id from a title: stable across a repair, and readable in the log."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:48] or "chart"


def chart_tools(
    workspace: Path,
    run_id: str,
    *,
    on_chart: Callable[[ChartRecord], None],
    context: Callable[[], StepContext],
    known: Callable[[str], ChartRecord | None],
) -> list[Any]:
    """`show_chart` and `show_table`, bound to one run.

    `context` is asked at call time rather than at build time because a runner
    builds one set of tools and drives several steps through it: the chart has to
    be attributed to the step that is running *now*.
    """
    attempts: dict[str, int] = {}

    def resolve(data_ref: str) -> Path | str:
        """The workspace file `data_ref` names, or why it cannot be used.

        Contained in the workspace by resolution, not by inspection of the
        string: `..` and an absolute path both land outside and are refused on
        the same rule.
        """
        if not data_ref:
            return "data_ref is required: write the table to a file first, then name it here"
        target = (workspace / data_ref).resolve()
        root = workspace.resolve()
        if root != target and root not in target.parents:
            return f"{data_ref} is outside the run workspace"
        if not target.is_file():
            return f"{data_ref} does not exist — write it with run_python first"
        return target

    def describe(target: Path) -> tuple[list[str], int | None, str]:
        """Columns, row count and a small preview, for the tool's own answer."""
        try:
            columns, head, total = read_table(target, PREVIEW_ROWS)
        except TableUnreadable as e:
            return [], None, f"(rows not read: {e})"
        preview = "\n".join(", ".join(str(c) for c in row) for row in head)
        return columns, total, preview

    def record(
        kind: str, chart_id: str, title: str, data_ref: str, section: str,
        spec: dict[str, Any] | None, columns: list[str], rows: int | None,
    ) -> ChartRecord:
        here = context()
        previous = known(chart_id)
        rec = ChartRecord(
            chart_id=chart_id,
            kind=kind,
            title=title,
            data_ref=data_ref,
            data_url=f"/runs/{run_id}/preview/{data_ref}",
            step=here.step,
            persona=here.persona,
            # The step says which section it writes; the persona may name a
            # different one for a card that belongs further down the report.
            section=section.strip() or here.section,
            spec=spec,
            columns=columns,
            rows=rows,
            version=(previous.version + 1) if previous else 1,
            ts=time.time(),
        )
        on_chart(rec)
        return rec

    @tool
    def show_chart(
        spec: dict,
        data_ref: str,
        title: str,
        chart_id: str = "",
        section: str = "",
    ) -> str:
        """Put an interactive chart in the run's report. Use this for every figure.

        `spec` is a Vega-Lite v5 specification **without data**: write
        `"data": {"name": "table"}` and leave the rows to `data_ref`. `data_ref`
        is a workspace-relative CSV or parquet you have already written (with
        `run_python`) holding exactly the rows the chart draws — aggregate first,
        then reference the aggregate, not the raw dataset.

        `title` is the sentence the reader sees above the chart: say what the
        chart shows, not what it is ("Only rain and snow record precipitation",
        not "Precipitation by category"). `chart_id` names the chart so you can
        emit a corrected version of it later; leave it empty and it is derived
        from the title. `section` is the report section it belongs to, if the
        workflow declares any.

        Make the chart interactive where it helps: `"params"` with a `point`
        select on `pointerover` for hover, `{"select": "interval", "bind":
        "scales"}` for pan and zoom on a quantitative axis, an interval selection
        for a brush the reader can drag. Always give every encoded field a
        `tooltip`.

        If the spec does not satisfy the Vega-Lite schema you get the validator's
        message back — fix the spec and call again with the same `chart_id`.
        Only after three failures fall back to writing a PNG.
        """
        target = resolve(data_ref)
        if isinstance(target, str):
            return f"error: {target}"

        cid = chart_id.strip() or slug(title)
        problem = validate_spec(spec)
        if problem:
            attempts[cid] = attempts.get(cid, 0) + 1
            if attempts[cid] >= MAX_ATTEMPTS:
                return (
                    f"error: this spec has failed the Vega-Lite schema {attempts[cid]} times. "
                    f"Stop repairing it and write the figure as a PNG instead.\n{problem}"
                )
            return (
                f"error: the spec is not valid Vega-Lite (attempt {attempts[cid]} of "
                f"{MAX_ATTEMPTS}). Fix it and call show_chart again with "
                f'chart_id="{cid}".\n{problem}'
            )

        attempts.pop(cid, None)
        columns, rows, preview = describe(target)
        clean = {**spec, "data": {"name": "table"}}
        rec = record("chart", cid, title, data_ref, section, clean, columns, rows)
        return (
            f"chart {rec.chart_id} v{rec.version} is in the report.\n"
            f"data: {data_ref} · {rows if rows is not None else '?'} rows · "
            f"columns: {', '.join(columns) or 'unknown'}\n"
            f"first rows:\n{preview}"
        )

    @tool
    def show_table(
        data_ref: str,
        title: str,
        columns: list[str] | None = None,
        table_id: str = "",
        section: str = "",
    ) -> str:
        """Put a sortable, filterable table in the run's report.

        `data_ref` is a workspace-relative CSV or parquet. The reader gets the
        real table — sort, filter, pivot — so do not paste rows into the report
        yourself and do not pre-truncate the file. `columns` narrows which
        columns are shown, in the order you give them; omit it to show all.

        Use this for anything a reader will want to look *through* — a column
        profile, a check table, a ranked list. Use `show_chart` for anything they
        should see the shape of.
        """
        target = resolve(data_ref)
        if isinstance(target, str):
            return f"error: {target}"

        found, rows, preview = describe(target)
        if rows is None:
            return (
                f"error: {data_ref} is not a table this server can read. "
                f"Write it as CSV or parquet."
            )
        wanted = [c for c in (columns or []) if c]
        unknown = [c for c in wanted if c not in found]
        if unknown:
            return (
                f"error: {data_ref} has no column(s) {', '.join(unknown)}. "
                f"It has: {', '.join(found)}"
            )

        rec = record(
            "table", table_id.strip() or slug(title), title, data_ref, section,
            None, wanted or found, rows,
        )
        return (
            f"table {rec.chart_id} v{rec.version} is in the report.\n"
            f"data: {data_ref} · {rows} rows · columns: {', '.join(rec.columns)}\n"
            f"first rows:\n{preview}"
        )

    return [show_chart, show_table]
