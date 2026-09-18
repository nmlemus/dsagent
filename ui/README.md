# DSAgent harness UI

A Vite + React single-page app that drives the `dsagent serve` REST + SSE API.
Run-centric: launch a workflow, watch its DAG run live, answer human gates, and
read the artifacts and living report. A chat pane talks to the orchestrator over
AG-UI and renders A2UI generative widgets.

## How it is served

The production build (`ui/dist/`) is served by `dsagent serve` at `/`, on the
same origin as the API — **one process, one URL, no Node runtime**. The chat pane
talks straight to the FastAPI `/agent` AG-UI endpoint via `@ag-ui/client`
(`HttpAgent`); A2UI surfaces render with `@copilotkit/a2ui-renderer`.

```bash
# 1. Build the UI (writes ui/dist/)
cd ui && npm install && npm run build

# 2. Serve it together with the API
cd .. && dsagent serve                 # → http://localhost:8000
# or, with no model, against a recorded run:
dsagent serve --replay tests/fixtures/replay/run-eda-003
```

`serve.py` mounts `ui/dist` only when it exists (see `_mount_ui`); a checkout
without a build still serves the API. `DSAGENT_UI_DIST` overrides the location.

## Development

```bash
cd ui && npm run dev        # Vite on :5173, proxying /agent /runs /cartridges … to :8000
```

Run `dsagent serve` (or `--replay …`) on :8000 in another terminal; the dev
server proxies the API so there is no CORS story.

## Layout

- `src/lib/` — the typed REST client (`api.ts`), event types (`types.ts`), the
  live-run reducer (`useRunStream.ts`), formatting and the vega detector.
- `src/components/` — `RunList` (rail), `Launcher`, `RunView` (DAG + narration +
  gate), `GateCard`, `Canvas` + `FileViewer` (md/csv/png/html/json + `ChartView`
  for Vega-Lite v5), `Chat` (AG-UI + A2UI).

Heavy libraries are code-split: the vega runtime loads only when a chart is
shown, and the chat/A2UI bundle only when the chat is opened. The core boots at
~77 KB gzip.

## The one invariant to keep

Vega-Lite must match the server's validation grammar: `vega-lite` in
`package.json` and `dsagent.runner.charts.VEGA_LITE` are both v5, and
`tests/test_charts.py` fails if they drift. Move them together.
