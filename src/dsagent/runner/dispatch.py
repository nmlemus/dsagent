"""Adapter: runner events → LangChain custom events.

This is the entire seam between the runner and the wire. `WorkflowRunner` knows
nothing about LangChain, AG-UI or HTTP; it hands `RunnerEvent`s to `on_event`.
This module turns one into a LangChain custom event, and `ag-ui-langgraph` turns
that into an AG-UI `CUSTOM` event with the same `name` and `value`
(`agent.py:3629`). Nothing in between reshapes the payload, so the schemas in
`docs/ui-slice.md` §3 are what a browser receives.

**`dispatch_custom_event`, not `get_stream_writer`.** The bridge consumes
`graph.astream_events()` and never passes `stream_mode`, so writer output — which
only reaches `.astream(stream_mode="custom")` — is invisible to it. Verified on
langgraph 1.2.11; see `docs/ui-slice.md` §1.
"""

from __future__ import annotations

from langchain_core.callbacks.manager import dispatch_custom_event

from dsagent.runner.runner import RunnerEvent


def dispatch_runner_event(event: RunnerEvent) -> None:
    """Emit one `RunnerEvent` as a LangChain custom event.

    Pass it straight to `WorkflowRunner(on_event=...)`.

    Only callable from inside a run — a tool, a `RunnableLambda`, a graph node —
    because that is where the callback manager lives; elsewhere LangChain raises
    `RuntimeError`. `run_workflow` is a tool, so that holds wherever the runner
    is driven by the orchestrator. The sync form is deliberate: `run_workflow` is
    a sync tool, and a sync dispatch from one still reaches an async
    `astream_events` consumer through contextvars (Python ≥ 3.11, which the
    harness requires).
    """
    dispatch_custom_event(event.name, event.value)
