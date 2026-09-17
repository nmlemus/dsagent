"""Build a replay fixture from a real run directory.

    python tools/make_replay_fixture.py .dsagent/runs/<run> ui/fixtures/run-eda-003

The UI has to be developed against a run that behaves like a run — four steps,
a gate that waits, ten files landing over six minutes — and doing that against a
model costs half a dollar a look. So one real run is turned into a fixture that
`dsagent serve --replay` plays back, and the browser cannot tell the difference.

**What is real and what is reconstructed.** The run's `run.json` and its whole
workspace are real: step boundaries, the 37 seconds the gate waited, every file
with the mtime it actually got, the tool calls each step made counted by name,
and each persona's closing summary. What `run.json` does not record is the
*order* and the individual timestamps of tool calls inside a step, so those are
spread evenly across the step they belong to. Nothing else is invented — no
narration is written that a persona did not write, and no file appears that the
run did not produce.

The fixture is for developing screens. Evidence for `docs/ui-product.md` §7 comes
from real runs, never from here.
"""

from __future__ import annotations

import json
import shutil
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsagent.cartridge import load_cartridge  # noqa: E402
from dsagent.runner.runner import is_pattern  # noqa: E402

CARTRIDGE = Path("cartridges/ds")


def matched(entry: str, paths: list[str]) -> list[str]:
    """`produces` entry → the recorded paths it names, glob semantics included.

    Segment-wise, so `figures/*.png` cannot claim `artifacts/figures/x.png` —
    the same rule `WorkflowRunner.matched` gets from `Path.glob`.
    """
    if not is_pattern(entry):
        return [entry] if entry in paths else []
    want = entry.split("/")
    out = []
    for p in paths:
        parts = p.split("/")
        if len(parts) == len(want) and all(fnmatchcase(a, b) for a, b in zip(parts, want)):
            out.append(p)
    return sorted(out)


def build(run_dir: Path, out: Path, cartridge_path: Path = CARTRIDGE) -> int:
    """The fixture for one run: its event log, its workspace, its record.

    A run that wrote its own `events.jsonl` needs nothing reconstructed — the log
    *is* the recording, charts and all, and copying it is both simpler and more
    faithful than rebuilding it from `run.json`. The reconstruction below stays
    for run 003, which predates the event log and is still the M2.5 fixture.
    """
    real = run_dir / "events.jsonl"
    if real.is_file() and real.stat().st_size:
        return _copy(run_dir, out)
    return _reconstruct(run_dir, out, cartridge_path)


def _copy(run_dir: Path, out: Path) -> int:
    events = [line for line in real_lines(run_dir) if line.strip()]
    _write_out(run_dir, out, "".join(events))
    return len(events)


def real_lines(run_dir: Path) -> list[str]:
    return (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)


def _write_out(run_dir: Path, out: Path, events: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "events.jsonl").write_text(events, encoding="utf-8")
    shutil.copytree(
        run_dir / "workspace", out / "workspace",
        # `.dsagent/` is the harness's materialized skills, not the run's output,
        # and the files endpoint refuses to serve it anyway.
        ignore=shutil.ignore_patterns(".dsagent", "__pycache__"),
    )
    shutil.copy2(run_dir / "run.json", out / "run.json")


def _reconstruct(run_dir: Path, out: Path, cartridge_path: Path = CARTRIDGE) -> int:
    state = json.loads((run_dir / "run.json").read_text())
    cartridge = load_cartridge(cartridge_path)
    wf = cartridge.workflows[state["workflow"]]
    order = [s.id for s in wf.ordered_steps()]
    run_id = out.name

    landed: list[str] = []
    events: list[dict[str, Any]] = []

    def emit(name: str, ts: float, value: dict[str, Any]) -> None:
        events.append({"name": name, "value": {"run_id": run_id, **value, "ts": ts}})

    def step_event(step, status: str, ts: float, gate: dict | None = None) -> None:
        emit("dsagent.step", ts, {
            "workflow": wf.name, "step": step.id, "persona": step.persona,
            "env": step.env or wf.env, "index": order.index(step.id), "total": len(order),
            "status": status, "needs": list(step.needs), "produces": list(step.produces),
            "produces_matched": {
                e: ([] if status == "started" else matched(e, landed)) for e in step.produces
            },
            "gate": gate, "error": None,
        })

    for step_id in order:
        rec = state["steps"][step_id]
        step = wf.step(step_id)
        start, end = rec["started_at"], rec["finished_at"]
        step_event(step, "started", start)

        # Tool calls: the counts are real, the individual timestamps are not
        # recorded, so they are spread across the step round-robin by name.
        calls = [(name, i) for name, n in rec["tool_calls"].items() for i in range(n)]
        calls.sort(key=lambda c: (c[1], c[0]))
        span = max(end - start, 1.0)
        for i, (name, _) in enumerate(calls):
            at = start + span * (i + 0.5) / (len(calls) + 1)
            call_id = f"replay_{step_id}_{i}"
            emit("dsagent.tool", at, {
                "step": step_id, "persona": step.persona, "tool": name,
                "tool_call_id": call_id, "phase": "started", "args_preview": {},
            })
            emit("dsagent.tool", at + span * 0.01, {
                "step": step_id, "persona": step.persona, "tool": name,
                "tool_call_id": call_id, "phase": "finished", "args_preview": None,
            })

        declared = {p for entry in step.produces for p in matched(entry, [f["path"] for f in rec["files"]])}
        for f in rec["files"]:
            landed.append(f["path"])
            size = (run_dir / "workspace" / f["path"]).stat().st_size
            emit("dsagent.file", f["mtime"], {
                "step": step_id, "path": f["path"],
                "kind": "deliverable" if f["path"] in declared else "working",
                "change": "created", "size": size, "mtime": f["mtime"],
            })

        if rec.get("output"):
            emit("dsagent.note", end - 1, {
                "step": step_id, "persona": step.persona, "text": rec["output"],
            })

        step_event(step, rec["status"], end)

        gate = rec.get("gate")
        if step.gate and gate:
            # `asked_at` predates the field; the step finishing is when the run
            # stopped, and the recorded `ts` is when someone answered.
            asked = gate.get("asked_at") or end
            prompt = step.gate.prompt or f"Step '{step_id}' finished. Continue?"
            step_event(step, "awaiting_gate", asked, gate={
                "kind": step.gate.kind, "prompt": prompt, "asked_at": asked,
                "decision": None, "note": "", "decided_at": None,
            })
            step_event(step, rec["status"], gate["ts"], gate={
                "kind": step.gate.kind, "prompt": prompt, "asked_at": asked,
                "decision": gate["decision"], "note": gate.get("note", ""),
                "decided_at": gate["ts"],
            })

    events.sort(key=lambda e: e["value"]["ts"])
    _write_out(run_dir, out, "".join(json.dumps(e) + "\n" for e in events))
    return len(events)


if __name__ == "__main__":
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    n = build(src, dest)
    print(f"{n} events → {dest}/events.jsonl")
