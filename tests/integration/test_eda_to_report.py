"""End-to-end `eda-to-report` run against a real model and the kernel env.

This is the only test in the suite that spends money: it drives four persona
agents through the real Deep Agents loop, so it is skipped unless
``DSAGENT_INTEGRATION=1``.

The fixture is a small public CSV vendored at ``tests/data/seattle-weather.csv``
(provenance in ``tests/data/README.md``). It is clean by construction, which is
what makes the gate assertion meaningful: every threshold in the `eda` skill is
satisfied, so anything other than ``GATE: PASS`` is the model misreading the
skill rather than the data being bad.

Artifacts land in ``.dsagent/runs/`` rather than ``tmp_path`` so the run can be
read after the fact — that is the raw material for ``docs/runs/``. Override the
location with ``DSAGENT_INTEGRATION_RUN_DIR``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.host.build import DEFAULT_MODEL
from dsagent.runner import GateDecision, RunState, WorkflowRunner

REPO = Path(__file__).resolve().parents[2]
CARTRIDGE = REPO / "cartridges" / "ds"
DATASET = REPO / "tests" / "data" / "seattle-weather.csv"

EXPECTED_ROWS = 1461
EXPECTED_COLUMNS = {"date", "precipitation", "temp_max", "temp_min", "wind", "weather"}
KEY_COLUMN = "date"
QUESTION = (
    "How do precipitation and temperature differ across the weather categories, "
    "and how did they change between 2012 and 2015?"
)

# The kernel env runs in this interpreter (jupyter_client's native `python3`
# kernelspec points at sys.executable), so an in-process check is the right
# proxy for what the persona will find when it calls `run_python`.
# pandas: the `eda` skill's scripts/profile.py. matplotlib: the figures step 03
# asks for. markdown: the `reports` skill's scripts/render_html.py.
KERNEL_REQUIREMENTS = ("pandas", "matplotlib", "markdown")

# Credential per model provider, so a missing key skips up front instead of
# failing once the kernel is up and the first step is already running. Only the
# providers this project ships extras for; an unknown provider is left alone
# because we cannot know what it needs.
PROVIDER_CREDENTIALS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}

GATE_LINE = re.compile(r"^GATE: (PASS|FAIL — .+)$")

pytestmark = pytest.mark.skipif(
    os.environ.get("DSAGENT_INTEGRATION") != "1",
    reason="costs money and minutes: set DSAGENT_INTEGRATION=1 to run it",
)


def _missing_requirements() -> list[str]:
    return [m for m in KERNEL_REQUIREMENTS if importlib.util.find_spec(m) is None]


def _missing_credential() -> tuple[str, str] | None:
    """(provider, env var) when DSAGENT_MODEL's provider needs a key we lack."""
    provider = DEFAULT_MODEL.split(":", 1)[0] if ":" in DEFAULT_MODEL else ""
    var = PROVIDER_CREDENTIALS.get(provider)
    if not var or os.environ.get(var):
        return None
    return provider, var


def _run_dir() -> Path:
    base = Path(os.environ.get("DSAGENT_INTEGRATION_RUN_DIR") or REPO / ".dsagent" / "runs")
    return base / f"eda-to-report-integration-{time.strftime('%Y%m%d-%H%M%S')}"


def _diagnostics(state: RunState, run_dir: Path, log: list[str]) -> str:
    """Everything needed to tell a harness bug from a prompt gap, in one message."""
    steps = "\n".join(
        f"  {r.id:<10} {r.status:<14} {r.error or ''}".rstrip() for r in state.steps.values()
    )
    return (
        f"\nrun status: {state.status}\nrun dir: {run_dir}\n"
        f"model: {DEFAULT_MODEL}\n"
        f"steps:\n{steps}\n"
        f"last log lines:\n" + "\n".join(f"  {line}" for line in log[-15:])
    )


@pytest.fixture(scope="module")
def completed_run() -> tuple[RunState, Path, str]:
    """Run the workflow once; every test below reads the same artifacts."""
    missing = _missing_requirements()
    if missing:
        pytest.skip(
            f"kernel env lacks {', '.join(missing)}; the `eda` and `reports` skill "
            f"scripts need them: pip install {' '.join(missing)}"
        )

    credential = _missing_credential()
    if credential:
        provider, var = credential
        pytest.skip(f"DSAGENT_MODEL is {DEFAULT_MODEL!r}: set {var} for the {provider} provider")

    run_dir = _run_dir()
    log: list[str] = []
    runner = WorkflowRunner(
        load_cartridge(CARTRIDGE),
        run_dir,
        # Unattended: the gate's verdict is asserted from its report, not acted on.
        ask_human=lambda prompt: GateDecision.APPROVE,
        log=log.append,
    )
    data_path = Path("data") / DATASET.name
    (runner.workspace / data_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATASET, runner.workspace / data_path)

    started = time.time()
    state = runner.run(
        "eda-to-report",
        {"data_path": str(data_path), "question": QUESTION, "key_column": KEY_COLUMN},
    )
    elapsed = time.time() - started
    (run_dir / "runner.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"\n[integration] {state.status} in {elapsed:.0f}s → {run_dir}")
    return state, runner.workspace, _diagnostics(state, run_dir, log)


def test_run_completes_and_every_step_is_done(completed_run):
    state, _, diag = completed_run
    assert state.status == "done", diag
    assert [r.status for r in state.steps.values()] == ["done"] * 4, diag


def test_declared_artifacts_are_on_disk_and_non_empty(completed_run):
    state, workspace, diag = completed_run
    if state.status != "done":
        pytest.fail(diag)
    workflow = load_cartridge(CARTRIDGE).workflows["eda-to-report"]
    declared = [p for step in workflow.steps for p in step.produces]
    empty = [p for p in declared if not (workspace / p).exists() or (workspace / p).stat().st_size == 0]
    # The runner only checks existence; emptiness is the failure mode it misses.
    assert not empty, f"declared but empty or missing: {empty}{diag}"


def test_profile_describes_the_real_dataset(completed_run):
    state, workspace, diag = completed_run
    if state.status != "done":
        pytest.fail(diag)
    profile = json.loads((workspace / "artifacts" / "data-profile.json").read_text())
    assert profile["rows"] == EXPECTED_ROWS, diag
    assert {c["column"] for c in profile["columns_detail"]} == EXPECTED_COLUMNS, diag
    assert profile["exact_duplicates_pct"] == 0.0, diag


def test_data_gate_passes_on_a_clean_dataset(completed_run):
    state, workspace, diag = completed_run
    if state.status != "done":
        pytest.fail(diag)
    report = (workspace / "artifacts" / "data-gate.md").read_text(encoding="utf-8")
    last = [line.strip() for line in report.splitlines() if line.strip()][-1]
    assert GATE_LINE.match(last), f"gate report must end with the verdict line, got {last!r}{diag}"
    # The fixture satisfies all four thresholds in the `eda` skill.
    assert last == "GATE: PASS", diag


def test_analysis_produced_figures_for_the_report(completed_run):
    state, workspace, diag = completed_run
    if state.status != "done":
        pytest.fail(diag)
    figures = sorted((workspace / "artifacts" / "figures").glob("*.png"))
    # Step 03 says "up to five figures"; a findings report with none defeats
    # the point of the `reports` skill's chart rules.
    assert 1 <= len(figures) <= 5, f"figures found: {[f.name for f in figures]}{diag}"


def test_report_is_self_contained_html(completed_run):
    state, workspace, diag = completed_run
    if state.status != "done":
        pytest.fail(diag)
    html = (workspace / "report" / "findings.html").read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!doctype html>"), diag
    # render_html.py inlines every referenced figure as a data URI.
    assert "src=\"data:image/" in html or "data:image/png;base64," in html, diag
    assert "<img" in html, diag
