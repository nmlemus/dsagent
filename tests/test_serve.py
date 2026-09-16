"""`dsagent serve`: the AG-UI endpoint, the files route, and the gate payload.

Needs the `ui` extra (`pip install -e ".[ui]"`); skipped without it, so the core
suite still runs on a base install. No model and no browser: the app is built
against the real cartridge and driven with `TestClient`.
"""

from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.envs.base import Env
from dsagent.runner import GateDecision, GateRequest

pytest.importorskip("fastapi", reason="needs the 'ui' extra")
pytest.importorskip("ag_ui_langgraph", reason="needs the 'ui' extra")
pytest.importorskip("copilotkit", reason="needs the 'ui' extra")

# These have to come after the skips: importing them without the `ui` extra is the
# ImportError the skips exist to avoid. Ruff allows it because the skips above are
# bare expression statements, which E402 counts as preamble — binding one to a name
# would end the preamble and make this E402.
from fastapi.testclient import TestClient

from dsagent import serve

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


class FakeBackend:
    def close(self):
        pass


@pytest.fixture
def runs_dir(tmp_path):
    """A run directory with a deliverable, a working file, and a harness file."""
    ws = tmp_path / "runs" / "eda-to-report-toolu_01ABC" / "workspace"
    (ws / "report").mkdir(parents=True)
    (ws / "report" / "findings.html").write_text("<h1>findings</h1>")
    (ws / "artifacts").mkdir()
    (ws / "artifacts" / "data-profile.json").write_text('{"rows": 1461}')
    (ws / ".dsagent" / "skills" / "marie").mkdir(parents=True)
    (ws / ".dsagent" / "skills" / "marie" / "SKILL.md").write_text("secret")
    (tmp_path / "outside.txt").write_text("not yours")
    return tmp_path / "runs"


@pytest.fixture
def client(tmp_path, runs_dir, monkeypatch):
    monkeypatch.setenv("DSAGENT_MODEL", "anthropic:claude-sonnet-5")
    cart = load_cartridge(DS)
    spec = cart.envs["default"]
    env = Env(spec=spec, workspace=tmp_path / "ws", backend=FakeBackend())
    app = serve.build_app([cart], env, tmp_path / "ws", runs_dir)
    return TestClient(app)


# --- the AG-UI endpoint ----------------------------------------------------


def test_the_agent_endpoint_is_mounted_and_healthy(client):
    r = client.get("/agent/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "agent": {"name": "dsagent"}}


def test_the_agent_accepts_posts(client):
    """`add_langgraph_fastapi_endpoint` registers POST at the path; GET is not it."""
    assert client.get("/agent").status_code in (404, 405)
    # A malformed RunAgentInput is rejected by validation, which proves the route
    # exists and is typed — without starting a model.
    assert client.post("/agent", json={"nope": True}).status_code == 422


# --- the files route -------------------------------------------------------

RUN = "eda-to-report-toolu_01ABC"


def test_a_deliverable_is_served_with_a_useful_content_type(client):
    r = client.get(f"/runs/{RUN}/files/report/findings.html")
    assert r.status_code == 200
    assert r.text == "<h1>findings</h1>"
    assert r.headers["content-type"].startswith("text/html")


def test_json_keeps_its_type(client):
    r = client.get(f"/runs/{RUN}/files/artifacts/data-profile.json")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    "path",
    [
        "../../outside.txt",
        "report/../../../outside.txt",
        "/etc/hosts",
        "report/nope.html",
        "report",  # a directory is not a file
    ],
)
def test_paths_that_escape_or_miss_are_404(client, path):
    assert client.get(f"/runs/{RUN}/files/{path}").status_code == 404


def test_the_harness_own_files_are_denied(client):
    """`.dsagent/` is materialized skills, not the run's output."""
    r = client.get(f"/runs/{RUN}/files/.dsagent/skills/marie/SKILL.md")
    assert r.status_code == 404


def test_an_unknown_run_is_404(client):
    assert client.get("/runs/no-such-run/files/report/findings.html").status_code == 404


def test_a_run_id_may_not_be_a_path(runs_dir):
    """Checked directly: a client cannot send `..` as a path segment through TestClient."""
    assert serve.resolve_run_file(runs_dir, "..", "outside.txt") is None
    assert serve.resolve_run_file(runs_dir, ".", "x") is None
    assert serve.resolve_run_file(runs_dir, "", "x") is None
    assert serve.resolve_run_file(runs_dir, "a/b", "x") is None


def test_resolve_accepts_a_real_deliverable(runs_dir):
    target = serve.resolve_run_file(runs_dir, RUN, "report/findings.html")
    assert target is not None and target.read_text() == "<h1>findings</h1>"


def test_a_symlink_out_of_the_workspace_is_refused(runs_dir, tmp_path):
    ws = runs_dir / RUN / "workspace"
    (ws / "escape.txt").symlink_to(tmp_path / "outside.txt")
    assert serve.resolve_run_file(runs_dir, RUN, "escape.txt") is None


# --- the gate --------------------------------------------------------------


def _request() -> GateRequest:
    return GateRequest(
        run_id="eda-to-report-toolu_01ABC", workflow="eda-to-report", step="data-gate",
        persona="marie", produces=["artifacts/data-gate.md"],
        prompt="Data gate report is in artifacts/data-gate.md. Proceed to analysis?",
    )


def test_the_gate_payload_is_the_documented_shape():
    """`docs/ui-slice.md` §3. The first three keys are the ones the bridge lifts."""
    payload = serve.gate_payload(_request())
    assert payload["reason"] == "dsagent.gate"
    assert payload["message"].startswith("Data gate report")
    assert payload["response_schema"]["required"] == ["decision"]
    assert payload["response_schema"]["properties"]["decision"]["enum"] == ["approve", "reject"]
    assert payload["run_id"] == "eda-to-report-toolu_01ABC"
    assert payload["workflow"] == "eda-to-report"
    assert payload["step"] == "data-gate"
    assert payload["persona"] == "marie"
    assert payload["produces"] == ["artifacts/data-gate.md"]


def test_the_payload_survives_the_bridges_own_mapping():
    """The keys are only useful if `lg_interrupt_to_agui` keeps them."""
    from ag_ui_langgraph.interrupts import lg_interrupt_to_agui
    from langgraph.types import Interrupt

    agui = lg_interrupt_to_agui(Interrupt(value=serve.gate_payload(_request()), id="i1"))
    assert agui.reason == "dsagent.gate"
    assert agui.message.startswith("Data gate report")
    assert agui.response_schema["required"] == ["decision"]
    raw = agui.metadata["langgraph"]["raw"]
    assert raw["step"] == "data-gate"
    assert raw["produces"] == ["artifacts/data-gate.md"]


def test_serve_mode_ask_human_interrupts_with_that_payload(monkeypatch):
    """No graph needed: `interrupt` is patched, which is the whole contract here."""
    seen: list[dict] = []

    def fake_interrupt(value):
        seen.append(value)
        return {"decision": "approve"}

    monkeypatch.setattr("langgraph.types.interrupt", fake_interrupt)
    assert serve.interrupt_gate(_request()) is GateDecision.APPROVE
    assert seen == [serve.gate_payload(_request())]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"decision": "approve"}, GateDecision.APPROVE),
        ({"decision": "reject"}, GateDecision.REJECT),
        ({"decision": "approve", "note": "looks fine"}, GateDecision.APPROVE),
        ("approve", GateDecision.APPROVE),
        ("APPROVE", GateDecision.APPROVE),
        (GateDecision.APPROVE, GateDecision.APPROVE),
        ({"decision": "maybe"}, GateDecision.REJECT),
        ({}, GateDecision.REJECT),
        (None, GateDecision.REJECT),
        ({"__agui_cancelled__": True}, GateDecision.REJECT),
    ],
)
def test_a_resumed_answer_maps_to_a_decision(answer, expected):
    """Anything that is not plainly an approval is a rejection — the safe direction."""
    assert serve.decision_of(answer) is expected
