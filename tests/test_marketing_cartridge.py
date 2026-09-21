"""The marketing cartridge: the proof that a second domain loads, validates and
reaches the UI's contract with no harness change. No model, no kernel."""

from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge, load_cartridges

ROOT = Path(__file__).resolve().parents[1] / "cartridges"
DS, MARKETING = ROOT / "ds", ROOT / "marketing"


def test_marketing_cartridge_loads_and_matrix_is_consistent():
    c = load_cartridge(MARKETING)
    assert set(c.personas) == {"lucia", "tomas"}
    assert c.personas["lucia"].role == "Brand Strategist"
    assert [s.name for s in c.skills_for("tomas")] == ["briefs", "copy"]
    assert [s.name for s in c.traversal_skills()] == ["briefs"]
    assert set(c.workflows) == {"campaign-brief"}
    assert c.envs["default"].kind == "kernel"
    assert c.envs["default"].requirements == []


def test_marketing_workflow_is_declared_for_the_screens():
    wf = load_cartridge(MARKETING).workflows["campaign-brief"]
    assert [s.id for s in wf.ordered_steps()] == ["research", "strategy", "copy"]
    assert wf.step("strategy").gate and wf.step("strategy").gate.kind == "human"
    assert (wf.title_input, wf.data_input) == ("objective", "brief_path")
    assert wf.title == "Campaña desde un brief"
    assert wf.summary and wf.summary != wf.description
    assert wf.inputs["objective"].label == "Objetivo de la campaña"
    assert wf.step("copy").title == "Escribir las piezas"


def test_marketing_experience_speaks_marketing():
    e = load_cartridge(MARKETING).experience
    assert e.nouns.run.plural == "campañas"
    assert e.nouns.deliverable.singular == "pieza"
    assert e.nav.resources == "Biblioteca de marca"
    assert len(e.assistant.suggestions) == 3
    assert e.resources.formats == ["md", "txt", "pdf"]


def test_both_cartridges_coexist_on_one_server():
    carts = load_cartridges([DS, MARKETING])
    assert [c.name for c in carts] == ["ds", "marketing"]
    fastapi = pytest.importorskip("fastapi", reason="needs the 'ui' extra")
    from fastapi.testclient import TestClient

    from dsagent import serve

    del fastapi
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        app = serve.build_app(
            carts, None, tmp_path / "ws", tmp_path / "runs",
            replay=Path(__file__).resolve().parent / "fixtures" / "replay" / "run-eda-003",
        )
        body = TestClient(app).get("/cartridges").json()
    by_name = {c["name"]: c for c in body["cartridges"]}
    assert by_name["ds"]["experience"]["nouns"]["run"]["singular"] == "análisis"
    assert by_name["marketing"]["experience"]["nouns"]["run"]["singular"] == "campaña"
    assert {w["name"]: w["cartridge"] for w in body["workflows"]} == {
        "eda-to-report": "ds", "mmm-meridian": "ds", "campaign-brief": "marketing",
    }
