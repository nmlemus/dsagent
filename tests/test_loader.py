import shutil
from pathlib import Path

import pytest

from dsagent.cartridge import CartridgeError, load_cartridge

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


def test_ds_cartridge_loads_and_matrix_is_consistent():
    c = load_cartridge(DS)
    assert set(c.personas) == {"marie", "noel", "ana", "pablo"}
    assert c.skills["reports"].scope == "all"
    assert c.skills["ml"].scope == ["noel"]
    assert [s.name for s in c.skills_for("marie")] == ["reports", "eda"]
    assert [s.name for s in c.traversal_skills()] == ["reports"]
    assert set(c.workflows) == {"eda-to-report", "mmm-meridian"}
    assert c.envs["meridian"].kind == "docker"


def test_workflow_topological_order():
    c = load_cartridge(DS)
    order = [s.id for s in c.workflows["mmm-meridian"].ordered_steps()]
    assert order == ["ingest", "data-gate", "model-spec", "fit", "optimize", "report"]


def test_command_skills_are_generated():
    c = load_cartridge(DS)
    for wf in c.workflows:
        md = (DS / "skills" / f"ds-{wf}" / "SKILL.md").read_text()
        assert md.startswith("---\nname: ds-" + wf)
        assert "run_workflow" in md
    # generated commands must not leak into the persona matrix
    assert not any(k.startswith("ds-") for k in c.skills)


def test_matrix_conflict_is_rejected(tmp_path):
    root = tmp_path / "bad"
    shutil.copytree(DS, root)
    marie = root / "agents" / "marie.md"
    marie.write_text(marie.read_text().replace("skills: [reports, eda]", "skills: [reports, eda, ml]"))
    with pytest.raises(CartridgeError, match="claims skill 'ml'"):
        load_cartridge(root)


def test_missing_traversal_skill_is_rejected(tmp_path):
    root = tmp_path / "bad"
    shutil.copytree(DS, root)
    noel = root / "agents" / "noel.md"
    noel.write_text(noel.read_text().replace("skills: [reports, eda, ml]", "skills: [eda, ml]"))
    with pytest.raises(CartridgeError, match="missing traversal skill 'reports'"):
        load_cartridge(root)


def test_plain_plugin_without_cartridge_yaml_is_rejected(tmp_path):
    root = tmp_path / "plugin"
    (root / "skills").mkdir(parents=True)
    with pytest.raises(CartridgeError, match="cartridge.yaml is missing"):
        load_cartridge(root)
