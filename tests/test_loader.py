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


def test_two_steps_with_the_same_id_are_rejected(tmp_path):
    """`ordered_steps` resolves an id through `step()`, which returns the first
    match, so a duplicate silently ran the first step twice and never ran the
    second — its instructions, its persona and its produces were all ignored."""
    from dsagent.cartridge.models import Step, Workflow

    with pytest.raises(ValueError, match="duplicate step id"):
        Workflow(
            name="probe",
            path=Path("."),
            steps=[
                Step(id="same", persona="a", instructions="first.md"),
                Step(id="same", persona="b", instructions="second.md"),
            ],
        )


def test_a_duplicate_step_id_in_a_cartridge_fails_to_load(tmp_path):
    root = tmp_path / "dup"
    shutil.copytree(DS, root)
    wf = root / "workflows" / "eda-to-report" / "workflow.yaml"
    wf.write_text(wf.read_text().replace("  - id: analyze\n", "  - id: profile\n", 1))
    with pytest.raises(CartridgeError, match="duplicate step id"):
        load_cartridge(root)


def test_two_cartridges_may_not_declare_the_same_workflow(tmp_path):
    """Personas were checked; workflows were not, and every lookup the API, the
    driver and the tools build is a dict keyed by workflow name — so the last
    cartridge loaded silently won and runs went to the wrong team."""
    from dsagent.cartridge import load_cartridges

    # A minimal cartridge whose only collision with `ds` is the workflow name:
    # different cartridge, different persona, different skill.
    twin = tmp_path / "twin"
    (twin / "agents").mkdir(parents=True)
    (twin / "skills" / "notes").mkdir(parents=True)
    (twin / "workflows" / "eda-to-report" / "steps").mkdir(parents=True)
    (twin / "agents" / "sam.md").write_text("---\nname: sam\nskills: [notes]\n---\nYou are Sam.\n")
    (twin / "skills" / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: n\n---\nnotes\n"
    )
    (twin / "workflows" / "eda-to-report" / "steps" / "a.md").write_text("Do the thing.")
    (twin / "workflows" / "eda-to-report" / "workflow.yaml").write_text(
        "name: eda-to-report\nsteps:\n"
        "  - {id: a, persona: sam, instructions: steps/a.md, produces: [out.md]}\n"
    )
    (twin / "cartridge.yaml").write_text(
        "name: twin\npersonas:\n  sam: {role: Writer, workflows: [eda-to-report]}\n"
        "skills:\n  notes: {scope: all}\nworkflows: [workflows/eda-to-report]\n"
    )
    with pytest.raises(CartridgeError, match="workflow 'eda-to-report'"):
        load_cartridges([DS, twin])


def test_plain_plugin_without_cartridge_yaml_is_rejected(tmp_path):
    root = tmp_path / "plugin"
    (root / "skills").mkdir(parents=True)
    with pytest.raises(CartridgeError, match="cartridge.yaml is missing"):
        load_cartridge(root)
