"""The `experience:` block of cartridge.yaml and the presentation fields of
workflow.yaml — what a screen reads instead of recognising a domain's words.

Pure loader tests: no model, no kernel."""

from pathlib import Path

import pytest

from dsagent.cartridge import CartridgeError, load_cartridge
from dsagent.cartridge.models import Experience

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


def _cartridge(tmp_path: Path, experience_yaml: str) -> Path:
    """A one-persona, one-workflow cartridge with no domain at all."""
    root = tmp_path / "cart"
    (root / "agents").mkdir(parents=True)
    (root / "skills" / "notes").mkdir(parents=True)
    (root / "workflows" / "one" / "steps").mkdir(parents=True)
    (root / "agents" / "sam.md").write_text(
        "---\nname: sam\nskills: [notes]\n---\nYou are Sam.\n", encoding="utf-8"
    )
    (root / "skills" / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: n\n---\nnotes\n", encoding="utf-8"
    )
    (root / "workflows" / "one" / "steps" / "a.md").write_text("Do {topic}.", encoding="utf-8")
    (root / "workflows" / "one" / "workflow.yaml").write_text(
        "name: one\n"
        "title: Primer trabajo\n"
        "icon: file\n"
        "badge: Beta\n"
        "inputs:\n"
        "  topic: {type: string, label: Tema, placeholder: 'Ej. lanzamiento', help: Una frase}\n"
        "title_input: topic\n"
        "steps:\n"
        "  - {id: a, persona: sam, title: Preparar, instructions: steps/a.md, produces: [out.md]}\n",
        encoding="utf-8",
    )
    (root / "cartridge.yaml").write_text(
        "name: cart\n"
        "personas:\n  sam: {role: Writer, workflows: [one]}\n"
        "skills:\n  notes: {scope: all}\n"
        "workflows: [workflows/one]\n" + experience_yaml,
        encoding="utf-8",
    )
    return root


def test_experience_defaults_are_generic():
    e = Experience()
    assert e.nouns.run.plural == "trabajos"
    assert e.nav.resources == "Recursos"
    assert e.assistant.suggestions == []
    assert e.resources.formats == []


def test_cartridge_without_experience_gets_defaults(tmp_path):
    c = load_cartridge(_cartridge(tmp_path, ""))
    assert c.experience == Experience()
    wf = c.workflows["one"]
    assert (wf.title, wf.icon, wf.badge) == ("Primer trabajo", "file", "Beta")


def test_experience_and_presentation_fields_load(tmp_path):
    c = load_cartridge(
        _cartridge(
            tmp_path,
            "experience:\n"
            "  title: Estudio\n"
            "  nouns:\n    run: {singular: campaña, plural: campañas}\n"
            "  assistant:\n    suggestions: [Hola]\n",
        )
    )
    assert c.experience.title == "Estudio"
    assert c.experience.nouns.run.plural == "campañas"
    assert c.experience.nouns.resource.plural == "recursos"  # untouched default
    assert c.experience.assistant.suggestions == ["Hola"]
    wf = c.workflows["one"]
    assert wf.steps[0].title == "Preparar"
    assert wf.inputs["topic"].label == "Tema"
    assert wf.inputs["topic"].placeholder == "Ej. lanzamiento"
    assert wf.inputs["topic"].help == "Una frase"


def test_unknown_experience_key_is_rejected(tmp_path):
    with pytest.raises(CartridgeError, match="experience"):
        load_cartridge(_cartridge(tmp_path, "experience:\n  colour: red\n"))


def test_nested_unknown_experience_key_is_rejected(tmp_path):
    with pytest.raises(CartridgeError, match="experience"):
        load_cartridge(_cartridge(tmp_path, "experience:\n  nav:\n    settings: Ajustes\n"))


def test_generated_skill_prefix_uses_cartridge_name(tmp_path):
    root = _cartridge(tmp_path, "")
    c = load_cartridge(root)
    assert (root / "skills" / "cart-one" / "SKILL.md").exists()
    assert "cart-one" not in c.skills
    # A second load sees the generated skill on disk and must still not treat it
    # as a matrix entry — the prefix is the cartridge's name, not a fixed one.
    again = load_cartridge(root)
    assert set(again.skills) == {"notes"}


def test_ds_experience_is_declared():
    c = load_cartridge(DS)
    assert c.experience.nouns.run.singular == "análisis"
    assert c.experience.assistant.suggestions
    assert c.experience.resources.formats == ["csv", "parquet"]
    eda = c.workflows["eda-to-report"]
    assert eda.title == "Análisis exploratorio"
    assert eda.summary and eda.summary != eda.description
    assert eda.step("profile").title == "Conocer los datos"
    assert eda.inputs["question"].label
