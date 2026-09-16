"""Step instructions are markdown, not format strings.

`_fill` and `visible_input_names` have to agree on what a placeholder is, and
`_PLACEHOLDER` is the definition. `str.format_map` disagreed with it: it treats
`{"rhat_max": float}` as a field with a format spec and raises before any model
is called, so `mmm-meridian`'s `fit` step — whose instructions document a JSON
artifact — could not run at all.
"""

from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.runner.runner import _fill, visible_input_names

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"


def test_a_named_input_is_substituted():
    assert _fill("Load {data_path} now.", {"data_path": "data/sales.csv"}) == "Load data/sales.csv now."


def test_an_unknown_name_stays_literal():
    assert _fill("Load {data_path} and {nope}.", {"data_path": "x"}) == "Load x and {nope}."


def test_none_renders_as_none():
    assert _fill("key={key_column}", {"key_column": None}) == "key=None"


def test_json_in_the_instructions_survives_untouched():
    text = '`{"rhat_max": float, "divergences": int, "params": {name: rhat}}` — the shape'
    assert _fill(text, {"kpi": "units"}) == text


@pytest.mark.parametrize(
    "text",
    [
        "a literal brace: {{",
        "}} on its own",
        "a dict literal {'a': 1}",
        "an f-string example {value!r}",
        "a spec example {x:>10}",
        "css: .cls { margin: 0 }",
    ],
)
def test_braces_that_are_not_placeholders_pass_through(text):
    """`format_map` raised or mangled several of these; markdown may contain any."""
    assert _fill(text, {"x": "X", "value": "V"}) == text


def test_fill_and_visibility_agree_on_what_a_placeholder_is():
    text = 'see {data_path}, ignore {"rhat_max": float} and {x:>10}'
    names = visible_input_names(_step_without_sees(), text)
    assert names == ["data_path"]
    filled = _fill(text, {"data_path": "d.csv", "x": "X"})
    assert "d.csv" in filled
    assert "{x:>10}" in filled


def _step_without_sees():
    from dsagent.cartridge.models import Step

    return Step(id="s", persona="marie", instructions="s.md")


def test_the_mmm_fit_step_renders():
    """The step this bug blocked. It documents a JSON artifact in its markdown.

    It interpolates nothing itself — it declares `sees: [kpi, date_range]` — so
    what matters is that the JSON comes through byte for byte and nothing raises.
    """
    wf = load_cartridge(DS).workflows["mmm-meridian"]
    step = next(s for s in wf.steps if s.id == "fit")
    raw = (wf.path / step.instructions).read_text(encoding="utf-8")
    assert '"rhat_max"' in raw, "fixture check: this step is the one with JSON in it"

    out = _fill(raw, {"kpi": "units", "date_range": "2023-01-01..2024-12-31"})
    assert out == raw


def test_a_step_that_does_interpolate_still_does():
    """`03-model-spec.md` carries `{kpi}`; it must still be replaced."""
    wf = load_cartridge(DS).workflows["mmm-meridian"]
    step = next(s for s in wf.steps if s.id == "model-spec")
    raw = (wf.path / step.instructions).read_text(encoding="utf-8")
    assert "{kpi}" in raw, "fixture check: this step is the one with a placeholder"

    out = _fill(raw, {"kpi": "units sold"})
    assert "units sold" in out
    assert "{kpi}" not in out


def test_every_step_in_every_workflow_renders():
    """A cartridge-wide guard: no step's markdown may break templating."""
    cart = load_cartridge(DS)
    for wf in cart.workflows.values():
        inputs = {name: f"<{name}>" for name in wf.inputs}
        for step in wf.steps:
            raw = (wf.path / step.instructions).read_text(encoding="utf-8")
            _fill(raw, inputs)  # must not raise
