"""What a run cost — the table, the arithmetic, and where the number ends up.

The reconciliation against run 003 is the test that matters: the same token
counts that document says cost $0.47 have to come out of this code as $0.47.
"""

from pathlib import Path

import pytest

from dsagent.cartridge import load_cartridge
from dsagent.pricing import Prices, find_prices
from dsagent.runner import WorkflowRunner
from dsagent.runs import summarize
from tests.fakes import stub_env
from tests.test_runner_events import StreamingFakeAgent

DS = Path(__file__).resolve().parents[1] / "cartridges" / "ds"
REPO = Path(__file__).resolve().parents[1]

# `docs/runs/eda-to-report-003.md`, the totals row.
RUN_003 = {
    "input_tokens": 560_421,
    "output_tokens": 23_025,
    "cache_read": 506_204,
    "cache_creation": 54_143,
}


def test_the_repository_ships_a_price_for_the_default_model():
    prices = Prices.load(REPO / "prices.yaml")
    assert prices.rate_for("anthropic:claude-sonnet-5") is not None
    # and by the bare name, which is what a persona's frontmatter would say
    assert prices.rate_for("claude-sonnet-5") is not None


def test_run_003_costs_what_run_003_cost():
    prices = Prices.load(REPO / "prices.yaml")
    total = prices.cost("anthropic:claude-sonnet-5", RUN_003)
    assert total is not None
    assert round(total, 2) == 0.47, total
    # the shape of the bill, not just its total: cache reads are the bulk of the
    # input and cost a tenth of it, which is why the cached share is on screen
    assert 0.466 < total < 0.468


def test_an_unknown_model_costs_nothing_known():
    prices = Prices.load(REPO / "prices.yaml")
    assert prices.cost("some-model-nobody-priced", RUN_003) is None
    assert Prices().cost("anthropic:claude-sonnet-5", RUN_003) is None


def test_an_unreadable_table_is_no_table(tmp_path):
    (tmp_path / "prices.yaml").write_text("models: [this is a list]")
    assert Prices.load(tmp_path / "prices.yaml").rates == {}
    (tmp_path / "broken.yaml").write_text("models: {a: {input: [1]}}")
    assert Prices.load(tmp_path / "broken.yaml").rates == {}
    assert Prices.load(tmp_path / "absent.yaml").rates == {}


def test_uncached_input_is_the_remainder():
    prices = Prices(Prices.load(REPO / "prices.yaml").rates)
    # 1M input of which 900k cached and 50k written: 50k at full rate
    usage = {"input_tokens": 1_000_000, "cache_read": 900_000,
             "cache_creation": 50_000, "output_tokens": 0}
    expected = (50_000 * 2.0 + 900_000 * 0.20 + 50_000 * 2.50) / 1_000_000
    assert prices.cost("anthropic:claude-sonnet-5", usage) == pytest.approx(expected)


def test_a_step_records_what_it_cost(tmp_path, monkeypatch):
    """The number lands in `run.json`, per step, and totals on the run."""
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )

    class Costly(StreamingFakeAgent):
        def stream(self, payload, stream_mode=None):
            for mode, chunk in super().stream(payload, stream_mode):
                if mode == "values":
                    for message in chunk["messages"]:
                        message.usage_metadata = {
                            "input_tokens": 100_000,
                            "output_tokens": 1_000,
                            "input_token_details": {"cache_read": 90_000, "cache_creation": 5_000},
                        }
                yield mode, chunk

    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "priced",
        agent_factory=lambda c, persona, env, ws: Costly(persona, ws),
        log=lambda m: None,
        prices=Prices.load(REPO / "prices.yaml"),
    )
    state = runner.run("eda-to-report", {"data_path": "x.csv"})

    profile = state.steps["profile"]
    assert profile.model == "anthropic:claude-sonnet-5"
    assert profile.cost_usd and profile.cost_usd > 0
    summary = summarize(tmp_path / "priced")
    assert summary.cost_usd == pytest.approx(
        sum(s.cost_usd for s in state.steps.values() if s.cost_usd), rel=1e-6
    )


def test_without_a_table_a_run_reports_no_cost_rather_than_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dsagent.runner.runner.make_env",
        stub_env,
    )
    runner = WorkflowRunner(
        load_cartridge(DS), tmp_path / "unpriced",
        agent_factory=lambda c, persona, env, ws: StreamingFakeAgent(persona, ws),
        log=lambda m: None,
    )
    state = runner.run("eda-to-report", {"data_path": "x.csv"})
    assert all(s.cost_usd is None for s in state.steps.values())
    assert summarize(tmp_path / "unpriced").cost_usd is None


def test_the_table_is_found_from_a_subdirectory():
    found = find_prices(REPO / "src" / "dsagent")
    assert found == REPO / "prices.yaml"
