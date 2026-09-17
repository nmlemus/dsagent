"""A gate that is still answerable after the server has been restarted (§4.3, §7.11).

The interrupt lives in the checkpoint. With an in-memory saver it dies with the
process — which is exactly the run most likely to still be waiting when the
process goes away. These build a graph, interrupt it, throw the saver *and the
graph* away, build both again over the same file, and answer.
"""


import pytest

pytest.importorskip("langgraph.checkpoint.sqlite", reason="needs the 'ui' extra")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from dsagent.serve import CHECKPOINTS, checkpointer_for


def test_the_default_checkpointer_is_a_file(tmp_path):
    saver = checkpointer_for(tmp_path)
    assert isinstance(saver, SqliteSaver)
    assert (tmp_path / ".dsagent" / CHECKPOINTS).is_file()


def test_memory_is_still_available_for_a_throwaway_session(tmp_path):
    assert isinstance(checkpointer_for(tmp_path, memory=True), InMemorySaver)
    assert not (tmp_path / ".dsagent" / CHECKPOINTS).exists()


def _gated_graph(saver):
    """A two-node graph that stops at an `interrupt()` — a gate, in miniature."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def ask(state: dict) -> dict:
        answer = interrupt({"reason": "dsagent.gate", "step": "data-gate"})
        return {"decision": answer["decision"], "steps": [*state.get("steps", []), "asked"]}

    graph = StateGraph(dict)
    graph.add_node("ask", ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=saver)


def test_a_gate_survives_the_process_that_asked_it(tmp_path):
    """Kill the server at a gate, start it again, answer — the run goes on."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "run:eda-to-report-1"}}

    # the first "process": reach the gate and stop there
    first = checkpointer_for(tmp_path)
    out = _gated_graph(first).invoke({"steps": []}, config=config)
    assert out["__interrupt__"], "expected the graph to stop at its gate"
    del first  # the server goes away with the interrupt still unanswered

    # a second "process" over the same directory, and a new graph object
    second = checkpointer_for(tmp_path)
    resumed = _gated_graph(second).invoke(Command(resume={"decision": "approve"}), config=config)
    assert resumed["decision"] == "approve"
    assert resumed["steps"] == ["asked"]


def test_in_memory_loses_the_gate_with_the_process(tmp_path):
    """The behaviour §4.3 is replacing, asserted so the difference is on record."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "run:eda-to-report-2"}}
    first = checkpointer_for(tmp_path, memory=True)
    assert _gated_graph(first).invoke({"steps": []}, config=config)["__interrupt__"]

    # A fresh in-memory saver knows nothing about that thread, so the answer
    # lands nowhere at all — the graph returns `None`, having found nothing to
    # resume. That is the operator coming back to a question nobody is holding.
    second = checkpointer_for(tmp_path, memory=True)
    assert _gated_graph(second).invoke(Command(resume={"decision": "approve"}), config=config) is None
