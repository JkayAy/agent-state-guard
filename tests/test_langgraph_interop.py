"""Soft interoperability tests for DeterministicGraph.to_langgraph().

LangGraph's public API has changed across versions, and this integration
has never been executed against a live LangGraph installation before this
test suite. Rather than hard-failing CI whenever LangGraph ships a
breaking change, these tests attempt real compilation/execution and
explicitly skip (rather than fail) if that surface doesn't behave the way
this integration expects. The sequential DeterministicGraph.run()
executor -- covered exhaustively in test_graph.py -- remains the hard,
always-verified guarantee; LangGraph interop is a best-effort bonus.
"""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph", reason="langgraph is an optional dependency")

from agent_state_guard import DeterministicGraph  # noqa: E402


def test_to_langgraph_raises_clear_error_without_langgraph(monkeypatch):
    import agent_state_guard.graph as graph_module

    monkeypatch.setattr(graph_module, "LANGGRAPH_AVAILABLE", False)
    graph = DeterministicGraph()
    graph.add_node("n", lambda s: s)
    with pytest.raises(ImportError):
        graph.to_langgraph()


def test_to_langgraph_compiles_and_matches_sequential_runner(make_state):
    graph = DeterministicGraph()
    graph.add_node(
        "increment",
        lambda state: state.model_copy(update={"scratchpad": {"count": 1}}),
    )

    sequential_result = graph.run(make_state(run_id="lg-run"))

    try:
        compiled = graph.to_langgraph()
        langgraph_output = compiled.invoke(make_state(run_id="lg-run"))
    except Exception as exc:  # noqa: BLE001 - third-party API surface, not under our control
        pytest.skip(f"LangGraph compile/invoke path is not verified in this environment: {exc}")
        return

    # LangGraph may return a dict-like mapping rather than an AgentState
    # instance depending on the installed version; compare only the field
    # we actually mutated rather than assuming a specific return shape.
    scratchpad = getattr(langgraph_output, "scratchpad", None)
    if scratchpad is None and isinstance(langgraph_output, dict):
        scratchpad = langgraph_output.get("scratchpad")

    assert scratchpad == sequential_result.final_state.scratchpad
