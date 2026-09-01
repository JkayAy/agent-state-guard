"""Tests for agent_state_guard.graph."""
from __future__ import annotations

import pytest

from agent_state_guard import (
    DeterministicGraph,
    MaxRetriesExceededError,
    NodeStatus,
    RetryPolicy,
    SchemaViolationError,
    stable_hash,
)


def test_run_executes_nodes_in_order_and_increments_step(make_state):
    graph = DeterministicGraph()

    def step_one(state):
        return state.model_copy(update={"scratchpad": {**state.scratchpad, "step_one": True}})

    def step_two(state):
        return state.model_copy(update={"scratchpad": {**state.scratchpad, "step_two": True}})

    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)

    result = graph.run(make_state())
    assert result.final_state.scratchpad == {"step_one": True, "step_two": True}
    assert result.final_state.step == 2
    assert [t.node_name for t in result.transitions] == ["step_one", "step_two"]
    assert all(t.status == NodeStatus.SUCCEEDED for t in result.transitions)


def test_schema_violation_with_no_fallback_surfaces_as_max_retries_exceeded(make_state):
    # A node returning the wrong type is a SchemaViolationError internally,
    # but with no fallback registered, _execute_node re-raises it wrapped as
    # MaxRetriesExceededError once the (here, single) attempt budget is
    # exhausted. The original SchemaViolationError is preserved as the
    # exception's __cause__ for anyone who needs to inspect it.
    graph = DeterministicGraph()
    graph.add_node("bad", lambda state: {"not": "a state"})
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        graph.run(make_state())
    assert isinstance(exc_info.value.__cause__, SchemaViolationError)


def test_retry_then_success(make_state):
    attempts = {"n": 0}

    def flaky(state):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("not yet")
        return state.model_copy(update={"scratchpad": {"ok": True}})

    graph = DeterministicGraph()
    graph.add_node(
        "flaky",
        flaky,
        retry_policy=RetryPolicy(max_attempts=3, sleep_fn=lambda _s: None),
    )
    result = graph.run(make_state())
    assert result.final_state.scratchpad == {"ok": True}
    assert [t.status for t in result.transitions] == [NodeStatus.FAILED, NodeStatus.SUCCEEDED]


def test_fallback_used_when_primary_exhausts_retries(make_state):
    def always_fails(state):
        raise RuntimeError("primary down")

    def fallback_node(state):
        return state.model_copy(update={"scratchpad": {"used_fallback": True}})

    graph = DeterministicGraph()
    graph.add_node(
        "primary",
        always_fails,
        retry_policy=RetryPolicy(max_attempts=2, sleep_fn=lambda _s: None),
        fallback="backup",
    )
    graph.add_node("backup", fallback_node)

    spec = graph._nodes["primary"]
    final_state, records = graph._execute_node(spec, make_state())

    assert final_state.scratchpad == {"used_fallback": True}
    assert [r.status for r in records] == [
        NodeStatus.FAILED,
        NodeStatus.FAILED,
        NodeStatus.FALLBACK,
    ]
    assert records[-1].node_name == "backup"


def test_unknown_fallback_raises_schema_violation(make_state):
    def always_fails(state):
        raise RuntimeError("boom")

    graph = DeterministicGraph()
    graph.add_node(
        "primary",
        always_fails,
        retry_policy=RetryPolicy(max_attempts=1, sleep_fn=lambda _s: None),
        fallback="does-not-exist",
    )
    with pytest.raises(SchemaViolationError):
        graph.run(make_state())


def test_max_retries_exceeded_without_fallback(make_state):
    def always_fails(state):
        raise RuntimeError("boom")

    graph = DeterministicGraph()
    graph.add_node(
        "primary",
        always_fails,
        retry_policy=RetryPolicy(max_attempts=2, sleep_fn=lambda _s: None),
    )
    with pytest.raises(MaxRetriesExceededError):
        graph.run(make_state())


def test_terminal_state_stops_execution_early(make_state):
    def set_terminal(state):
        return state.model_copy(update={"terminal": True})

    ran = {"second": False}

    def should_not_run(state):
        ran["second"] = True
        return state

    graph = DeterministicGraph()
    graph.add_node("first", set_terminal)
    graph.add_node("second", should_not_run)

    result = graph.run(make_state())
    assert result.final_state.terminal is True
    assert ran["second"] is False
    assert [t.node_name for t in result.transitions] == ["first"]


def test_add_node_rejects_duplicate_and_empty_names():
    graph = DeterministicGraph()
    graph.add_node("n1", lambda s: s)
    with pytest.raises(ValueError):
        graph.add_node("n1", lambda s: s)
    with pytest.raises(ValueError):
        graph.add_node("", lambda s: s)


def test_stable_hash_is_deterministic_and_order_independent(make_state):
    s1 = make_state(run_id="fixed-run", scratchpad={"a": 1, "b": 2})
    s2 = make_state(run_id="fixed-run", scratchpad={"b": 2, "a": 1})
    s3 = make_state(run_id="fixed-run", scratchpad={"a": 1, "b": 3})
    assert stable_hash(s1) == stable_hash(s2)
    assert stable_hash(s1) != stable_hash(s3)
