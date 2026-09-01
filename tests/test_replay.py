"""Tests for agent_state_guard.replay."""
from __future__ import annotations

from agent_state_guard import (
    DeterministicGraph,
    NodeStatus,
    StateTransitionRecord,
    replay_and_diff,
    verify_chain_integrity,
)


def _record(run_id, step, node_name, status, input_hash, output_hash=None, attempt=1, error=None):
    return StateTransitionRecord(
        run_id=run_id,
        step=step,
        node_name=node_name,
        status=status,
        input_hash=input_hash,
        output_hash=output_hash,
        error=error,
        attempt=attempt,
        duration_ms=1.0,
    )


def test_verify_chain_integrity_detects_consistent_chain():
    r1 = _record("r1", 0, "a", NodeStatus.SUCCEEDED, "h0", output_hash="h1")
    r2 = _record("r1", 1, "b", NodeStatus.SUCCEEDED, "h1", output_hash="h2")
    result = verify_chain_integrity("r1", [r1, r2])
    assert result.consistent is True
    assert result.mismatches == []


def test_verify_chain_integrity_detects_broken_chain():
    r1 = _record("r1", 0, "a", NodeStatus.SUCCEEDED, "h0", output_hash="h1")
    r2 = _record("r1", 1, "b", NodeStatus.SUCCEEDED, "WRONG", output_hash="h2")
    result = verify_chain_integrity("r1", [r1, r2])
    assert result.consistent is False
    assert len(result.mismatches) == 1
    assert result.mismatches[0].step == 1


def test_verify_chain_integrity_ignores_failed_records():
    r1 = _record("r1", 0, "a", NodeStatus.SUCCEEDED, "h0", output_hash="h1")
    r_failed = _record("r1", 1, "b", NodeStatus.FAILED, "h1", error="boom", attempt=1)
    r2 = _record("r1", 1, "b", NodeStatus.SUCCEEDED, "h1", output_hash="h2", attempt=2)
    result = verify_chain_integrity("r1", [r1, r_failed, r2])
    assert result.consistent is True


def test_replay_and_diff_matches_for_deterministic_graph(make_state):
    graph = DeterministicGraph()
    graph.add_node("double", lambda state: state.model_copy(update={"scratchpad": {"value": 2}}))

    first_run = graph.run(make_state(run_id="replay-run"))

    result = replay_and_diff(graph, make_state(run_id="replay-run"), first_run.transitions)
    assert result.matches is True
    assert result.diffs == []


def test_replay_and_diff_detects_nondeterministic_node(make_state):
    counter = {"n": 0}

    def non_deterministic(state):
        counter["n"] += 1
        return state.model_copy(update={"scratchpad": {"call_count": counter["n"]}})

    graph = DeterministicGraph()
    graph.add_node("counter", non_deterministic)

    first_run = graph.run(make_state(run_id="replay-run-2"))

    result = replay_and_diff(graph, make_state(run_id="replay-run-2"), first_run.transitions)
    assert result.matches is False
    assert len(result.diffs) == 1
    assert result.diffs[0].node_name == "counter"
