"""Replay and audit-chain verification for AGENT-STATE-GUARD.

Two complementary checks are provided:

- `verify_chain_integrity` is a fast, storage-only check: given the
  records already persisted for a run, does the state hash chain look
  internally consistent (each successful/fallback step's input_hash
  should match the previous one's output_hash)? This can be run purely
  from the audit log, with no access to the original node functions.

- `replay_and_diff` is the real determinism test: it re-executes a graph
  against the same initial state and compares the freshly produced
  transitions against the previously stored ones step by step. Any
  mismatch means the nodes are not actually deterministic given identical
  input, which is exactly the failure mode this whole library exists to
  catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import DeterministicGraph
from .schemas import AgentState, NodeStatus, StateTransitionRecord


@dataclass
class ChainMismatch:
    step: int
    reason: str


@dataclass
class ChainIntegrityResult:
    run_id: str
    consistent: bool
    mismatches: list[ChainMismatch] = field(default_factory=list)


def verify_chain_integrity(
    run_id: str, records: list[StateTransitionRecord]
) -> ChainIntegrityResult:
    """Check that a run's stored records form a consistent hash chain.

    `records` is expected in the order it was produced (as returned by a
    TransitionStore's `load()`). For each successful/fallback record
    after the first, its `input_hash` should equal the previous
    successful/fallback record's `output_hash` -- if it doesn't, either
    the log was tampered with, records were dropped, or they were stored
    out of order.
    """
    mismatches: list[ChainMismatch] = []
    successful = [
        r for r in records if r.status in (NodeStatus.SUCCEEDED, NodeStatus.FALLBACK)
    ]
    for previous, current in zip(successful, successful[1:]):
        if previous.output_hash is None:
            mismatches.append(
                ChainMismatch(
                    step=current.step,
                    reason=f"record {previous.record_id} has no output_hash to chain from",
                )
            )
            continue
        if current.input_hash != previous.output_hash:
            mismatches.append(
                ChainMismatch(
                    step=current.step,
                    reason=(
                        f"input_hash {current.input_hash} does not match previous "
                        f"output_hash {previous.output_hash}"
                    ),
                )
            )
    return ChainIntegrityResult(run_id=run_id, consistent=not mismatches, mismatches=mismatches)


@dataclass
class ReplayDiff:
    step: int
    node_name: str
    stored_output_hash: "str | None"
    replayed_output_hash: "str | None"


@dataclass
class ReplayResult:
    run_id: str
    matches: bool
    diffs: list[ReplayDiff] = field(default_factory=list)


def replay_and_diff(
    graph: DeterministicGraph,
    initial_state: AgentState,
    expected_transitions: list[StateTransitionRecord],
) -> ReplayResult:
    """Re-execute `graph` from `initial_state` and diff against a prior run.

    This is the actual determinism guarantee: it does not trust the stored
    log at all, it re-runs the real node functions and compares fresh
    output hashes against what was recorded before. Only successful and
    fallback records are compared, since retried failures are expected to
    vary in timing (though not in eventual outcome).
    """
    result = graph.run(initial_state)
    replayed = [
        r for r in result.transitions if r.status in (NodeStatus.SUCCEEDED, NodeStatus.FALLBACK)
    ]
    expected = [
        r for r in expected_transitions if r.status in (NodeStatus.SUCCEEDED, NodeStatus.FALLBACK)
    ]

    diffs: list[ReplayDiff] = []
    for exp, rep in zip(expected, replayed):
        if exp.output_hash != rep.output_hash:
            diffs.append(
                ReplayDiff(
                    step=exp.step,
                    node_name=exp.node_name,
                    stored_output_hash=exp.output_hash,
                    replayed_output_hash=rep.output_hash,
                )
            )
    if len(expected) != len(replayed):
        diffs.append(
            ReplayDiff(
                step=-1,
                node_name="<length mismatch>",
                stored_output_hash=f"{len(expected)} stored records",
                replayed_output_hash=f"{len(replayed)} replayed records",
            )
        )

    run_id = expected_transitions[0].run_id if expected_transitions else initial_state.run_id
    return ReplayResult(run_id=run_id, matches=not diffs, diffs=diffs)
