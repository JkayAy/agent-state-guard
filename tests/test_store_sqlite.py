"""Tests for agent_state_guard.sqlite_store."""
from __future__ import annotations

import pytest

from agent_state_guard import NodeStatus, SqliteTransitionStore, StateTransitionRecord


def make_record(
    run_id="run-1",
    step=0,
    attempt=1,
    node_name="n",
    status=NodeStatus.SUCCEEDED,
    output_hash="out",
):
    return StateTransitionRecord(
        run_id=run_id,
        step=step,
        node_name=node_name,
        status=status,
        input_hash="in",
        output_hash=output_hash,
        attempt=attempt,
        duration_ms=1.0,
    )


@pytest.fixture
def store():
    with SqliteTransitionStore(":memory:") as s:
        yield s


def test_save_and_load_round_trip(store):
    record = make_record()
    store.save(record)
    loaded = store.load("run-1")
    assert loaded == [record]


def test_load_orders_by_step_then_attempt(store):
    r_step1 = make_record(step=1, attempt=1, node_name="b")
    r_step0_attempt2 = make_record(step=0, attempt=2, node_name="a")
    r_step0_attempt1 = make_record(step=0, attempt=1, node_name="a")

    # Save out of order on purpose to exercise the ORDER BY clause.
    store.save(r_step1)
    store.save(r_step0_attempt2)
    store.save(r_step0_attempt1)

    loaded = store.load("run-1")
    assert [(r.step, r.attempt) for r in loaded] == [(0, 1), (0, 2), (1, 1)]


def test_all_run_ids_returns_distinct_sorted_ids(store):
    store.save(make_record(run_id="run-b"))
    store.save(make_record(run_id="run-a"))
    store.save(make_record(run_id="run-a", step=1))

    assert store.all_run_ids() == ["run-a", "run-b"]


def test_save_all_persists_every_record(store):
    records = [make_record(step=i) for i in range(3)]
    store.save_all(records)
    assert len(store.load("run-1")) == 3


def test_save_replaces_existing_row_with_same_record_id(store):
    record = make_record()
    store.save(record)
    updated = record.model_copy(update={"error": "retried"})
    # record_id is unchanged by model_copy, so this exercises INSERT OR REPLACE.
    store.save(updated)
    loaded = store.load("run-1")
    assert len(loaded) == 1
    assert loaded[0].error == "retried"
