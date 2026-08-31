"""Tests for agent_state_guard.exceptions."""
from __future__ import annotations

from agent_state_guard import (
    AgentStateGuardError,
    CircuitBreakerOpenError,
    MaxRetriesExceededError,
    NodeStatus,
    NonDeterministicStateError,
    SchemaViolationError,
    StateTransitionRecord,
    ToolCallValidationError,
)


def test_base_error_stores_message_and_optional_record():
    err = AgentStateGuardError("boom")
    assert str(err) == "boom"
    assert err.record is None

    record = StateTransitionRecord(
        run_id="r1",
        step=0,
        node_name="n1",
        status=NodeStatus.FAILED,
        input_hash="abc",
        duration_ms=1.0,
    )
    err_with_record = AgentStateGuardError("boom", record=record)
    assert err_with_record.record is record


def test_all_subclasses_inherit_base():
    for cls in (
        SchemaViolationError,
        ToolCallValidationError,
        NonDeterministicStateError,
        CircuitBreakerOpenError,
        MaxRetriesExceededError,
    ):
        instance = cls("failure")
        assert isinstance(instance, AgentStateGuardError)
        assert str(instance) == "failure"
