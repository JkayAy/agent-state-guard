"""Tests for agent_state_guard.schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_state_guard import (
    AgentState,
    NodeStatus,
    StateTransitionRecord,
    ToolCallRequest,
    ToolCallResult,
)


def test_node_status_values():
    assert NodeStatus.PENDING == "pending"
    assert NodeStatus.SUCCEEDED == "succeeded"
    assert NodeStatus.FALLBACK == "fallback"


def test_tool_call_request_defaults_and_validation():
    req = ToolCallRequest(tool_name="search.web")
    assert req.call_id
    assert req.arguments == {}

    with pytest.raises(ValidationError):
        ToolCallRequest(tool_name="bad name!")


def test_tool_call_request_is_frozen():
    req = ToolCallRequest(tool_name="search")
    with pytest.raises(ValidationError):
        req.tool_name = "other"


def test_tool_call_request_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ToolCallRequest(tool_name="search", unexpected="nope")


def test_tool_call_result_requires_non_negative_latency():
    ToolCallResult(call_id="c1", tool_name="search", success=True, latency_ms=0)
    with pytest.raises(ValidationError):
        ToolCallResult(call_id="c1", tool_name="search", success=True, latency_ms=-1)


def test_agent_state_forbids_extra_fields():
    with pytest.raises(ValidationError):
        AgentState(task="demo", unexpected_field=True)


def test_agent_state_validates_on_assignment():
    state = AgentState(task="demo")
    with pytest.raises(ValidationError):
        state.step = -1


def test_state_transition_record_is_frozen_and_bounded():
    record = StateTransitionRecord(
        run_id="run-1",
        step=0,
        node_name="n1",
        status=NodeStatus.SUCCEEDED,
        input_hash="abc",
        duration_ms=1.0,
    )
    with pytest.raises(ValidationError):
        record.step = 5
    with pytest.raises(ValidationError):
        StateTransitionRecord(
            run_id="run-1",
            step=0,
            node_name="n1",
            status=NodeStatus.SUCCEEDED,
            input_hash="abc",
            duration_ms=-1.0,
        )
