"""Tests for agent_state_guard.tool_wrapper."""
from __future__ import annotations

import pytest

from agent_state_guard import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    ToolCallRequest,
    ToolCallValidationError,
    invoke_tool,
)


def test_invoke_tool_requires_tool_call_request():
    with pytest.raises(ToolCallValidationError):
        invoke_tool("not-a-request", lambda: None)


def test_invoke_tool_success():
    request = ToolCallRequest(tool_name="add", arguments={"a": 1, "b": 2})

    def add(a, b):
        return a + b

    result = invoke_tool(request, add)
    assert result.success is True
    assert result.output == 3
    assert result.attempt == 1
    assert result.error is None


def test_invoke_tool_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(a):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("boom")
        return a * 2

    request = ToolCallRequest(tool_name="double", arguments={"a": 5})
    policy = RetryPolicy(max_attempts=3, sleep_fn=lambda _s: None)
    result = invoke_tool(request, flaky, retry_policy=policy)
    assert result.success is True
    assert result.output == 10
    assert result.attempt == 2


def test_invoke_tool_exhausts_retries_and_reports_failure():
    def always_fails():
        raise RuntimeError("nope")

    request = ToolCallRequest(tool_name="broken")
    policy = RetryPolicy(max_attempts=2, sleep_fn=lambda _s: None)
    result = invoke_tool(request, always_fails, retry_policy=policy)
    assert result.success is False
    assert result.output is None
    assert "broken" in result.error
    assert result.attempt == 2


def test_invoke_tool_short_circuits_when_breaker_open():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()  # opens the circuit

    called = {"n": 0}

    def fn():
        called["n"] += 1
        return "should not run"

    request = ToolCallRequest(tool_name="guarded")
    result = invoke_tool(request, fn, circuit_breaker=breaker)
    assert result.success is False
    assert result.attempt == 0
    assert called["n"] == 0


def test_invoke_tool_feeds_circuit_breaker_on_failure():
    breaker = CircuitBreaker(failure_threshold=2)
    request = ToolCallRequest(tool_name="flaky")

    def always_fails():
        raise RuntimeError("boom")

    invoke_tool(request, always_fails, circuit_breaker=breaker)
    assert breaker.state == CircuitState.CLOSED  # only 1 failure, threshold is 2

    invoke_tool(request, always_fails, circuit_breaker=breaker)
    assert breaker.state == CircuitState.OPEN
