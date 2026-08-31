"""Tests for agent_state_guard.retry."""
from __future__ import annotations

import pytest

from agent_state_guard import MaxRetriesExceededError, RetryPolicy


def test_defaults_are_sane():
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.delay_for_attempt(1) == 0.0


def test_delay_grows_and_is_capped():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=3.0, jitter_s=0.0)
    assert policy.delay_for_attempt(1) == 0.0
    assert policy.delay_for_attempt(2) == 1.0
    assert policy.delay_for_attempt(3) == 2.0
    assert policy.delay_for_attempt(4) == 3.0  # capped
    assert policy.delay_for_attempt(10) == 3.0


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_s=-1)


def test_run_returns_first_success():
    calls = []

    def flaky():
        calls.append(1)
        return "ok"

    policy = RetryPolicy(sleep_fn=lambda _s: None)
    assert policy.run(flaky, context="test") == "ok"
    assert len(calls) == 1


def test_run_retries_then_succeeds():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("not yet")
        return "ok"

    policy = RetryPolicy(max_attempts=3, sleep_fn=lambda _s: None)
    assert policy.run(flaky, context="test") == "ok"
    assert attempts["count"] == 2


def test_run_exhausts_and_raises():
    def always_fails():
        raise RuntimeError("nope")

    policy = RetryPolicy(max_attempts=2, sleep_fn=lambda _s: None)
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        policy.run(always_fails, context="test-op")
    assert "test-op" in str(exc_info.value)
    assert "attempt 1" in str(exc_info.value)
    assert "attempt 2" in str(exc_info.value)
