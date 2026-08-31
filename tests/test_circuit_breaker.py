"""Tests for agent_state_guard.circuit_breaker."""
from __future__ import annotations

import pytest

from agent_state_guard import CircuitBreaker, CircuitBreakerOpenError, CircuitState


class FakeClock:
    """A manually-advanced clock so tests never rely on real wall time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_closed():
    breaker = CircuitBreaker()
    assert breaker.state == CircuitState.CLOSED
    breaker.before_call()  # should not raise


def test_opens_after_threshold_failures():
    breaker = CircuitBreaker(failure_threshold=3)
    for _ in range(2):
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpenError):
        breaker.before_call(context="my-tool")


def test_moves_to_half_open_after_timeout_then_recovers():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10.0, clock=clock)
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    clock.advance(5.0)
    assert breaker.state == CircuitState.OPEN

    clock.advance(6.0)
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_half_open_failure_reopens_immediately():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(11.0)
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_reset_forces_closed():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    breaker.reset()
    assert breaker.state == CircuitState.CLOSED


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(reset_timeout_s=-1)
