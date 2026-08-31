"""Circuit breaker for protecting downstream tools/services from repeated
failures.

Once a wrapped call fails `failure_threshold` times in a row, the breaker
opens and short-circuits further calls with `CircuitBreakerOpenError`
instead of letting them hit an already-struggling dependency. After
`reset_timeout_s` elapses, the breaker moves to a half-open trial state
where exactly one call is allowed through to test recovery.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .exceptions import CircuitBreakerOpenError


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """A simple consecutive-failure-counting circuit breaker.

    Attributes:
        failure_threshold: Consecutive failures before the circuit opens.
        reset_timeout_s: Seconds to wait after opening before allowing a
            single half-open trial call.
        clock: Injectable monotonic clock, defaults to time.monotonic.
    """

    failure_threshold: int = 5
    reset_timeout_s: float = 30.0
    clock: Callable[[], float] = field(default=time.monotonic, compare=False)

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _opened_at: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.reset_timeout_s < 0:
            raise ValueError("reset_timeout_s must be non-negative")

    @property
    def state(self) -> CircuitState:
        """Current state, lazily transitioning OPEN -> HALF_OPEN once the
        reset timeout has elapsed."""
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            if self.clock() - self._opened_at >= self.reset_timeout_s:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def before_call(self, *, context: str = "operation") -> None:
        """Raise CircuitBreakerOpenError if calls are currently blocked."""
        if self.state is CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"circuit open for {context!r}; retry after the cooldown window"
            )

    def record_success(self) -> None:
        """Reset the breaker to CLOSED after a successful call."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call, opening the circuit if the threshold is
        reached or if the failure happened during a half-open trial."""
        self._failure_count += 1
        if self._state is CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self.clock()

    def reset(self) -> None:
        """Force the breaker back to a clean CLOSED state."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None
