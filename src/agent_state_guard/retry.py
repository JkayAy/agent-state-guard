"""Exponential-backoff retry policy for AGENT-STATE-GUARD.

A `RetryPolicy` never swallows failures silently: it either returns the
first successful result or raises `MaxRetriesExceededError` carrying every
attempt's error message, so callers get a complete audit trail instead of
just the final exception.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from .exceptions import MaxRetriesExceededError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff retry policy.

    Attributes:
        max_attempts: Total number of attempts (including the first), >= 1.
        base_delay_s: Delay before the second attempt, in seconds.
        max_delay_s: Upper bound on any single backoff delay.
        jitter_s: Maximum random jitter added to each delay.
        sleep_fn: Injectable sleep function (defaults to time.sleep). Tests
            can pass a no-op to keep the suite fast and deterministic.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter_s: float = 0.1
    sleep_fn: Callable[[float], None] = field(default=time.sleep, compare=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0 or self.jitter_s < 0:
            raise ValueError("delay values must be non-negative")

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the backoff delay before the given attempt (1-indexed).

        Attempt 1 never waits. Attempt 2 waits ~base_delay_s, attempt 3
        ~2x that, capped at max_delay_s, plus a small random jitter so
        concurrent callers don't retry in lockstep.
        """
        if attempt <= 1:
            return 0.0
        raw = self.base_delay_s * (2 ** (attempt - 2))
        capped = min(raw, self.max_delay_s)
        return capped + random.uniform(0.0, self.jitter_s)

    def run(self, fn: Callable[[], T], *, context: str = "operation") -> T:
        """Execute `fn` under this retry policy.

        Returns the first successful result. If every attempt fails,
        raises `MaxRetriesExceededError` with all attempt errors attached
        in the message (no per-attempt StateTransitionRecord here -- that
        level of detail is the graph executor's job, not this generic
        helper's).
        """
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            delay = self.delay_for_attempt(attempt)
            if delay > 0:
                self.sleep_fn(delay)
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - intentional broad catch, re-raised as typed error
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        raise MaxRetriesExceededError(
            f"{context} failed after {self.max_attempts} attempt(s): " + " | ".join(errors)
        )
