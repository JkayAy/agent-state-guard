"""Exception hierarchy for AGENT-STATE-GUARD.

Every error raised by this library is a subclass of `AgentStateGuardError`.
Errors raised during node execution carry the `StateTransitionRecord` that
describes the failed attempt (when one is available) so callers and the
audit/replay layer can inspect exactly what happened without parsing
exception strings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .schemas import StateTransitionRecord


class AgentStateGuardError(Exception):
    """Base class for all AGENT-STATE-GUARD errors."""

    def __init__(
        self, message: str, *, record: "Optional[StateTransitionRecord]" = None
    ) -> None:
        super().__init__(message)
        self.record = record


class SchemaViolationError(AgentStateGuardError):
    """Raised when a node's input or output fails strict schema validation."""


class ToolCallValidationError(AgentStateGuardError):
    """Raised when a tool call request or result fails schema/contract checks."""


class NonDeterministicStateError(AgentStateGuardError):
    """Raised when replaying a run produces a state hash that does not match
    the recorded audit log, indicating the node's behavior was not
    deterministic given the same input."""


class CircuitBreakerOpenError(AgentStateGuardError):
    """Raised when a tool's circuit breaker is open and calls are being
    short-circuited instead of hitting the underlying tool."""


class MaxRetriesExceededError(AgentStateGuardError):
    """Raised when a node exhausts its retry budget without producing valid
    output and no fallback handler is registered."""
