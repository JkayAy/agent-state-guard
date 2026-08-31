"""AGENT-STATE-GUARD

Deterministic state-graph orchestration and fallback mechanics for
non-deterministic enterprise multi-agent workflows.
"""
from .circuit_breaker import CircuitBreaker, CircuitState
from .exceptions import (
    AgentStateGuardError,
    CircuitBreakerOpenError,
    MaxRetriesExceededError,
    NonDeterministicStateError,
    SchemaViolationError,
    ToolCallValidationError,
)
from .graph import DeterministicGraph, ExecutionResult, stable_hash
from .retry import RetryPolicy
from .schemas import (
    AgentState,
    NodeStatus,
    StateTransitionRecord,
    ToolCallRequest,
    ToolCallResult,
)
from .tool_wrapper import invoke_tool

__version__ = "0.2.0"

__all__ = [
    "AgentState",
    "AgentStateGuardError",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "DeterministicGraph",
    "ExecutionResult",
    "MaxRetriesExceededError",
    "NodeStatus",
    "NonDeterministicStateError",
    "RetryPolicy",
    "SchemaViolationError",
    "StateTransitionRecord",
    "ToolCallRequest",
    "ToolCallResult",
    "ToolCallValidationError",
    "invoke_tool",
    "stable_hash",
]
