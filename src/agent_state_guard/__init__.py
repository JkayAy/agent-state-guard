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
from .postgres_store import PostgresTransitionStore
from .replay import (
    ChainIntegrityResult,
    ChainMismatch,
    ReplayDiff,
    ReplayResult,
    replay_and_diff,
    verify_chain_integrity,
)
from .retry import RetryPolicy
from .schemas import (
    AgentState,
    NodeStatus,
    StateTransitionRecord,
    ToolCallRequest,
    ToolCallResult,
)
from .sqlite_store import SqliteTransitionStore
from .store import TransitionStore
from .tool_wrapper import invoke_tool

__version__ = "0.3.0"

__all__ = [
    "AgentState",
    "AgentStateGuardError",
    "ChainIntegrityResult",
    "ChainMismatch",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "DeterministicGraph",
    "ExecutionResult",
    "MaxRetriesExceededError",
    "NodeStatus",
    "NonDeterministicStateError",
    "PostgresTransitionStore",
    "ReplayDiff",
    "ReplayResult",
    "RetryPolicy",
    "SchemaViolationError",
    "SqliteTransitionStore",
    "StateTransitionRecord",
    "ToolCallRequest",
    "ToolCallResult",
    "ToolCallValidationError",
    "TransitionStore",
    "invoke_tool",
    "replay_and_diff",
    "stable_hash",
    "verify_chain_integrity",
]
