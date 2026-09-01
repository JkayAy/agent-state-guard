"""Strict runtime contracts for AGENT-STATE-GUARD.

Every value that crosses a node boundary in the state graph must conform to
one of these Pydantic models. Nothing is allowed to flow through the graph
as a bare dict or an unchecked LLM string -- if it doesn't validate, the
graph raises instead of silently continuing with corrupted state.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    FALLBACK = "fallback"
    ABORTED = "aborted"


class ToolCallRequest(BaseModel):
    """A validated request to invoke a named tool with structured arguments.

    Frozen because a request should never be mutated after it is issued --
    retries create a *new* ToolCallRequest with an incremented attempt
    count on the corresponding ToolCallResult, not a mutated original.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_at: datetime = Field(default_factory=_utcnow)

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_be_identifier_like(cls, v: str) -> str:
        if not v.replace("_", "").replace(".", "").isalnum():
            raise ValueError(
                f"tool_name {v!r} must contain only letters, digits, "
                "underscores, or dots"
            )
        return v


class ToolCallResult(BaseModel):
    """The validated outcome of a single tool call attempt.

    attempt allows 0 as a sentinel meaning "no attempt was actually
    made" -- e.g. an open circuit breaker short-circuiting the call before
    ever invoking the underlying tool. Attempts that did run are numbered
    starting at 1, matching RetryPolicy's convention.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    latency_ms: float = Field(ge=0)
    attempt: int = Field(ge=0, default=1)


class AgentState(BaseModel):
    """The single canonical state object that flows through the graph.

    Nodes receive an AgentState and must return an AgentState (or raise).
    No node may smuggle extra fields through -- extra="forbid" means an
    unexpected key anywhere in the payload is a validation error, not a
    silently-ignored typo.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step: int = Field(ge=0, default=0)
    status: NodeStatus = NodeStatus.PENDING
    task: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_results: list[ToolCallResult] = Field(default_factory=list)
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    error_count: int = Field(ge=0, default=0)
    last_error: str | None = None
    terminal: bool = False


class StateTransitionRecord(BaseModel):
    """Immutable audit record of a single node execution attempt.

    This is what gets persisted (Postgres/Supabase, or the local sqlite
    fallback) so any run can be inspected and deterministically replayed
    after the fact. One record is emitted per attempt, including failed
    and fallback attempts, not just successful ones.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    step: int
    node_name: str
    status: NodeStatus
    input_hash: str
    output_hash: str | None = None
    error: str | None = None
    attempt: int = Field(ge=1, default=1)
    duration_ms: float = Field(ge=0)
    created_at: datetime = Field(default_factory=_utcnow)
