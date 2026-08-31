"""Schema-enforced tool-call execution with retry and circuit-breaker
protection.

`invoke_tool` is the single sanctioned way node implementations should
call external tools: it validates the request, enforces the circuit
breaker, applies the retry policy, times execution, and always returns a
`ToolCallResult` -- callers never need to catch tool exceptions directly,
which keeps failure handling uniform and auditable.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .circuit_breaker import CircuitBreaker
from .exceptions import ToolCallValidationError
from .retry import RetryPolicy
from .schemas import ToolCallRequest, ToolCallResult


def invoke_tool(
    request: ToolCallRequest,
    fn: Callable[..., Any],
    *,
    retry_policy: Optional[RetryPolicy] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> ToolCallResult:
    """Invoke a tool callable under schema validation, retry, and
    circuit-breaker rules.

    Args:
        request: A validated ToolCallRequest describing the call.
        fn: The underlying tool implementation, invoked as
            fn(**request.arguments).
        retry_policy: Optional RetryPolicy; defaults to a single attempt
            with no backoff.
        circuit_breaker: Optional CircuitBreaker guarding this tool. If the
            circuit is open, the call is short-circuited without invoking
            fn at all.

    Returns:
        A ToolCallResult. All failure modes -- exceptions raised by fn,
        circuit-open short-circuits, and retry exhaustion -- are captured
        in the result rather than raised, so callers get a uniform,
        always-present audit record instead of having to catch multiple
        exception types.
    """
    if not isinstance(request, ToolCallRequest):
        raise ToolCallValidationError("invoke_tool requires a validated ToolCallRequest")

    start = time.perf_counter()

    if circuit_breaker is not None:
        try:
            circuit_breaker.before_call(context=request.tool_name)
        except Exception as exc:  # noqa: BLE001 - CircuitBreakerOpenError, captured not raised
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                success=False,
                output=None,
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
                attempt=0,
            )

    policy = retry_policy or RetryPolicy(max_attempts=1)
    attempts_used = 0

    def _attempt() -> Any:
        nonlocal attempts_used
        attempts_used += 1
        try:
            result = fn(**request.arguments)
        except Exception:
            if circuit_breaker is not None:
                circuit_breaker.record_failure()
            raise
        if circuit_breaker is not None:
            circuit_breaker.record_success()
        return result

    try:
        output = policy.run(_attempt, context=request.tool_name)
    except Exception as exc:  # noqa: BLE001 - MaxRetriesExceededError, captured not raised
        return ToolCallResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            success=False,
            output=None,
            error=str(exc),
            latency_ms=(time.perf_counter() - start) * 1000,
            attempt=attempts_used,
        )

    return ToolCallResult(
        call_id=request.call_id,
        tool_name=request.tool_name,
        success=True,
        output=output,
        error=None,
        latency_ms=(time.perf_counter() - start) * 1000,
        attempt=attempts_used,
    )
