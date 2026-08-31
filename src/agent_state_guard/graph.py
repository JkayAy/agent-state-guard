"""Deterministic state-graph orchestration with retry and fallback mechanics.

`DeterministicGraph` wraps a sequence of node functions so that:

1. Every node's output is validated against `AgentState` before it is
   allowed to become the next state. Schema violations raise immediately
   -- they are never silently coerced, logged-and-ignored, or allowed to
   corrupt downstream state.
2. Every node execution produces an immutable `StateTransitionRecord`
   describing what ran, how long it took, and a content hash of the state
   before and after. This is the basis for the audit log and replay
   mechanics implemented in later modules.
3. A node may be registered with a `RetryPolicy` and/or a `fallback` node
   name. On failure, the node is retried according to its policy; if every
   attempt fails, execution routes to the fallback node (if one is
   registered) instead of aborting the run. Every attempt -- including
   failed ones and the fallback invocation -- produces its own
   `StateTransitionRecord`, so the audit log always shows the complete
   sequence of what was tried, not just the final outcome.

If LangGraph is installed, `to_langgraph()` compiles the same node registry
into a real `langgraph.graph.StateGraph`, so this library's guarantees hold
whether execution is driven by the built-in sequential runner or by
LangGraph itself. LangGraph is an optional dependency, never a requirement
for the core guarantees.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .exceptions import MaxRetriesExceededError, SchemaViolationError
from .retry import RetryPolicy
from .schemas import AgentState, NodeStatus, StateTransitionRecord

try:
    from langgraph.graph import END, StateGraph  # type: ignore

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without langgraph
    LANGGRAPH_AVAILABLE = False

NodeFn = Callable[[AgentState], AgentState]


def stable_hash(state: AgentState) -> str:
    """Deterministic content hash of a state snapshot.

    Dumps with `mode="json"` and sorted keys so field ordering, dict
    insertion order, and datetime serialization never affect the hash --
    the same logical state always hashes identically, which is what makes
    replay verification meaningful.
    """
    payload = state.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class NodeSpec:
    name: str
    fn: NodeFn
    retry_policy: Optional[RetryPolicy] = None
    fallback: Optional[str] = None


@dataclass
class ExecutionResult:
    final_state: AgentState
    transitions: list[StateTransitionRecord] = field(default_factory=list)


class DeterministicGraph:
    """A registry of schema-validated nodes executed in a fixed order."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeSpec] = {}
        self._order: list[str] = []

    @property
    def node_names(self) -> list[str]:
        return list(self._order)

    def add_node(
        self,
        name: str,
        fn: NodeFn,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        fallback: Optional[str] = None,
    ) -> "DeterministicGraph":
        if not name:
            raise ValueError("node name must be non-empty")
        if name in self._nodes:
            raise ValueError(f"node {name!r} is already registered")
        self._nodes[name] = NodeSpec(
            name=name, fn=fn, retry_policy=retry_policy, fallback=fallback
        )
        self._order.append(name)
        return self

    def _run_attempt(
        self, spec: NodeSpec, state: AgentState, *, attempt: int
    ) -> tuple[AgentState, StateTransitionRecord]:
        """Run a single attempt of `spec` against `state`.

        Returns the validated resulting state and its transition record on
        success. Raises `SchemaViolationError` (with a `record` attached)
        on any failure -- an execution exception, a non-`AgentState`
        return value, or a value that fails re-validation.
        """
        input_hash = stable_hash(state)
        start = time.perf_counter()

        try:
            candidate = spec.fn(state)
        except Exception as exc:  # noqa: BLE001 - converted to a typed, recorded error below
            duration_ms = (time.perf_counter() - start) * 1000
            record = StateTransitionRecord(
                run_id=state.run_id,
                step=state.step,
                node_name=spec.name,
                status=NodeStatus.FAILED,
                input_hash=input_hash,
                error=f"{type(exc).__name__}: {exc}",
                attempt=attempt,
                duration_ms=duration_ms,
            )
            raise SchemaViolationError(
                f"node {spec.name!r} raised {type(exc).__name__}: {exc}",
                record=record,
            ) from exc

        if not isinstance(candidate, AgentState):
            duration_ms = (time.perf_counter() - start) * 1000
            record = StateTransitionRecord(
                run_id=state.run_id,
                step=state.step,
                node_name=spec.name,
                status=NodeStatus.FAILED,
                input_hash=input_hash,
                error=f"returned {type(candidate).__name__}, expected AgentState",
                attempt=attempt,
                duration_ms=duration_ms,
            )
            raise SchemaViolationError(
                f"node {spec.name!r} returned {type(candidate).__name__}, "
                "expected AgentState",
                record=record,
            )

        try:
            # Re-validate through the model even though `candidate` is
            # already an AgentState instance: this guards against a node
            # mutating fields in a way that bypasses validate_assignment,
            # or returning a subtly malformed model built by hand.
            validated = AgentState.model_validate(candidate.model_dump())
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start) * 1000
            record = StateTransitionRecord(
                run_id=state.run_id,
                step=state.step,
                node_name=spec.name,
                status=NodeStatus.FAILED,
                input_hash=input_hash,
                error=f"output failed re-validation: {exc}",
                attempt=attempt,
                duration_ms=duration_ms,
            )
            raise SchemaViolationError(
                f"node {spec.name!r} output failed re-validation: {exc}",
                record=record,
            ) from exc

        duration_ms = (time.perf_counter() - start) * 1000
        record = StateTransitionRecord(
            run_id=validated.run_id,
            step=validated.step,
            node_name=spec.name,
            status=NodeStatus.SUCCEEDED,
            input_hash=input_hash,
            output_hash=stable_hash(validated),
            attempt=attempt,
            duration_ms=duration_ms,
        )
        return validated, record

    def _execute_node(
        self, spec: NodeSpec, state: AgentState
    ) -> tuple[AgentState, list[StateTransitionRecord]]:
        """Execute `spec` with its retry policy and fallback routing.

        Returns the resulting state and the complete list of transition
        records produced along the way -- every failed attempt, plus
        either the eventual success or the fallback invocation's records.
        Only raises if every attempt fails and no fallback is registered
        (or the fallback itself fails).
        """
        policy = spec.retry_policy or RetryPolicy(max_attempts=1)
        records: list[StateTransitionRecord] = []
        last_error: Optional[SchemaViolationError] = None

        for attempt in range(1, policy.max_attempts + 1):
            delay = policy.delay_for_attempt(attempt)
            if delay > 0:
                policy.sleep_fn(delay)
            try:
                validated, record = self._run_attempt(spec, state, attempt=attempt)
                records.append(record)
                return validated, records
            except SchemaViolationError as exc:
                last_error = exc
                if exc.record is not None:
                    records.append(exc.record)

        assert last_error is not None  # at least one attempt always runs

        if spec.fallback is not None:
            if spec.fallback not in self._nodes:
                raise SchemaViolationError(
                    f"node {spec.name!r} declares unknown fallback {spec.fallback!r}",
                    record=last_error.record,
                ) from last_error
            fallback_spec = self._nodes[spec.fallback]
            fallback_state, fallback_records = self._execute_node(fallback_spec, state)
            # Re-tag the fallback's successful record as FALLBACK so the
            # audit log distinguishes "this is the primary path" from
            # "this ran because the primary path exhausted its retries".
            retagged = [
                r.model_copy(update={"status": NodeStatus.FALLBACK})
                if r.status == NodeStatus.SUCCEEDED
                else r
                for r in fallback_records
            ]
            records.extend(retagged)
            return fallback_state, records

        raise MaxRetriesExceededError(
            f"node {spec.name!r} exhausted {policy.max_attempts} attempt(s) "
            "with no fallback registered",
            record=last_error.record,
        ) from last_error

    def run(self, initial_state: AgentState) -> ExecutionResult:
        """Execute all registered nodes in registration order.

        This sequential executor is the deterministic reference
        implementation: graphs compiled by `to_langgraph()` are validated
        against it in tests to confirm both execution paths produce
        identical final states for identical inputs.
        """
        state = initial_state
        transitions: list[StateTransitionRecord] = []
        for name in self._order:
            spec = self._nodes[name]
            state, node_records = self._execute_node(spec, state)
            transitions.extend(node_records)
            state = state.model_copy(update={"step": state.step + 1})
            if state.terminal:
                break
        return ExecutionResult(final_state=state, transitions=transitions)

    def to_langgraph(self):
        """Compile this node registry into a LangGraph `StateGraph`.

        Requires the optional `langgraph` dependency
        (`pip install agent-state-guard[langgraph]`). Each node is wrapped
        with the same validation, retry, and fallback logic used by
        `run()`, so schema guarantees are identical regardless of which
        executor drives the graph.
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is not installed; install the 'langgraph' extra "
                "(pip install agent-state-guard[langgraph]) or use .run() "
                "instead, which has no LangGraph dependency."
            )

        graph = StateGraph(AgentState)
        for name in self._order:
            spec = self._nodes[name]

            def _wrapped(state: AgentState, _spec: NodeSpec = spec) -> AgentState:
                validated, _records = self._execute_node(_spec, state)
                return validated

            graph.add_node(name, _wrapped)

        for a, b in zip(self._order, self._order[1:]):
            graph.add_edge(a, b)
        if self._order:
            graph.set_entry_point(self._order[0])
            graph.add_edge(self._order[-1], END)
        return graph.compile()
