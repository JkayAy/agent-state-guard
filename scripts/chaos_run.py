"""Chaos/failure-injection demo for AGENT-STATE-GUARD.

Exercises DeterministicGraph's retry and fallback mechanics against
deliberately unreliable "tools" to show what the audit trail looks like
when things go wrong. Run directly with:

    python -m scripts.chaos_run

or import the scenario functions from tests to assert on their behavior
programmatically (see tests/test_chaos_script.py).
"""
from __future__ import annotations

from agent_state_guard import (
    AgentState,
    CircuitBreaker,
    DeterministicGraph,
    RetryPolicy,
    ToolCallRequest,
    invoke_tool,
)


def make_flaky_tool(fail_times: int):
    """Return a tool function that fails fail_times times, then succeeds."""

    calls = {"n": 0}

    def flaky(query: str) -> str:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ConnectionError(f"backend unreachable (attempt {calls['n']})")
        return f"3 results for {query!r}"

    return flaky


def always_down_tool(query: str) -> str:
    raise ConnectionError("backend permanently unreachable")


def _search_node_factory(tool_fn, breaker):
    def search_node(state: AgentState) -> AgentState:
        request = ToolCallRequest(tool_name="web_search", arguments={"query": state.task})
        result = invoke_tool(
            request,
            tool_fn,
            retry_policy=RetryPolicy(max_attempts=3, sleep_fn=lambda _s: None),
            circuit_breaker=breaker,
        )
        if not result.success:
            raise RuntimeError(result.error)
        return state.model_copy(update={"scratchpad": {"search_result": result.output}})

    return search_node


def run_recoverable_scenario() -> dict:
    """A tool that fails twice then recovers -- retries alone save the run."""

    tool_fn = make_flaky_tool(fail_times=2)
    breaker = CircuitBreaker(failure_threshold=10)
    graph = DeterministicGraph()
    graph.add_node(
        "search",
        _search_node_factory(tool_fn, breaker),
        retry_policy=RetryPolicy(max_attempts=2, sleep_fn=lambda _s: None),
    )
    result = graph.run(AgentState(task="agent orchestration research"))
    return {
        "final_scratchpad": result.final_state.scratchpad,
        "transition_statuses": [t.status.value for t in result.transitions],
    }


def run_fallback_scenario() -> dict:
    """A tool that never recovers -- retries exhaust and the fallback node takes over."""

    breaker = CircuitBreaker(failure_threshold=10)
    graph = DeterministicGraph()
    graph.add_node(
        "search",
        _search_node_factory(always_down_tool, breaker),
        retry_policy=RetryPolicy(max_attempts=2, sleep_fn=lambda _s: None),
        fallback="offline_fallback",
    )

    def offline_fallback(state: AgentState) -> AgentState:
        return state.model_copy(
            update={"scratchpad": {"search_result": "offline cache: no live results available"}}
        )

    graph.add_node("offline_fallback", offline_fallback)
    result = graph.run(AgentState(task="agent orchestration research"))
    return {
        "final_scratchpad": result.final_state.scratchpad,
        "transition_statuses": [t.status.value for t in result.transitions],
    }


def main() -> dict:
    """Run both chaos scenarios and return a combined summary."""

    return {
        "recoverable": run_recoverable_scenario(),
        "fallback": run_fallback_scenario(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2))
