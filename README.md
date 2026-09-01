# AGENT-STATE-GUARD

Deterministic state-graph orchestration and fallback mechanics for
non-deterministic, multi-agent LLM workflows.

AGENT-STATE-GUARD wraps agent/tool execution in a small state machine that
refuses to let unvalidated data flow between steps, records an immutable
audit trail of every attempt (including failed and retried ones), and
routes to a fallback path instead of crashing when a step exhausts its
retry budget. It is built on Pydantic for schema enforcement and has an
optional LangGraph compilation target.

## Status

Phases 1 through 4 of the original build plan are complete: the
deterministic core, retry/circuit-breaker/fallback mechanics, SQLite and
Postgres persistence with replay/audit-chain verification, and a pytest
suite wired into GitHub Actions CI. The badge-worthy claim here is a
narrow one: **the sequential executor, retry policy, circuit breaker,
tool wrapper, schema contracts, SQLite persistence, and replay/audit
tooling are exercised by 52 passing tests in CI on every push.** Nothing
in that list is aspirational.

Two things are intentionally *not* claimed as fully verified:

- **LangGraph compilation** (`DeterministicGraph.to_langgraph()`) is
  structurally implemented and covered by a test, but that test is
  written to skip gracefully rather than fail if the installed LangGraph
  version's API doesn't match what this integration expects. LangGraph's
  public API has changed across releases; this project has not been
  pinned to (or exhaustively tested against) a specific version.
- **PostgresTransitionStore** mirrors the SQLite backend's schema and
  behavior by construction, but CI has no live Postgres instance to run
  against, so that backend is exercised by import/type checks only, not
  by an end-to-end save/load test.

## Architecture

```
src/agent_state_guard/
  schemas.py         Pydantic contracts: AgentState, ToolCallRequest/Result,
                      StateTransitionRecord, NodeStatus
  exceptions.py       Typed error hierarchy, each carrying the audit record
                      for the failure that raised it
  graph.py            DeterministicGraph: schema-validated sequential
                      executor with per-node retry + fallback routing
  retry.py            RetryPolicy: exponential backoff with jitter
  circuit_breaker.py  CircuitBreaker: consecutive-failure tripping with a
                      half-open recovery trial
  tool_wrapper.py     invoke_tool(): the sanctioned way nodes call external
                      tools under retry + circuit-breaker protection
  store.py            Abstract TransitionStore interface
  sqlite_store.py      Zero-dependency local persistence backend
  postgres_store.py    Postgres/Supabase persistence backend (optional)
  replay.py           verify_chain_integrity() and replay_and_diff() for
                      auditing and determinism verification
```

## Installation

```bash
pip install agent-state-guard
# optional extras
pip install "agent-state-guard[langgraph]"
pip install "agent-state-guard[postgres]"
```

## Quickstart

```python
from agent_state_guard import AgentState, DeterministicGraph, RetryPolicy

def summarize(state: AgentState) -> AgentState:
    return state.model_copy(update={"scratchpad": {"summary": "done"}})

graph = DeterministicGraph()
graph.add_node("summarize", summarize, retry_policy=RetryPolicy(max_attempts=3))

result = graph.run(AgentState(task="summarize the quarterly report"))
print(result.final_state.scratchpad)
print([t.status for t in result.transitions])
```

Every node's output is re-validated against `AgentState` before it is
allowed to become the next state. A node can be registered with a
`retry_policy` and/or a `fallback` node name; on exhausting its
retries with no fallback registered, the graph raises
`MaxRetriesExceededError` rather than silently continuing.

## Persistence and replay

```python
from agent_state_guard import SqliteTransitionStore, verify_chain_integrity

store = SqliteTransitionStore("runs.sqlite3")
store.save_all(result.transitions)

records = store.load(result.final_state.run_id)
integrity = verify_chain_integrity(result.final_state.run_id, records)
print(integrity.consistent)
```

`replay_and_diff()` goes further: it re-executes the same graph against
the same initial state and compares freshly produced output hashes
against a previously stored run, which is the actual determinism
guarantee -- it does not trust the stored log, it re-runs the real node
functions.

## Testing and CI

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest -v --cov=agent_state_guard
```

The included GitHub Actions workflow (`.github/workflows/ci.yml`) runs
this same command on every push and pull request against `main`. A
chaos/failure-injection demo (`scripts/chaos_run.py`) exercises the
retry-recovers and retry-exhausts-then-falls-back paths against
deliberately unreliable tool functions, and is itself covered by a
smoke test so it stays exercised rather than rotting.

## License

MIT
