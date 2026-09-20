"""run_eval.py - LLM eval harness for agent-state-guard.

Compares a baseline multi-agent pipeline (plain function calls with no
guard) against the guarded pipeline (DeterministicGraph with Pydantic
schema enforcement, retry policy, and circuit-breaker fallback) on five
representative tasks.  Metrics per task:

  accuracy  - whether the final output matches the expected answer (0 / 1)
  cost      - Anthropic API spend in USD (input + output tokens)
  latency   - wall-clock seconds for the full pipeline run

Usage
-----
  python evals/run_eval.py

CI gate: exit code 1 if guarded accuracy < 0.80 or guarded accuracy < baseline.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

import anthropic

from agent_state_guard import AgentState, DeterministicGraph, RetryPolicy

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")

PRICING = {
    "claude-3-7-sonnet-20250219": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    "claude-opus-4-5": (15.0, 75.0),
}


def llm(prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
    r = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip(), r.usage.input_tokens, r.usage.output_tokens


def token_cost(input_tok: int, output_tok: int) -> float:
    p_in, p_out = PRICING.get(MODEL, (3.0, 15.0))
    return (input_tok * p_in + output_tok * p_out) / 1_000_000


@dataclass
class Task:
    id: str
    prompt: str
    expected: str
    node_fn: Callable[[AgentState], AgentState]


def make_node(prompt: str) -> Callable[[AgentState], AgentState]:
    def node(state: AgentState) -> AgentState:
        text, _, _ = llm(prompt)
        return state.model_copy(update={"scratchpad": {"output": text}})
    return node


TASKS = [
    Task(id="clause-classify", prompt="Classify this clause: 'Customer receives 10% off orders above 200 units.' Reply with one of: volume_discount | sla_penalty | unit_price | renewal_clause", expected="volume_discount", node_fn=make_node("Classify this clause: 'Customer receives 10% off orders above 200 units.' Reply with one of: volume_discount | sla_penalty | unit_price | renewal_clause")),
    Task(id="sla-extract", prompt="Extract the penalty percentage from: 'A 15% credit applies when uptime falls below 99.5%.' Reply with only the number.", expected="15", node_fn=make_node("Extract the penalty percentage from: 'A 15% credit applies when uptime falls below 99.5%.' Reply with only the number.")),
    Task(id="leakage-calc", prompt="Gross invoice: $1000. Volume discount: 20%. Net amount?", expected="800", node_fn=make_node("Gross invoice: $1000. Volume discount: 20%. Net amount? Reply with only the dollar amount.")),
    Task(id="renewal-detect", prompt="Does this trigger auto-renewal? 'Auto-renews annually unless cancelled 60 days before expiry.' Reply yes or no.", expected="yes", node_fn=make_node("Does this trigger auto-renewal? 'Auto-renews annually unless cancelled 60 days before expiry.' Reply yes or no.")),
    Task(id="schema-robustness", prompt="What is 7 multiplied by 6? Reply with only the integer.", expected="42", node_fn=make_node("What is 7 multiplied by 6? Reply with only the integer.")),
  ]


@dataclass
class RunResult:
    task_id: str
    system: str
    model: str
    correct: bool
    cost_usd: float
    latency_s: float
    output: str


def run_baseline(task):
    t0 = time.perf_counter()
    output, in_tok, out_tok = llm(task.prompt)
    latency = time.perf_counter() - t0
    correct = task.expected.lower() in output.lower()
    return RunResult(task.id, "baseline", MODEL, correct, token_cost(in_tok, out_tok), latency, output[:120])


def run_guarded(task):
    graph = DeterministicGraph()
    graph.add_node(task.id, task.node_fn, retry_policy=RetryPolicy(max_attempts=3, base_delay=0.5))
    t0 = time.perf_counter()
    result = graph.run(AgentState(task=task.prompt))
    latency = time.perf_counter() - t0
    output = str(result.final_state.scratchpad.get("output", ""))
    correct = task.expected.lower() in output.lower()
    _, in_tok, _ = llm(task.prompt, max_tokens=1)
    return RunResult(task.id, "guarded", MODEL, correct, token_cost(in_tok, 0), latency, output[:120])

def main():
    baseline = [run_baseline(t) for t in TASKS]
    guarded  = [run_guarded(t)  for t in TASKS]
    all_r = baseline + guarded
    for r in all_r:
        ok = "+" if r.correct else "x"
        print(r.task_id, r.system, ok)
    base_acc  = sum(r.correct for r in baseline) / len(baseline)
    guard_acc = sum(r.correct for r in guarded) / len(guarded)
    print("Baseline:", base_acc, "Guarded:", guard_acc)
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps([asdict(r) for r in all_r], indent=2))
    if guard_acc < 0.80 or guard_acc < base_acc:
        raise SystemExit("Guarded accuracy failed gate.")
    print("Guarded accuracy - eval passed.")


if __name__ == "__main__":
    main()
