"""Smoke tests for scripts/chaos_run.py.

These import and execute the real demo scenarios (no mocking) so the
chaos script itself stays exercised by CI rather than silently rotting.
"""
from __future__ import annotations

from scripts.chaos_run import main, run_fallback_scenario, run_recoverable_scenario


def test_recoverable_scenario_succeeds_via_retry():
    result = run_recoverable_scenario()
    assert result["final_scratchpad"]["search_result"].startswith("3 results for")
    assert "succeeded" in result["transition_statuses"]


def test_fallback_scenario_falls_back_when_tool_never_recovers():
    result = run_fallback_scenario()
    assert result["final_scratchpad"]["search_result"] == (
        "offline cache: no live results available"
    )
    assert "fallback" in result["transition_statuses"]
    assert "failed" in result["transition_statuses"]


def test_main_runs_both_scenarios_without_raising():
    summary = main()
    assert set(summary.keys()) == {"recoverable", "fallback"}
