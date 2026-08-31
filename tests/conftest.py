"""Shared pytest fixtures for AGENT-STATE-GUARD's test suite."""
from __future__ import annotations

import pytest

from agent_state_guard import AgentState


@pytest.fixture
def make_state():
    """Factory fixture for building a minimal valid AgentState."""

    def _make(**overrides):
        defaults = {"task": "demo-task"}
        defaults.update(overrides)
        return AgentState(**defaults)

    return _make
