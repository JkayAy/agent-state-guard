"""Abstract persistence interface for AGENT-STATE-GUARD audit records.

Every backend (SQLite, Postgres/Supabase, or a future one) implements the
same small interface so the rest of the library -- and any caller doing
replay or audit review -- never has to know which storage engine produced
a given `StateTransitionRecord`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import StateTransitionRecord


class TransitionStore(ABC):
    """Persists and retrieves StateTransitionRecords, keyed by run_id."""

    @abstractmethod
    def save(self, record: StateTransitionRecord) -> None:
        """Persist a single transition record.

        Implementations must never mutate or drop fields -- the record
        persisted must be reconstructable byte-for-byte via
        `StateTransitionRecord.model_validate_json(...)`.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, run_id: str) -> list[StateTransitionRecord]:
        """Return all records for a run, ordered by step then attempt."""
        raise NotImplementedError

    @abstractmethod
    def all_run_ids(self) -> list[str]:
        """Return every run_id with at least one persisted record."""
        raise NotImplementedError

    def save_all(self, records: list[StateTransitionRecord]) -> None:
        """Convenience helper: persist a full ExecutionResult.transitions list."""
        for record in records:
            self.save(record)
