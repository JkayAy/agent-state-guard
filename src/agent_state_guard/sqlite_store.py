"""SQLite-backed TransitionStore.

This is the zero-dependency default: it works anywhere Python's stdlib
works, with no live database required, which makes it the right choice
for local development, tests, and any deployment that doesn't have (or
need) Supabase/Postgres running.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

from .schemas import StateTransitionRecord
from .store import TransitionStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS state_transitions (
    record_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    node_name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_transitions_run_id
    ON state_transitions (run_id, step, attempt);
"""


class SqliteTransitionStore(TransitionStore):
    """Persists StateTransitionRecords to a local SQLite file (or ":memory:").

    Each record is stored as its full validated JSON payload (via
    `model_dump_json()`), plus a handful of denormalized columns
    (run_id, step, node_name, status) purely to make querying and
    ordering fast -- the payload column is always the source of truth.
    """

    def __init__(self, path: Union[str, Path] = "agent_state_guard.sqlite3") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, record: StateTransitionRecord) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO state_transitions
                (record_id, run_id, step, attempt, node_name, status, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.run_id,
                record.step,
                record.attempt,
                record.node_name,
                record.status.value,
                record.created_at.isoformat(),
                record.model_dump_json(),
            ),
        )
        self._conn.commit()

    def load(self, run_id: str) -> list[StateTransitionRecord]:
        rows = self._conn.execute(
            """
            SELECT payload FROM state_transitions
            WHERE run_id = ?
            ORDER BY step ASC, attempt ASC
            """,
            (run_id,),
        ).fetchall()
        return [StateTransitionRecord.model_validate_json(row[0]) for row in rows]

    def all_run_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT run_id FROM state_transitions ORDER BY run_id"
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteTransitionStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
