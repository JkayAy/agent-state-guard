"""PostgreSQL/Supabase-backed TransitionStore.

Requires the optional `postgres` extra
(`pip install agent-state-guard[postgres]`, which installs `psycopg`). This
mirrors SqliteTransitionStore's schema and behavior exactly, so swapping
backends never changes what a caller can rely on -- only where the audit
log physically lives.
"""
from __future__ import annotations

from .schemas import StateTransitionRecord
from .store import TransitionStore

try:
    import psycopg  # type: ignore

    PSYCOPG_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without psycopg
    PSYCOPG_AVAILABLE = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS state_transitions (
    record_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    node_name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_transitions_run_id
    ON state_transitions (run_id, step, attempt);
"""


class PostgresTransitionStore(TransitionStore):
    """Persists StateTransitionRecords to Postgres (including Supabase).

    Attributes:
        conninfo: A libpq connection string / DSN, e.g.
            "postgresql://user:pass@host:5432/dbname".
    """

    def __init__(self, conninfo: str) -> None:
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "psycopg is not installed; install the 'postgres' extra "
                "(pip install agent-state-guard[postgres]) to use "
                "PostgresTransitionStore, or use SqliteTransitionStore instead."
            )
        self._conninfo = conninfo
        self._conn = psycopg.connect(conninfo, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA)

    def save(self, record: StateTransitionRecord) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO state_transitions
                    (record_id, run_id, step, attempt, node_name, status, created_at, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (record_id) DO UPDATE SET payload = EXCLUDED.payload
                """,
                (
                    record.record_id,
                    record.run_id,
                    record.step,
                    record.attempt,
                    record.node_name,
                    record.status.value,
                    record.created_at,
                    record.model_dump_json(),
                ),
            )

    def load(self, run_id: str) -> list[StateTransitionRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload FROM state_transitions
                WHERE run_id = %s
                ORDER BY step ASC, attempt ASC
                """,
                (run_id,),
            )
            rows = cur.fetchall()
        return [StateTransitionRecord.model_validate_json(row[0]) for row in rows]

    def all_run_ids(self) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT run_id FROM state_transitions ORDER BY run_id")
            rows = cur.fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PostgresTransitionStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
