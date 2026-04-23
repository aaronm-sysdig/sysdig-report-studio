"""Append rows to ingest_log for a given snapshot_id."""
from __future__ import annotations

from datetime import datetime, timezone


def log_stage(conn, *, snapshot_id: str, stage: str, rows_affected: int,
              duration_ms: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_log (snapshot_id, stage, rows_affected, duration_ms, logged_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (snapshot_id, stage) DO UPDATE SET
          rows_affected = EXCLUDED.rows_affected,
          duration_ms = EXCLUDED.duration_ms,
          logged_at = EXCLUDED.logged_at
        """,
        [snapshot_id, stage, rows_affected, duration_ms,
         datetime.now(timezone.utc)],
    )
