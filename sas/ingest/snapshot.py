"""Snapshot identity and idempotency. A snapshot is one CSV ingest event."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path


_FILENAME_DATE_RE = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _hash_file_content(path: Path) -> str:
    """Streaming sha256 of file contents. Bounded memory regardless of file size."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_snapshot_id(path: Path, row_count: int) -> str:
    """Deterministic ID from filename + row count + content hash.

    Same CSV (same name, same row count, same byte content) → same ID.
    Different content under the same name → different ID, preventing
    silent collisions when daily exports share a fixed filename.
    """
    h = hashlib.sha256()
    h.update(path.name.encode("utf-8"))
    h.update(b"|")
    h.update(str(row_count).encode("utf-8"))
    h.update(b"|")
    content_hash = _hash_file_content(path)
    h.update(content_hash.encode("utf-8"))
    return h.hexdigest()[:32]


def extract_snapshot_at(path: Path) -> datetime:
    """Parse a YYYY_MM_DD from the filename; fall back to now() if absent.

    Convention (per spec): reports are pulled at 12:00 UTC, so anchor to 12:00.
    """
    m = _FILENAME_DATE_RE.search(path.name)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, 12, 0, tzinfo=timezone.utc)
    return _now_utc()


def is_already_ingested(conn, snapshot_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM snapshot WHERE snapshot_id = ?", [snapshot_id]
    ).fetchone()
    return row is not None


def record_snapshot(
    conn,
    *,
    snapshot_id: str,
    snapshot_at: datetime,
    source_filename: str,
    row_count: int,
) -> None:
    conn.execute(
        "INSERT INTO snapshot (snapshot_id, snapshot_at, source_filename, row_count, ingested_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [snapshot_id, snapshot_at, source_filename, row_count, _now_utc()],
    )
