from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.snapshot import (
    compute_snapshot_id,
    extract_snapshot_at,
    is_already_ingested,
    record_snapshot,
)


def test_compute_snapshot_id_deterministic(fixtures_dir):
    a = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=1)
    b = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=1)
    assert a == b
    assert len(a) >= 16  # looks like a hash


def test_compute_snapshot_id_differs_on_row_count(fixtures_dir):
    a = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=1)
    b = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=2)
    assert a != b


def test_extract_snapshot_at_from_filename():
    # phoenix-vuln-findings-2026_04_23.csv → 2026-04-23 12:00:00 UTC
    ts = extract_snapshot_at(Path("phoenix-vuln-findings-2026_04_23.csv"))
    assert ts.year == 2026 and ts.month == 4 and ts.day == 23
    assert ts.hour == 12 and ts.minute == 0


def test_extract_snapshot_at_falls_back_to_now_on_unparseable(monkeypatch):
    fixed = datetime(2026, 5, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "sas.ingest.snapshot._now_utc", lambda: fixed
    )
    ts = extract_snapshot_at(Path("no-date-in-name.csv"))
    assert ts == fixed


def test_compute_snapshot_id_differs_on_content(tmp_path):
    """Two files with same name and same row count but different content → different IDs."""
    a = tmp_path / "same_name.csv"
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    b = b_dir / "same_name.csv"
    a.write_text("header\nvalue_a\n")
    b.write_text("header\nvalue_b\n")
    ida = compute_snapshot_id(a, row_count=1)
    idb = compute_snapshot_id(b, row_count=1)
    assert ida != idb


def test_compute_snapshot_id_stable_across_identical_content(tmp_path):
    """Same name, same row count, byte-identical content → same ID (even from different paths)."""
    a = tmp_path / "f.csv"
    b_dir = tmp_path / "sub"
    b_dir.mkdir()
    b = b_dir / "f.csv"
    content = "header\nrow1\n"
    a.write_text(content)
    b.write_text(content)
    assert compute_snapshot_id(a, row_count=1) == compute_snapshot_id(b, row_count=1)


def test_idempotency_flow(db):
    create_schema(db)
    sid = "test-snap-001"
    assert is_already_ingested(db, sid) is False
    record_snapshot(
        db,
        snapshot_id=sid,
        snapshot_at=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
        source_filename="foo.csv",
        row_count=42,
    )
    assert is_already_ingested(db, sid) is True
