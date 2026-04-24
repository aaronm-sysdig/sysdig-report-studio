"""Scenario: cisa_kev_known_ransomware propagates to finding_state at ingest.

Verifies:
1. NEW path: ransomware flag written to finding_state on first insert.
2. Non-ransomware finding on same image has flag = False.
3. RESEEN path: flag refreshed on second ingest (UPDATE branch).
4. Direct boolean filter on the column returns correct counts.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sas.ingest.ownership import NamespaceFallback, ResolverChain
from sas.ingest.pipeline import run_pipeline
from sas.ingest.schema import create_schema, migrate_schema

FIXTURES = Path(__file__).parent


@pytest.fixture()
def db():
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    migrate_schema(conn)
    yield conn
    conn.close()


def _resolver():
    return ResolverChain([NamespaceFallback()])


def test_kev_flag_propagates_on_new(db) -> None:
    """Ransomware flag must be True on finding_state after first ingest."""
    run_pipeline(conn=db, csv_path=FIXTURES / "day1_2026-05-01.csv",
                 resolver=_resolver())

    row = db.execute(
        "SELECT cisa_kev_known_ransomware FROM finding_state "
        "WHERE cve_id = 'CVE-2024-RANSOM-1'"
    ).fetchone()
    assert row is not None, "Finding for CVE-2024-RANSOM-1 not found"
    assert row[0] is True, f"Expected True, got {row[0]}"


def test_non_kev_flag_is_false(db) -> None:
    """Non-ransomware finding must have flag = False."""
    run_pipeline(conn=db, csv_path=FIXTURES / "day1_2026-05-01.csv",
                 resolver=_resolver())

    row = db.execute(
        "SELECT cisa_kev_known_ransomware FROM finding_state "
        "WHERE cve_id = 'CVE-2024-NORMAL-1'"
    ).fetchone()
    assert row is not None, "Finding for CVE-2024-NORMAL-1 not found"
    assert row[0] is False, f"Expected False, got {row[0]}"


def test_kev_flag_persists_on_reseen(db) -> None:
    """Ransomware flag must remain True after the RESEEN UPDATE on day 2."""
    run_pipeline(conn=db, csv_path=FIXTURES / "day1_2026-05-01.csv",
                 resolver=_resolver())
    run_pipeline(conn=db, csv_path=FIXTURES / "day2_2026-05-02.csv",
                 resolver=_resolver())

    rows = db.execute(
        "SELECT cisa_kev_known_ransomware FROM finding_state "
        "WHERE cve_id = 'CVE-2024-RANSOM-1'"
    ).fetchall()
    # Both the original OPEN row should exist — reseen, not re-inserted
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    assert rows[0][0] is True


def test_filter_on_kev_column_works(db) -> None:
    """Direct boolean filter on finding_state.cisa_kev_known_ransomware returns only ransomware rows."""
    run_pipeline(conn=db, csv_path=FIXTURES / "day1_2026-05-01.csv",
                 resolver=_resolver())

    kev_rows = db.execute(
        "SELECT COUNT(*) FROM finding_state WHERE cisa_kev_known_ransomware = TRUE"
    ).fetchone()[0]
    total_rows = db.execute("SELECT COUNT(*) FROM finding_state").fetchone()[0]

    assert kev_rows == 1, f"Expected 1 KEV row, got {kev_rows}"
    assert total_rows == 2, f"Expected 2 total rows, got {total_rows}"
