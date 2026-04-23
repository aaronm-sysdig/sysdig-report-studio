"""End-to-end integration test using the real Phoenix sample CSV.

Asserts invariants, not exact values — because a real sample will drift over time.
"""
from pathlib import Path
import pytest
import duckdb

from sas.ingest.schema import create_schema
from sas.ingest.pipeline import run_pipeline
from sas.ingest.ownership import ResolverChain, NamespaceFallback


REPO_ROOT = Path(__file__).parent.parent
SAMPLE_CSV = REPO_ROOT / "phoenix-vuln-findings-2026_04_23.csv"


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample CSV not present")
def test_real_sample_ingests_cleanly(tmp_path):
    db_path = tmp_path / "sas.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        create_schema(conn)
        resolver = ResolverChain([NamespaceFallback()])
        result = run_pipeline(conn=conn, csv_path=SAMPLE_CSV, resolver=resolver)
    finally:
        conn.close()

    assert not result["already_ingested"]
    assert result["rows"] > 0
    assert result["new"] == result["rows"] or result["new"] > 0

    # Reopen and assert invariants
    conn = duckdb.connect(str(db_path))
    try:
        # Some data landed in each key table
        for table in ("image", "cve", "package", "cluster", "namespace",
                      "workload", "repository", "finding_state",
                      "workload_runs_image_daily", "daily_metrics_by_image"):
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            assert count > 0, f"{table} was empty after ingest"

        # Every finding_state row has a valid state
        bad = conn.execute(
            "SELECT count(*) FROM finding_state WHERE state NOT IN ('OPEN','CLOSED')"
        ).fetchone()[0]
        assert bad == 0

        # On first ingest, every finding should be OPEN (no prior state to close)
        closed = conn.execute(
            "SELECT count(*) FROM finding_state WHERE state = 'CLOSED'"
        ).fetchone()[0]
        assert closed == 0

        # replica_count is never negative
        neg = conn.execute(
            "SELECT count(*) FROM workload_runs_image_daily WHERE replica_count < 1"
        ).fetchone()[0]
        assert neg == 0
    finally:
        conn.close()


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample CSV not present")
def test_real_sample_reingestion_is_idempotent(tmp_path):
    db_path = tmp_path / "sas.duckdb"
    resolver = ResolverChain([NamespaceFallback()])

    conn = duckdb.connect(str(db_path))
    try:
        create_schema(conn)
        first = run_pipeline(conn=conn, csv_path=SAMPLE_CSV, resolver=resolver)
        first_findings = conn.execute("SELECT count(*) FROM finding_state").fetchone()[0]
    finally:
        conn.close()

    conn = duckdb.connect(str(db_path))
    try:
        second = run_pipeline(conn=conn, csv_path=SAMPLE_CSV, resolver=resolver)
        second_findings = conn.execute("SELECT count(*) FROM finding_state").fetchone()[0]
    finally:
        conn.close()

    assert second["already_ingested"] is True
    assert first_findings == second_findings
