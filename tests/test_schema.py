import duckdb
import pytest
from sas.ingest.schema import create_schema, EXPECTED_TABLES


def test_create_schema_creates_all_tables(db):
    create_schema(db)
    rows = db.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    actual = [r[0] for r in rows]
    assert actual == sorted(EXPECTED_TABLES)


def test_create_schema_is_idempotent(db):
    create_schema(db)
    create_schema(db)  # should not raise
    rows = db.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchone()
    assert rows[0] == len(EXPECTED_TABLES)


def test_finding_state_has_expected_columns(db):
    create_schema(db)
    rows = db.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'finding_state' ORDER BY column_name"
    ).fetchall()
    cols = {r[0] for r in rows}
    required = {
        "finding_id", "image_id", "cve_id", "package_name",
        "package_version", "package_path", "severity", "cvss_score",
        "in_use", "fix_available", "fix_version", "risk_accepted",
        "public_exploit", "first_seen", "last_seen", "state",
        "reason_code", "closed_at", "reopened_at", "reopen_count",
        "days_open", "is_regression",
    }
    assert required.issubset(cols), f"missing: {required - cols}"
