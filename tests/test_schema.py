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


def test_rollup_tables_share_metric_columns(db):
    create_schema(db)
    shared_cols = {
        "count_open_critical", "count_open_high", "count_open_medium",
        "count_open_low", "count_open", "count_new",
        "count_fixed_patched", "count_fixed_retired", "count_fixed_accepted",
        "count_fixed_other", "count_regressed",
    }
    for table in ("daily_metrics_by_image", "daily_metrics_by_workload",
                  "daily_metrics_by_team", "daily_metrics_by_repository"):
        rows = db.execute(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{table}'"
        ).fetchall()
        cols = {r[0] for r in rows}
        missing = shared_cols - cols
        assert not missing, f"{table} missing {missing}"


def test_mttr_present_where_meaningful_absent_where_not(db):
    create_schema(db)
    for table in ("daily_metrics_by_image", "daily_metrics_by_workload",
                  "daily_metrics_by_team"):
        rows = db.execute(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{table}' AND column_name LIKE 'mttr_%'"
        ).fetchall()
        assert len(rows) == 2, f"{table} should have mttr_sum and mttr_count"

    rows = db.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'daily_metrics_by_repository' AND column_name LIKE 'mttr_%'"
    ).fetchall()
    assert len(rows) == 0, "repository rollup intentionally has no mttr cols"
