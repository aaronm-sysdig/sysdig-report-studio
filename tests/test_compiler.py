"""Compiler tests. Use an in-memory DuckDB with schema + seeded findings.

The seeded_db fixture creates 3 finding_state rows across a small date range
and seeds one rollup row so the rollup path can also be exercised.
"""
import duckdb
import pytest
from datetime import date
from sas.ingest.schema import create_schema
from sas.query.primitives import Query, TimeWindow, Filter
from sas.query.compiler import compile as sas_compile


@pytest.fixture
def seeded_db():
    """In-memory DuckDB with schema + 3 findings across 2 snapshot dates."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)

    # Seed required entity rows
    conn.execute(
        "INSERT INTO image VALUES ('sha256:aaa', 'linux', NOW(), NOW(), 'myrepo', 'v1')"
    )
    conn.execute(
        "INSERT INTO cve VALUES "
        "('CVE-2024-0001', NULL, NULL, 'v3', 'Critical', NULL, NULL, FALSE, NOW(), NOW()), "
        "('CVE-2024-0002', NULL, NULL, 'v3', 'High',     NULL, NULL, FALSE, NOW(), NOW()), "
        "('CVE-2024-0003', NULL, NULL, 'v3', 'Medium',   NULL, NULL, FALSE, NOW(), NOW())"
    )

    # 3 findings:
    #   1 — Critical, OPEN, first/last seen 2026-04-01/2026-04-10
    #   2 — High,     OPEN, first/last seen 2026-04-05/2026-04-10
    #   3 — Medium,   CLOSED (PATCHED 2026-04-08), first seen 2026-04-03
    conn.execute(
        """
        INSERT INTO finding_state VALUES
        (1, 'sha256:aaa', 'CVE-2024-0001', 'openssl', '1.0', '/usr/lib',
         'Critical', 9.8, TRUE,  TRUE,  '1.1',  FALSE, TRUE, FALSE,
         '2026-04-01'::TIMESTAMPTZ, '2026-04-10'::TIMESTAMPTZ,
         'OPEN', 'NEW', NULL, NULL, 0, 9, FALSE),
        (2, 'sha256:aaa', 'CVE-2024-0002', 'curl', '7.0', '/usr/bin',
         'High', 7.5, FALSE, FALSE, NULL, FALSE, FALSE, FALSE,
         '2026-04-05'::TIMESTAMPTZ, '2026-04-10'::TIMESTAMPTZ,
         'OPEN', 'NEW', NULL, NULL, 0, 5, FALSE),
        (3, 'sha256:aaa', 'CVE-2024-0003', 'zlib', '1.2', '/lib',
         'Medium', 5.0, FALSE, FALSE, NULL, FALSE, FALSE, FALSE,
         '2026-04-03'::TIMESTAMPTZ, '2026-04-08'::TIMESTAMPTZ,
         'CLOSED', 'PATCHED', '2026-04-08'::TIMESTAMPTZ, NULL, 0, 5, FALSE)
        """
    )

    # Seed one rollup row for 2026-04-10 so the rollup path returns data
    # Columns: date, image_id,
    #   count_open_critical, count_open_high, count_open_medium, count_open_low,
    #   count_open, count_new,
    #   count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
    #   count_regressed, mttr_sum, mttr_count
    conn.execute(
        """
        INSERT INTO daily_metrics_by_image VALUES
        ('2026-04-10'::DATE, 'sha256:aaa',
         1, 1, 0, 0,
         2, 0,
         1, 0, 0, 0,
         0, 5, 1)
        """
    )

    yield conn
    conn.close()


def _tw(n: int = 30) -> TimeWindow:
    return TimeWindow(mode="last_n_snapshots", n=n, granularity="day")


def _date_range_tw(start: str, end: str) -> TimeWindow:
    return TimeWindow(
        mode="date_range",
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        granularity="day",
    )


# ---------------------------------------------------------------------------
# One test per measure (basic smoke — just verify it runs and returns valid shape)
# ---------------------------------------------------------------------------

def test_count_open(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_open", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
    assert isinstance(result.series, list)
    assert isinstance(result.missing_days, list)


def test_count_new(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_new", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
    assert isinstance(result.series, list)


def test_count_fixed(seeded_db):
    # CVE lens forces the direct path (no rollup for CVE)
    q = Query(lens="CVE", traversal=[], time=_tw(), measure="count_fixed", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
    assert isinstance(result.series, list)


def test_count_regressed(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_regressed", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


def test_count_distinct_cve(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_distinct_cve", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


def test_mttr(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="mttr", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


# ---------------------------------------------------------------------------
# Rollup routing test: Image + count_open + day granularity → rollup table
# ---------------------------------------------------------------------------

def test_count_open_image_uses_rollup(seeded_db, monkeypatch):
    import sas.query.compiler as compiler_mod
    from sas.query.rollup_router import can_use_rollup as original
    calls = []

    def spy(q):
        result = original(q)
        calls.append(result)
        return result

    # Patch the name as used inside the compiler module
    monkeypatch.setattr(compiler_mod, "can_use_rollup", spy)
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_open", filters=[])
    sas_compile(q, seeded_db)
    assert len(calls) > 0
    assert calls[0] == "daily_metrics_by_image"


# ---------------------------------------------------------------------------
# group_by test: each Series key must include the grouped dimension
# ---------------------------------------------------------------------------

def test_group_by_severity(seeded_db):
    # Use direct path (count_distinct_cve not in rollup) to test group_by on finding_state
    q = Query(
        lens="Image",
        traversal=[],
        time=_date_range_tw("2026-04-01", "2026-04-10"),
        measure="count_distinct_cve",
        filters=[],
        group_by=["severity"],
    )
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
    for s in result.series:
        assert "severity" in s.key


# ---------------------------------------------------------------------------
# Filter test: filter by severity should only return Critical findings
# ---------------------------------------------------------------------------

def test_filter_by_severity(seeded_db):
    q = Query(
        lens="Image",
        traversal=[],
        time=_date_range_tw("2026-04-01", "2026-04-10"),
        measure="count_distinct_cve",
        filters=[Filter(field="severity", operator="eq", value="Critical")],
    )
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
    # With the filter only Critical findings pass; total distinct CVEs must be <= 1 per date
    for s in result.series:
        for count in s.y:
            # count_distinct_cve returns None for AVG on empty set; coerce safely
            if count is not None:
                assert count <= 1


# ---------------------------------------------------------------------------
# Empty-result test: unknown CVE should return empty series without raising
# ---------------------------------------------------------------------------

def test_empty_result_returns_empty_series(seeded_db):
    q = Query(
        lens="CVE",
        traversal=[],
        time=_date_range_tw("2026-04-01", "2026-04-10"),
        measure="count_open",
        filters=[Filter(field="cve_id", operator="eq", value="CVE-9999-9999")],
    )
    result = sas_compile(q, seeded_db)
    assert result.series == []
    assert result.exec_time_ms >= 0


# ---------------------------------------------------------------------------
# Missing-days test: seed data only on day 1 and day 3; day 2 must appear
# in missing_days when querying for all 3 days.
# ---------------------------------------------------------------------------

def test_missing_days_detected_from_snapshot_table():
    """missing_days reflects calendar dates with no snapshot ingested,
    not dates with no matching findings."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)

    # Simulate ingesting snapshots on day 1 and day 3 (skipping day 2)
    conn.execute(
        "INSERT INTO snapshot (snapshot_id, snapshot_at, source_filename, row_count, ingested_at) VALUES "
        "('snap1', '2026-06-01 12:00:00+00'::TIMESTAMPTZ, 'day1.csv', 100, NOW()), "
        "('snap3', '2026-06-03 12:00:00+00'::TIMESTAMPTZ, 'day3.csv', 100, NOW())"
    )

    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="date_range", start=date(2026, 6, 1), end=date(2026, 6, 3), granularity="day"),
        measure="count_open",
        filters=[],
    )
    result = sas_compile(q, conn)
    conn.close()

    assert date(2026, 6, 2) in result.missing_days
    assert date(2026, 6, 1) not in result.missing_days
    assert date(2026, 6, 3) not in result.missing_days


# ---------------------------------------------------------------------------
# Bug 1 — last_n_snapshots must use the snapshot table, not date.today()
# ---------------------------------------------------------------------------

def test_last_n_snapshots_uses_snapshot_table():
    """last_n_snapshots should query the snapshot table, not date.today()."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)

    # Insert snapshots on specific historical dates
    conn.execute(
        "INSERT INTO snapshot (snapshot_id, snapshot_at, source_filename, row_count, ingested_at) VALUES "
        "('s1', '2025-01-01 12:00:00+00'::TIMESTAMPTZ, 'd1.csv', 100, NOW()), "
        "('s2', '2025-01-02 12:00:00+00'::TIMESTAMPTZ, 'd2.csv', 100, NOW()), "
        "('s3', '2025-01-03 12:00:00+00'::TIMESTAMPTZ, 'd3.csv', 100, NOW())"
    )

    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=2, granularity="day"),
        measure="count_open",
        filters=[],
    )
    result = sas_compile(q, conn)
    conn.close()

    # Expected range: last 2 snapshots = 2025-01-02 to 2025-01-03
    assert result.snapshot_range[0] == date(2025, 1, 2)
    assert result.snapshot_range[1] == date(2025, 1, 3)


def test_last_n_snapshots_empty_db_returns_safe_range():
    """No snapshots ingested yet — shouldn't crash, should return empty range."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    q = Query(
        lens="Image", traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=5, granularity="day"),
        measure="count_open", filters=[],
    )
    result = sas_compile(q, conn)
    conn.close()
    assert result.series == []


# ---------------------------------------------------------------------------
# Bug 2 — direct path must join for Workload lens
# ---------------------------------------------------------------------------

def test_count_open_critical_returns_rollup_data(seeded_db):
    from sas.query.primitives import Query, TimeWindow
    from sas.query.compiler import compile as sas_compile
    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="date_range", start=date(2026, 4, 10), end=date(2026, 4, 10), granularity="day"),
        measure="count_open_critical",
        filters=[],
    )
    result = sas_compile(q, seeded_db)
    # The seeded rollup row has count_open_critical=1 for sha256:aaa on 2026-04-10
    assert len(result.series) == 1
    assert result.series[0].y[0] == 1


def test_direct_path_workload_lens_works():
    """Workload lens requires a join to workload_runs_image_daily.
    Before Bug 2 fix this would raise a DuckDB error about missing column."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    # Minimal seed: one snapshot + one finding + one workload-image run
    conn.execute(
        "INSERT INTO snapshot VALUES ('s1', '2026-05-01 12:00:00+00'::TIMESTAMPTZ, 'd.csv', 1, NOW())"
    )
    conn.execute(
        "INSERT INTO image VALUES ('sha256:x', 'linux', NOW(), NOW(), 'repo', 'v1')"
    )
    conn.execute(
        "INSERT INTO cve VALUES ('CVE-2026-1', NULL, NULL, 'v3', 'High', NULL, NULL, FALSE, NOW(), NOW())"
    )
    conn.execute(
        "INSERT INTO workload_runs_image_daily VALUES ('2026-05-01', 'c1', 'ns1', 'Deployment', 'wl1', 'main', 'sha256:x', 1)"
    )
    conn.execute(
        "INSERT INTO finding_state VALUES "
        "(100, 'sha256:x', 'CVE-2026-1', 'pkg', '1.0', '/p', 'High', 7.0, FALSE, FALSE, NULL, FALSE, FALSE, FALSE, "
        "'2026-05-01 12:00:00+00'::TIMESTAMPTZ, '2026-05-01 12:00:00+00'::TIMESTAMPTZ, "
        "'OPEN', 'NEW', NULL, NULL, 0, 0, FALSE)"
    )

    q = Query(
        lens="Workload", traversal=[],
        time=TimeWindow(mode="date_range", start=date(2026, 5, 1), end=date(2026, 5, 1), granularity="day"),
        measure="count_distinct_cve", filters=[],
    )
    result = sas_compile(q, conn)
    conn.close()

    # Should run without raising and return at least one series keyed by workload_name
    assert result.exec_time_ms >= 0
    assert len(result.series) >= 1
    assert "workload_name" in result.series[0].key


# ---------------------------------------------------------------------------
# current_tag filter test: rollup query with current_tag must join image table
# ---------------------------------------------------------------------------

def test_rollup_path_with_current_tag_filter_joins_image_table(seeded_db):
    """When current_tag filter is present, rollup query should join image table."""
    from sas.query.compiler import compile as sas_compile
    from sas.query.primitives import Query, TimeWindow, Filter

    conn = seeded_db
    # Seed snapshot row so last_n_snapshots resolves to 2026-04-10
    conn.execute(
        "INSERT INTO snapshot (snapshot_id, snapshot_at, source_filename, row_count, ingested_at) VALUES "
        "('snap1', '2026-04-10 12:00:00+00'::TIMESTAMPTZ, 'snapshot.csv', 3, NOW())"
    )
    # Seed a second image with different tag
    conn.execute(
        "INSERT INTO image VALUES ('sha256:bbb', 'linux', NOW(), NOW(), 'myrepo', 'v2')"
    )
    # Seed rollup row for second image
    conn.execute(
        """
        INSERT INTO daily_metrics_by_image (
            date, image_id, count_open_critical, count_open_high, count_open_medium, count_open_low,
            count_open, count_new, count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
            count_regressed, mttr_sum, mttr_count
        ) VALUES ('2026-04-10', 'sha256:bbb', 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        """
    )

    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open",
        filters=[Filter(field="current_tag", operator="eq", value="v1")],
    )

    result = sas_compile(q, conn)
    # Should only return data for sha256:aaa (tag v1), not sha256:bbb (tag v2)
    total = sum(sum(s.y) for s in result.series)
    assert total == 2  # Only the two OPEN findings from sha256:aaa
