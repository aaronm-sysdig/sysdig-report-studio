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
         'Critical', 9.8, TRUE,  TRUE,  '1.1',  FALSE, TRUE,
         '2026-04-01'::TIMESTAMPTZ, '2026-04-10'::TIMESTAMPTZ,
         'OPEN', 'NEW', NULL, NULL, 0, 9, FALSE),
        (2, 'sha256:aaa', 'CVE-2024-0002', 'curl', '7.0', '/usr/bin',
         'High', 7.5, FALSE, FALSE, NULL, FALSE, FALSE,
         '2026-04-05'::TIMESTAMPTZ, '2026-04-10'::TIMESTAMPTZ,
         'OPEN', 'NEW', NULL, NULL, 0, 5, FALSE),
        (3, 'sha256:aaa', 'CVE-2024-0003', 'zlib', '1.2', '/lib',
         'Medium', 5.0, FALSE, FALSE, NULL, FALSE, FALSE,
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

def test_missing_days_detected():
    conn = duckdb.connect(":memory:")
    create_schema(conn)

    conn.execute(
        "INSERT INTO image VALUES ('sha256:bbb', 'linux', NOW(), NOW(), 'repo2', 'latest')"
    )
    conn.execute(
        "INSERT INTO cve VALUES ('CVE-2025-0001', NULL, NULL, 'v3', 'High', NULL, NULL, FALSE, NOW(), NOW())"
    )

    # Insert findings with last_seen on day 1 and day 3 only (skip day 2)
    conn.execute(
        """
        INSERT INTO finding_state VALUES
        (10, 'sha256:bbb', 'CVE-2025-0001', 'libssl', '1.0', '/lib',
         'High', 7.0, FALSE, FALSE, NULL, FALSE, FALSE,
         '2026-06-01'::TIMESTAMPTZ, '2026-06-01'::TIMESTAMPTZ,
         'OPEN', 'NEW', NULL, NULL, 0, 1, FALSE),
        (11, 'sha256:bbb', 'CVE-2025-0001', 'libssl', '1.0', '/lib',
         'High', 7.0, FALSE, FALSE, NULL, FALSE, FALSE,
         '2026-06-01'::TIMESTAMPTZ, '2026-06-03'::TIMESTAMPTZ,
         'OPEN', 'NEW', NULL, NULL, 0, 3, FALSE)
        """
    )

    # Direct path (count_new is in rollup but Image table needs rollup data;
    # use count_distinct_cve to force direct path with no rollup dependency)
    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(
            mode="date_range",
            start=date(2026, 6, 1),
            end=date(2026, 6, 3),
            granularity="day",
        ),
        measure="count_distinct_cve",
        filters=[],
    )
    result = sas_compile(q, conn)
    conn.close()

    # The date spine is [2026-06-01, 2026-06-02, 2026-06-03].
    # last_seen (date anchor for count_distinct_cve) has data on 2026-06-01 and 2026-06-03
    # but nothing on 2026-06-02 → 2026-06-02 should be missing.
    assert date(2026, 6, 2) in result.missing_days
