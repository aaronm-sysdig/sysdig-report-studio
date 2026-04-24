"""Compiles a Query dataclass into DuckDB SQL and returns a QueryResult.

Two paths:
1. Rollup path — fast, pre-aggregated daily_metrics_* tables.
2. Direct path — query finding_state, compute measure predicate per date.

The caller (FastAPI route) owns the connection lifecycle.

**Known limitation — count_open over historical dates:**
The direct path for `count_open` uses `last_seen::DATE = target_date` as the date anchor. Because
ingestion updates `last_seen` for every OPEN finding on every snapshot, this returns accurate counts
only for the *most recent* ingested snapshot date. For historical dates, use the rollup path
(daily_metrics_by_*) which precomputes `count_open` per-day correctly. A future enhancement will
unify these paths.
"""

from __future__ import annotations
import time
from datetime import date, timedelta

from sas.query.primitives import Query, QueryResult, Series
from sas.query.registry import LENSES
from sas.query.rollup_router import can_use_rollup


_OPERATOR_MAP = {"eq": "=", "neq": "!=", "gte": ">=", "lte": "<=", "in": "IN"}

# Map measure name → rollup column expression.
# count_fixed is stored split across 4 reason-code columns; sum them.
_ROLLUP_MEASURE_EXPR = {
    "count_open":      "count_open",
    "count_new":       "count_new",
    "count_fixed":     "(count_fixed_patched + count_fixed_retired + count_fixed_accepted + count_fixed_other)",
    "count_regressed": "count_regressed",
}

# Primary key / entity identifier column per rollup table.
_ROLLUP_LENS_PK = {
    "daily_metrics_by_image":            "image_id",
    "daily_metrics_by_workload":         "workload_name",
    "daily_metrics_by_team":             "team_id",
    "daily_metrics_by_repository":       "repository",
    "daily_metrics_by_cluster_severity": "cluster_name",
}

# For the direct path: (date-anchor column, base WHERE predicate)
_DIRECT_DATE_COL = {
    "count_open":         ("finding_state.last_seen",   "finding_state.state = 'OPEN'"),
    "count_new":          ("finding_state.first_seen",  "finding_state.state = 'OPEN'"),
    "count_fixed":        ("finding_state.closed_at",   "finding_state.state = 'CLOSED'"),
    "count_regressed":    ("finding_state.reopened_at", "finding_state.reopened_at IS NOT NULL"),
    "count_distinct_cve": ("finding_state.last_seen",   "finding_state.state = 'OPEN'"),
    "mttr":               ("finding_state.closed_at",   "finding_state.state = 'CLOSED'"),
}

_DIRECT_AGGREGATE = {
    "count_open":         "COUNT(*)",
    "count_new":          "COUNT(*)",
    "count_fixed":        "COUNT(*)",
    "count_regressed":    "COUNT(*)",
    "count_distinct_cve": "COUNT(DISTINCT finding_state.cve_id)",
    "mttr":               "AVG(finding_state.days_open)",
}

# Direct-path join SQL per lens. Empty string = no join needed.
_LENS_JOIN_SQL = {
    "Image":      "",  # image_id is on finding_state
    "CVE":        "",  # cve_id is on finding_state
    "Package":    "",  # package_name is on finding_state
    "Repository": (
        "JOIN image_in_repository iir ON iir.image_id = finding_state.image_id"
    ),
    "Workload": (
        "JOIN workload_runs_image_daily wri ON wri.image_id = finding_state.image_id "
        "  AND wri.date = CAST(finding_state.last_seen AS DATE)"
    ),
    "Cluster": (
        "JOIN workload_runs_image_daily wri ON wri.image_id = finding_state.image_id "
        "  AND wri.date = CAST(finding_state.last_seen AS DATE)"
    ),
    "Namespace": (
        "JOIN workload_runs_image_daily wri ON wri.image_id = finding_state.image_id "
        "  AND wri.date = CAST(finding_state.last_seen AS DATE)"
    ),
    "Team": (
        "JOIN workload_runs_image_daily wri ON wri.image_id = finding_state.image_id "
        "  AND wri.date = CAST(finding_state.last_seen AS DATE) "
        "JOIN workload_owned_by wo ON wo.cluster_name = wri.cluster_name "
        "  AND wo.namespace_name = wri.namespace_name "
        "  AND wo.workload_type = wri.workload_type "
        "  AND wo.workload_name = wri.workload_name"
    ),
    "Owner": (
        "JOIN workload_runs_image_daily wri ON wri.image_id = finding_state.image_id "
        "  AND wri.date = CAST(finding_state.last_seen AS DATE) "
        "JOIN workload_owned_by wo ON wo.cluster_name = wri.cluster_name "
        "  AND wo.namespace_name = wri.namespace_name "
        "  AND wo.workload_type = wri.workload_type "
        "  AND wo.workload_name = wri.workload_name"
    ),
}

# Per-lens PK column (qualified with table alias) → unqualified alias for SELECT/GROUP BY
_DIRECT_LENS_PK_COL = {
    "Image":      "finding_state.image_id",
    "CVE":        "finding_state.cve_id",
    "Package":    "finding_state.package_name",
    "Repository": "iir.repository",
    "Workload":   "wri.workload_name",
    "Cluster":    "wri.cluster_name",
    "Namespace":  "wri.namespace_name",
    "Team":       "wo.team_id",
    "Owner":      "wo.owner_id",
}


def _resolve_window(query: Query, conn) -> tuple[date, date]:
    tw = query.time
    if tw.mode == "date_range":
        return tw.start, tw.end
    if tw.mode == "last_n_snapshots":
        rows = conn.execute(
            "SELECT DISTINCT CAST(snapshot_at AS DATE) as d "
            "FROM snapshot ORDER BY d DESC LIMIT ?",
            [tw.n]
        ).fetchall()
        if not rows:
            # No snapshots ingested yet — return an empty range (same date, no data)
            return date.today(), date.today()
        dates = sorted(r[0] for r in rows)
        return dates[0], dates[-1]
    # all_time — broad historical range
    return date(2020, 1, 1), date.today()


def _date_spine(start: date, end: date) -> list[date]:
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _build_filter_clause(filters) -> tuple[str, list]:
    """Build a SQL WHERE fragment and parameter list for the given filters."""
    if not filters:
        return "", []
    clauses, params = [], []
    for f in filters:
        op = _OPERATOR_MAP.get(f.operator, "=")
        if f.operator == "in":
            placeholders = ", ".join(["?"] * len(f.value))
            clauses.append(f"{f.field} IN ({placeholders})")
            params.extend(f.value)
        else:
            clauses.append(f"{f.field} {op} ?")
            params.append(f.value)
    return " AND " + " AND ".join(clauses), params


def _compute_missing_days(conn, start: date, end: date) -> list[date]:
    """Calendar dates in [start, end] where no snapshot was ingested."""
    rows = conn.execute(
        "SELECT DISTINCT CAST(snapshot_at AS DATE) as d "
        "FROM snapshot "
        "WHERE CAST(snapshot_at AS DATE) BETWEEN ? AND ?",
        [start, end]
    ).fetchall()
    ingested = {r[0] for r in rows}
    spine = set(_date_spine(start, end))
    return sorted(spine - ingested)


def _rows_to_series(
    rows: list,
    col_names: list[str],
    pk_col: str,
    group_by: list[str],
) -> list[Series]:
    """Convert raw DB rows into Series objects."""
    groups: dict[tuple, tuple[list, list]] = {}

    for row in rows:
        row_dict = dict(zip(col_names, row))
        # Build the series key: entity pk + any group_by dimensions
        key_vals = {pk_col: row_dict[pk_col]}
        for g in group_by:
            key_vals[g] = row_dict.get(g)
        group_key = tuple(sorted(key_vals.items()))
        if group_key not in groups:
            groups[group_key] = ([], [])
        row_date = row_dict["date"]
        # DuckDB may return date as a date object already
        if not isinstance(row_date, date):
            row_date = date.fromisoformat(str(row_date))
        groups[group_key][0].append(row_date)
        groups[group_key][1].append(row_dict["value"])

    return [Series(key=dict(k), x=v[0], y=v[1]) for k, v in groups.items()]


def _compile_rollup(
    query: Query, conn, table: str, start: date, end: date
) -> QueryResult:
    t0 = time.monotonic()
    pk_col = _ROLLUP_LENS_PK[table]
    measure_expr = _ROLLUP_MEASURE_EXPR[query.measure]
    filter_sql, filter_params = _build_filter_clause(query.filters)

    group_cols_sql = ""
    if query.group_by:
        group_cols_sql = ", " + ", ".join(query.group_by)

    sql = (
        f"SELECT date, {pk_col}{group_cols_sql}, {measure_expr} AS value "
        f"FROM {table} "
        f"WHERE date BETWEEN ? AND ?{filter_sql} "
        f"ORDER BY date"
    )

    rows = conn.execute(sql, [start, end] + filter_params).fetchall()
    col_names = [d[0] for d in conn.description]

    series = _rows_to_series(rows, col_names, pk_col, query.group_by)
    missing = _compute_missing_days(conn, start, end)
    exec_ms = int((time.monotonic() - t0) * 1000)
    return QueryResult(
        series=series,
        dimensions={},
        snapshot_range=(start, end),
        missing_days=missing,
        exec_time_ms=exec_ms,
    )


def _compile_direct(
    query: Query, conn, start: date, end: date
) -> QueryResult:
    t0 = time.monotonic()

    # Resolve per-lens join and PK column
    join_sql = _LENS_JOIN_SQL.get(query.lens, "")
    qualified_pk = _DIRECT_LENS_PK_COL.get(query.lens, "finding_state.image_id")
    # Unqualified alias for SELECT column name and GROUP BY
    pk_col = qualified_pk.split(".")[-1]

    date_col, base_predicate = _DIRECT_DATE_COL[query.measure]
    aggregate = _DIRECT_AGGREGATE[query.measure]
    filter_sql, filter_params = _build_filter_clause(query.filters)

    group_cols_sql = ""
    if query.group_by:
        group_cols_sql = ", " + ", ".join(query.group_by)

    join_clause = f"\n{join_sql}" if join_sql else ""

    date_expr = f"CAST({date_col} AS DATE)"
    sql = (
        f"SELECT {date_expr} AS date, {qualified_pk} AS {pk_col}{group_cols_sql}, {aggregate} AS value "
        f"FROM finding_state{join_clause} "
        f"WHERE {base_predicate} "
        f"  AND {date_expr} BETWEEN ? AND ?{filter_sql} "
        f"GROUP BY {date_expr}, {qualified_pk}{group_cols_sql} "
        f"ORDER BY date"
    )

    rows = conn.execute(sql, [start, end] + filter_params).fetchall()
    col_names = [d[0] for d in conn.description]

    series = _rows_to_series(rows, col_names, pk_col, query.group_by)
    missing = _compute_missing_days(conn, start, end)
    exec_ms = int((time.monotonic() - t0) * 1000)
    return QueryResult(
        series=series,
        dimensions={},
        snapshot_range=(start, end),
        missing_days=missing,
        exec_time_ms=exec_ms,
    )


def compile(query: Query, conn) -> QueryResult:
    """Compile a Query to SQL and execute against conn. Returns QueryResult.

    Never raises on empty results — returns a QueryResult with an empty series list.
    """
    start, end = _resolve_window(query, conn)
    rollup_table = can_use_rollup(query)
    if rollup_table:
        return _compile_rollup(query, conn, rollup_table, start, end)
    return _compile_direct(query, conn, start, end)
