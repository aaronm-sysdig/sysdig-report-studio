"""Compiles a Query dataclass into DuckDB SQL and returns a QueryResult.

Two paths:
1. Rollup path — fast, pre-aggregated daily_metrics_* tables.
2. Direct path — query finding_state, compute measure predicate per date.

The caller (FastAPI route) owns the connection lifecycle.
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
    "count_open":         ("last_seen",   "state = 'OPEN'"),
    "count_new":          ("first_seen",  "state = 'OPEN'"),
    "count_fixed":        ("closed_at",   "state = 'CLOSED'"),
    "count_regressed":    ("reopened_at", "reopened_at IS NOT NULL"),
    "count_distinct_cve": ("last_seen",   "state = 'OPEN'"),
    "mttr":               ("closed_at",   "state = 'CLOSED'"),
}

_DIRECT_AGGREGATE = {
    "count_open":         "COUNT(*)",
    "count_new":          "COUNT(*)",
    "count_fixed":        "COUNT(*)",
    "count_regressed":    "COUNT(*)",
    "count_distinct_cve": "COUNT(DISTINCT cve_id)",
    "mttr":               "AVG(days_open)",
}


def _resolve_window(query: Query) -> tuple[date, date]:
    tw = query.time
    if tw.mode == "date_range":
        return tw.start, tw.end
    if tw.mode == "last_n_snapshots":
        end = date.today()
        start = end - timedelta(days=tw.n - 1)
        return start, end
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


def _rows_to_series(
    rows: list,
    col_names: list[str],
    pk_col: str,
    group_by: list[str],
    all_dates: set[date],
) -> tuple[list[Series], list[date]]:
    """Convert raw DB rows into Series objects and compute missing days."""
    seen_dates: set[date] = set()
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
        seen_dates.add(row_date)

    series = [Series(key=dict(k), x=v[0], y=v[1]) for k, v in groups.items()]
    missing = sorted(all_dates - seen_dates)
    return series, missing


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
    all_dates = set(_date_spine(start, end))

    series, missing = _rows_to_series(rows, col_names, pk_col, query.group_by, all_dates)
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
    lens_meta = LENSES[query.lens]
    pk = lens_meta["pk"]
    # For compound PKs use the first element as the grouping key
    pk_col = pk if isinstance(pk, str) else pk[0]

    date_col, base_predicate = _DIRECT_DATE_COL[query.measure]
    aggregate = _DIRECT_AGGREGATE[query.measure]
    filter_sql, filter_params = _build_filter_clause(query.filters)

    group_cols_sql = ""
    if query.group_by:
        group_cols_sql = ", " + ", ".join(query.group_by)

    sql = (
        f"SELECT CAST({date_col} AS DATE) AS date, {pk_col}{group_cols_sql}, {aggregate} AS value "
        f"FROM finding_state "
        f"WHERE {base_predicate} "
        f"  AND CAST({date_col} AS DATE) BETWEEN ? AND ?{filter_sql} "
        f"GROUP BY date, {pk_col}{group_cols_sql} "
        f"ORDER BY date"
    )

    rows = conn.execute(sql, [start, end] + filter_params).fetchall()
    col_names = [d[0] for d in conn.description]
    all_dates = set(_date_spine(start, end))

    series, missing = _rows_to_series(rows, col_names, pk_col, query.group_by, all_dates)
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
    start, end = _resolve_window(query)
    rollup_table = can_use_rollup(query)
    if rollup_table:
        return _compile_rollup(query, conn, rollup_table, start, end)
    return _compile_direct(query, conn, start, end)
