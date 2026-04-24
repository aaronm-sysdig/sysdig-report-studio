"""Decides whether a Query can be served from a pre-aggregated rollup table.

Returns the rollup table name (str) if yes, False otherwise.
The API/compiler never needs to know the routing logic — it just calls this.
"""

from sas.query.primitives import Query

_ROLLUP_MEASURES = {"count_open", "count_new", "count_fixed", "count_regressed"}

_LENS_TO_ROLLUP = {
    "Image":      "daily_metrics_by_image",
    "Workload":   "daily_metrics_by_workload",
    "Team":       "daily_metrics_by_team",
    "Repository": "daily_metrics_by_repository",
    "Cluster":    "daily_metrics_by_cluster_severity",
}

# Columns on each rollup table that are safe to filter on.
# Measure/aggregate columns (count_open etc.) are intentionally excluded.
# Finding-level columns absent from a table must not appear here — their
# presence means the query needs the direct path (joining finding_state).
_ROLLUP_FILTER_COLUMNS: dict[str, set[str]] = {
    "daily_metrics_by_image": {"image_id"},
    "daily_metrics_by_workload": {"cluster_name", "namespace_name", "workload_type", "workload_name"},
    "daily_metrics_by_team": {"team_id"},
    "daily_metrics_by_repository": {"repository"},
    "daily_metrics_by_cluster_severity": {"cluster_name", "severity"},
}


def can_use_rollup(query: Query) -> bool | str:
    """Return rollup table name if eligible, False otherwise.

    Eligible when ALL of:
    - granularity == 'day'
    - measure is in _ROLLUP_MEASURES
    - lens has a corresponding rollup table
    - every filter field exists as a column on that rollup table
    """
    if query.time.granularity != "day":
        return False
    if query.measure not in _ROLLUP_MEASURES:
        return False
    table = _LENS_TO_ROLLUP.get(query.lens)
    if table is None:
        return False
    allowed = _ROLLUP_FILTER_COLUMNS.get(table, set())
    for f in query.filters:
        if f.field not in allowed:
            return False
    return table
