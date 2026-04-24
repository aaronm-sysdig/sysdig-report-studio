import pytest
from sas.query.primitives import Query, TimeWindow
from sas.query.rollup_router import can_use_rollup

_DAY_TW  = TimeWindow(mode="last_n_snapshots", n=7, granularity="day")
_WEEK_TW = TimeWindow(mode="last_n_snapshots", n=4, granularity="week")


def _q(lens, measure, granularity="day"):
    tw = TimeWindow(mode="last_n_snapshots", n=7, granularity=granularity)
    return Query(lens=lens, traversal=[], time=tw, measure=measure, filters=[])


# --- cases that SHOULD use rollup ---
def test_image_count_open_uses_rollup():
    result = can_use_rollup(_q("Image", "count_open"))
    assert result == "daily_metrics_by_image"

def test_workload_count_new_uses_rollup():
    result = can_use_rollup(_q("Workload", "count_new"))
    assert result == "daily_metrics_by_workload"

def test_team_count_fixed_uses_rollup():
    result = can_use_rollup(_q("Team", "count_fixed"))
    assert result == "daily_metrics_by_team"

def test_repository_count_regressed_uses_rollup():
    result = can_use_rollup(_q("Repository", "count_regressed"))
    assert result == "daily_metrics_by_repository"

def test_cluster_count_open_uses_rollup():
    result = can_use_rollup(_q("Cluster", "count_open"))
    assert result == "daily_metrics_by_cluster_severity"

# --- cases that should NOT use rollup ---
def test_cve_lens_no_rollup():
    assert can_use_rollup(_q("CVE", "count_open")) is False

def test_namespace_lens_no_rollup():
    assert can_use_rollup(_q("Namespace", "count_open")) is False

def test_package_lens_no_rollup():
    assert can_use_rollup(_q("Package", "count_open")) is False

def test_mttr_no_rollup():
    assert can_use_rollup(_q("Image", "mttr")) is False

def test_count_distinct_cve_no_rollup():
    assert can_use_rollup(_q("Image", "count_distinct_cve")) is False

def test_week_granularity_no_rollup():
    assert can_use_rollup(_q("Image", "count_open", granularity="week")) is False

def test_month_granularity_no_rollup():
    assert can_use_rollup(_q("Image", "count_open", granularity="month")) is False


# --- filter-column gate tests ---

def test_query_with_finding_level_filter_falls_through_to_direct_path():
    """A filter on 'severity' (not on rollup tables) must force the direct path."""
    from sas.query.primitives import Filter
    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open",
        filters=[Filter(field="severity", operator="eq", value="Critical")],
    )
    assert can_use_rollup(q) is False


def test_query_with_image_id_filter_still_uses_rollup():
    """A filter on image_id (which IS on daily_metrics_by_image) stays on rollup."""
    from sas.query.primitives import Filter
    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open",
        filters=[Filter(field="image_id", operator="eq", value="sha256:abc")],
    )
    assert can_use_rollup(q) == "daily_metrics_by_image"


def test_cluster_severity_rollup_allows_severity_filter():
    """The daily_metrics_by_cluster_severity table DOES have severity as a filterable column."""
    from sas.query.primitives import Filter
    q = Query(
        lens="Cluster",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open",
        filters=[Filter(field="severity", operator="eq", value="Critical")],
    )
    assert can_use_rollup(q) == "daily_metrics_by_cluster_severity"


def test_count_open_critical_routes_to_image_rollup():
    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open_critical",
        filters=[],
    )
    assert can_use_rollup(q) == "daily_metrics_by_image"


def test_count_open_high_routes_to_workload_rollup():
    q = Query(
        lens="Workload",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open_high",
        filters=[],
    )
    assert can_use_rollup(q) == "daily_metrics_by_workload"
