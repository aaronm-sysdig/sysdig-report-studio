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
