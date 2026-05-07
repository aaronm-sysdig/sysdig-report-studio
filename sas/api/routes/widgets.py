"""GET /api/widgets/catalog — hardcoded catalog of 10 starter widget definitions."""

from fastapi import APIRouter

router = APIRouter()

_CATALOG = [
    {
        "id": "image-remediation-story",
        "title": "Image Remediation Story",
        "widget_type": "line",
        "query": {
            "lens": "Image",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"},
            "measure": "count_open",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "fleet-critical-trend",
        "title": "Fleet Critical Trend",
        "widget_type": "line",
        "query": {
            "lens": "Image",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"},
            "measure": "count_open",
            "filters": [{"field": "severity", "operator": "eq", "value": "Critical"}],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "new-findings-this-week",
        "title": "New Findings This Week",
        "widget_type": "bar",
        "query": {
            "lens": "Image",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 7, "granularity": "day"},
            "measure": "count_new",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "fixed-findings-this-week",
        "title": "Fixed Findings This Week",
        "widget_type": "bar",
        "query": {
            "lens": "Image",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 7, "granularity": "day"},
            "measure": "count_fixed",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "regressed-findings",
        "title": "Regressed Findings",
        "widget_type": "bar",
        "query": {
            "lens": "Image",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 30, "granularity": "day"},
            "measure": "count_regressed",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "mttr-by-team",
        "title": "MTTR by Team",
        "widget_type": "bar",
        "query": {
            "lens": "Team",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"},
            "measure": "mttr",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "unique-cves-open",
        "title": "Unique CVEs Open",
        "widget_type": "line",
        "query": {
            "lens": "Image",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"},
            "measure": "count_distinct_cve",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "team-leaderboard-open",
        "title": "Team Leaderboard — Open",
        "widget_type": "bar",
        "query": {
            "lens": "Team",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 30, "granularity": "day"},
            "measure": "count_open",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "repository-risk-trend",
        "title": "Repository Risk Trend",
        "widget_type": "line",
        "query": {
            "lens": "Repository",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 60, "granularity": "day"},
            "measure": "count_open",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
    {
        "id": "cluster-open-findings",
        "title": "Cluster Open Findings",
        "widget_type": "bar",
        "query": {
            "lens": "Cluster",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 30, "granularity": "day"},
            "measure": "count_open",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        },
    },
]


@router.get("/widgets/catalog", tags=["widgets"])
def get_catalog() -> list[dict]:
    """Return the 10 hardcoded starter widget definitions."""
    return _CATALOG
