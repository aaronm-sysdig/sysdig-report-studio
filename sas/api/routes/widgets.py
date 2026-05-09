"""GET /api/widgets/catalog — hardcoded catalog of starter widget definitions."""

from fastapi import APIRouter

router = APIRouter()

_CATALOG: list[dict] = [
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
    # -- deprecated / hidden (kept for reference) --
    {
        "id": "mttr-by-team",
        "title": "MTTR by Team",
        "widget_type": "bar",
        "hidden": True,
        "hide_reason": "Team lens removed — namespace-derived teams don't represent real ownership boundaries",
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
        "id": "team-leaderboard-open",
        "title": "Team Leaderboard — Open",
        "widget_type": "bar",
        "hidden": True,
        "hide_reason": "Team lens removed — namespace-derived teams don't represent real ownership boundaries",
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
]


@router.get("/widgets/catalog", tags=["widgets"])
def get_catalog(hidden: bool = False) -> list[dict]:
    """Return starter widget definitions. Pass ?hidden=true to include deprecated widgets."""
    if hidden:
        return _CATALOG
    return [w for w in _CATALOG if not w.get("hidden")]
