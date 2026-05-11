"""Entity picker endpoints for UI dropdowns.

Routes:
- GET /api/entities/tags?repository=X  -- distinct tags for a repository
- GET /api/entities/{lens}             -- entity values for a given lens
"""

from fastapi import APIRouter, Depends, HTTPException

from sas.api.deps import get_db
from sas.query.registry import LENSES

router = APIRouter()

_ENTITY_QUERIES: dict[str, str] = {
    "Image": (
        "SELECT image_id AS id, "
        "COALESCE(current_repository || ':' || current_tag, image_id) AS label, "
        "current_repository AS repository, current_tag AS tag, "
        "strftime(last_seen, '%Y-%m-%dT%H:%M:%S+00:00') AS last_seen "
        "FROM image ORDER BY label"
    ),
    "CVE": (
        "SELECT cve_id AS id, cve_id AS label, initial_severity AS severity "
        "FROM cve ORDER BY cve_id"
    ),
    "Workload": (
        "SELECT workload_name AS id, "
        "cluster_name || '/' || namespace_name || '/' || workload_name AS label, "
        "cluster_name, namespace_name "
        "FROM workload ORDER BY label"
    ),
    "Cluster": (
        "SELECT cluster_name AS id, cluster_name AS label "
        "FROM cluster ORDER BY label"
    ),
    "Namespace": (
        "SELECT cluster_name || '/' || namespace_name AS id, "
        "cluster_name || '/' || namespace_name AS label, "
        "cluster_name, namespace_name "
        "FROM namespace ORDER BY label"
    ),
    "Package": (
        "SELECT package_name AS id, package_name AS label, package_type "
        "FROM package ORDER BY label"
    ),
    "Repository": (
        "SELECT repository AS id, repository AS label "
        "FROM repository ORDER BY label"
    ),
    "Team": (
        "SELECT team_id AS id, display_name AS label "
        "FROM team ORDER BY label"
    ),
    "Owner": (
        "SELECT owner_id AS id, display_name AS label "
        "FROM owner ORDER BY label"
    ),
}


@router.get("/entities/tags", tags=["entities"])
def get_tags(repository: str, conn=Depends(get_db)) -> list[dict]:
    """Return distinct tags for a repository, with image counts."""
    rows = conn.execute(
        """
        SELECT current_tag AS tag, COUNT(DISTINCT image_id) AS image_count
        FROM image
        WHERE current_repository = ?
        GROUP BY current_tag
        ORDER BY current_tag
        """,
        [repository],
    ).fetchall()
    return [{"tag": r[0], "image_count": r[1]} for r in rows]


@router.get("/entities/{lens}", tags=["entities"])
def get_entities(lens: str, conn=Depends(get_db)) -> list[dict]:
    """Return entity values for a given lens, for use in UI pickers."""
    if lens not in LENSES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown lens: {lens!r}. Valid lenses: {list(LENSES.keys())}",
        )
    sql = _ENTITY_QUERIES[lens]
    rows = conn.execute(sql).fetchall()
    col_names = [d[0] for d in conn.description]
    return [dict(zip(col_names, row)) for row in rows]
