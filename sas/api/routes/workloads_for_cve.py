"""GET /api/workloads-for-cve — workloads running images affected by a CVE."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from sas.api.deps import get_db

router = APIRouter()


class WorkloadRow(BaseModel):
    cluster_name: str
    namespace_name: str
    workload_type: str
    workload_name: str
    container_name: str
    team_id: str | None


class WorkloadsForCveResponse(BaseModel):
    cve_id: str
    workloads: list[WorkloadRow]
    total: int


@router.get("/workloads-for-cve", response_model=WorkloadsForCveResponse, tags=["workloads"])
def get_workloads_for_cve(
    cve_id: str = Query(..., min_length=1, description="CVE ID to look up"),
    conn=Depends(get_db),
) -> WorkloadsForCveResponse:
    """Return distinct workloads running images that contain the given CVE (OPEN state only)."""
    # Get latest snapshot date
    latest = conn.execute("SELECT MAX(date) FROM workload_runs_image_daily").fetchone()[0]
    if latest is None:
        return WorkloadsForCveResponse(cve_id=cve_id, workloads=[], total=0)

    rows = conn.execute("""
        SELECT DISTINCT
            wr.cluster_name,
            wr.namespace_name,
            wr.workload_type,
            wr.workload_name,
            wr.container_name,
            wo.team_id
        FROM finding_state fs
        INNER JOIN workload_runs_image_daily wr
            ON wr.image_id = fs.image_id
            AND wr.date = ?
        LEFT JOIN workload_owned_by wo
            ON wo.cluster_name = wr.cluster_name
            AND wo.namespace_name = wr.namespace_name
            AND wo.workload_type = wr.workload_type
            AND wo.workload_name = wr.workload_name
        WHERE fs.cve_id = ? AND fs.state = 'OPEN'
        ORDER BY wr.cluster_name, wr.namespace_name, wr.workload_name
    """, [latest, cve_id]).fetchall()

    return WorkloadsForCveResponse(
        cve_id=cve_id,
        workloads=[
            WorkloadRow(
                cluster_name=r[0],
                namespace_name=r[1],
                workload_type=r[2],
                workload_name=r[3],
                container_name=r[4],
                team_id=r[5],
            )
            for r in rows
        ],
        total=len(rows),
    )
