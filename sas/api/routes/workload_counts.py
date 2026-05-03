"""GET /api/workload-counts — CVE-level workload counts from latest snapshot."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from sas.api.deps import get_db

router = APIRouter()


class WorkloadCount(BaseModel):
    cve_id: str
    workload_count: int


class WorkloadCountsResponse(BaseModel):
    counts: list[WorkloadCount]
    snapshot_date: str


@router.get("/workload-counts", response_model=WorkloadCountsResponse)
def get_workload_counts(
    conn=Depends(get_db),
) -> WorkloadCountsResponse:
    """Return workload counts per CVE aggregated from the latest snapshot date."""
    # Get latest snapshot date
    latest = conn.execute("SELECT MAX(date) FROM workload_runs_image_daily").fetchone()[0]

    # Join finding_state to workload_runs_image_daily to get workload counts per CVE
    rows = conn.execute("""
        SELECT
            fs.cve_id,
            COUNT(DISTINCT wr.cluster_name || '|' || wr.namespace_name || '|' ||
                             wr.workload_type || '|' || wr.workload_name || '|' ||
                             wr.container_name) AS workload_count
        FROM finding_state fs
        INNER JOIN workload_runs_image_daily wr
            ON wr.image_id = fs.image_id
            AND wr.date = ?
        WHERE fs.state = 'OPEN'
        GROUP BY fs.cve_id
        ORDER BY workload_count DESC
    """, [latest]).fetchall()

    return WorkloadCountsResponse(
        counts=[WorkloadCount(cve_id=r[0], workload_count=r[1]) for r in rows],
        snapshot_date=str(latest),
    )
