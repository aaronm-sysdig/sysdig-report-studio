"""GET /api/workload-counts — CVE-level workload counts from latest snapshot."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from sas.api.deps import get_db

router = APIRouter()


class WorkloadCount(BaseModel):
    """Deprecated — use WeightedCve."""
    cve_id: str
    workload_count: int


class WeightedCve(BaseModel):
    cve_id: str
    severity: str
    workload_count: int
    in_use: bool
    fix_available: bool
    public_exploit: bool


class WorkloadCountsResponse(BaseModel):
    counts: list[WeightedCve]
    snapshot_date: str


@router.get("/workload-counts", response_model=WorkloadCountsResponse)
def get_workload_counts(
    conn=Depends(get_db),
) -> WorkloadCountsResponse:
    """Return workload counts per CVE aggregated from the latest snapshot date."""
    # Get latest snapshot date
    latest = conn.execute("SELECT MAX(date) FROM workload_runs_image_daily").fetchone()[0]
    if latest is None:
        return WorkloadCountsResponse(counts=[], snapshot_date="")

    # Join finding_state to workload_runs_image_daily to get workload counts per CVE
    # Also aggregate severity (max) and actionability flags (BOOL_OR)
    rows = conn.execute("""
        SELECT
            fs.cve_id,
            (SELECT fs2.severity FROM finding_state fs2
             WHERE fs2.cve_id = fs.cve_id AND fs2.state = 'OPEN'
             ORDER BY
               CASE fs2.severity
                 WHEN 'Critical' THEN 5 WHEN 'High' THEN 4
                 WHEN 'Medium' THEN 3 WHEN 'Low' THEN 2 WHEN 'Negligible' THEN 1
               END DESC
             LIMIT 1) AS severity,
            COUNT(DISTINCT wr.cluster_name || '|' || wr.namespace_name || '|' ||
                             wr.workload_type || '|' || wr.workload_name || '|' ||
                             wr.container_name) AS workload_count,
            BOOL_OR(fs.in_use) AS in_use,
            BOOL_OR(fs.fix_available) AS fix_available,
            BOOL_OR(fs.public_exploit) AS public_exploit
        FROM finding_state fs
        INNER JOIN workload_runs_image_daily wr
            ON wr.image_id = fs.image_id
            AND wr.date = ?
        WHERE fs.state = 'OPEN'
        GROUP BY fs.cve_id
        ORDER BY workload_count DESC
    """, [latest]).fetchall()

    return WorkloadCountsResponse(
        counts=[WeightedCve(
            cve_id=r[0],
            severity=r[1],
            workload_count=r[2],
            in_use=bool(r[3]),
            fix_available=bool(r[4]),
            public_exploit=bool(r[5]),
        ) for r in rows],
        snapshot_date=str(latest),
    )
