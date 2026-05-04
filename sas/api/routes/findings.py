"""GET /api/findings — paginated raw finding_state rows for the Findings Table widget."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sas.api.deps import get_db

router = APIRouter()


class FindingRow(BaseModel):
    finding_id: int
    cve_id: str
    severity: str
    image_id: str
    image_name: str | None
    package_name: str
    first_seen: str
    last_seen: str
    state: str
    reason_code: str | None
    in_use: bool
    fix_available: bool
    public_exploit: bool


class FindingsResponse(BaseModel):
    rows: list[FindingRow]
    total: int


ALLOWED_SEVERITIES = {"Critical", "High", "Medium", "Low", "Negligible"}
ALLOWED_STATES = {"OPEN", "CLOSED"}


@router.get("/findings", response_model=FindingsResponse)
def list_findings(
    severity: str | None = None,
    state: str | None = None,
    fix_available: bool | None = None,
    in_use: bool | None = None,
    public_exploit: bool | None = None,
    limit: int = 100,
    offset: int = 0,
    conn=Depends(get_db),
) -> FindingsResponse:
    """Paginated finding_state rows. Filters: severity, state, fix_available, in_use, public_exploit. Ordered by last_seen desc."""
    if limit > 500:
        limit = 500  # safety cap
    if severity and severity not in ALLOWED_SEVERITIES:
        raise HTTPException(status_code=422, detail=f"severity must be one of {sorted(ALLOWED_SEVERITIES)}")
    if state and state not in ALLOWED_STATES:
        raise HTTPException(status_code=422, detail=f"state must be one of {sorted(ALLOWED_STATES)}")

    where = []
    params: list[Any] = []
    if severity:
        where.append("fs.severity = ?")
        params.append(severity)
    if state:
        where.append("fs.state = ?")
        params.append(state)
    if fix_available is not None:
        where.append("fs.fix_available = ?")
        params.append(1 if fix_available else 0)
    if in_use is not None:
        where.append("fs.in_use = ?")
        params.append(1 if in_use else 0)
    if public_exploit is not None:
        where.append("fs.public_exploit = ?")
        params.append(1 if public_exploit else 0)

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    # Total count for pagination
    count_sql = f"SELECT COUNT(*) FROM finding_state fs {where_sql}"
    total = conn.execute(count_sql, params).fetchone()[0]

    # Paginated rows
    sql = f"""
        SELECT fs.finding_id, fs.cve_id, fs.severity, fs.image_id,
               COALESCE(i.current_repository || ':' || i.current_tag, fs.image_id) AS image_name,
               fs.package_name,
               strftime(fs.first_seen, '%Y-%m-%dT%H:%M:%S+00:00') AS first_seen,
               strftime(fs.last_seen, '%Y-%m-%dT%H:%M:%S+00:00') AS last_seen,
               fs.state, fs.reason_code,
               fs.in_use, fs.fix_available, fs.public_exploit
        FROM finding_state fs
        LEFT JOIN image i ON i.image_id = fs.image_id
        {where_sql}
        ORDER BY fs.last_seen DESC, fs.severity
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(sql, params + [limit, offset]).fetchall()

    return FindingsResponse(
        rows=[
            FindingRow(
                finding_id=r[0], cve_id=r[1], severity=r[2], image_id=r[3],
                image_name=r[4], package_name=r[5], first_seen=r[6],
                last_seen=r[7], state=r[8], reason_code=r[9],
                in_use=bool(r[10]), fix_available=bool(r[11]), public_exploit=bool(r[12]),
            ) for r in rows
        ],
        total=total,
    )
