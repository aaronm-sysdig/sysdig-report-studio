"""POST /api/query — accepts a serialised Query, returns a QueryResult."""

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from sas.api.deps import get_db
from sas.query.primitives import Filter, Ordering, Query, TimeWindow
from sas.query.compiler import compile as sas_compile

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models for the API boundary (Option A from the plan)
# ---------------------------------------------------------------------------

class TimeWindowIn(BaseModel):
    mode: str
    n: int | None = None
    start: date | None = None
    end: date | None = None
    granularity: str = "day"


class FilterIn(BaseModel):
    field: str
    operator: str
    value: Any


class OrderingIn(BaseModel):
    field: str
    direction: str = "asc"


class QueryIn(BaseModel):
    lens: str
    traversal: list[str] = []
    time: TimeWindowIn
    measure: str
    filters: list[FilterIn] = []
    group_by: list[str] = []
    order_by: OrderingIn | None = None
    limit: int | None = None


# ---------------------------------------------------------------------------
# Pydantic models for the API response
# ---------------------------------------------------------------------------

class SeriesOut(BaseModel):
    key: dict[str, Any]
    x: list[str]
    y: list[float | int | None]


class QueryResultOut(BaseModel):
    series: list[SeriesOut]
    dimensions: dict[str, list[Any]]
    snapshot_range: list[str]  # [start_iso, end_iso]
    missing_days: list[str]
    exec_time_ms: int


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/query", tags=["query"], response_model=QueryResultOut)
def run_query(body: QueryIn, conn=Depends(get_db)) -> QueryResultOut:
    """Execute a Query against DuckDB and return a QueryResult."""
    # Convert Pydantic → dataclasses
    tw = TimeWindow(
        mode=body.time.mode,
        n=body.time.n,
        start=body.time.start,
        end=body.time.end,
        granularity=body.time.granularity,
    )
    filters = [Filter(field=f.field, operator=f.operator, value=f.value) for f in body.filters]
    order_by = (
        Ordering(field=body.order_by.field, direction=body.order_by.direction)
        if body.order_by
        else None
    )
    query = Query(
        lens=body.lens,
        traversal=body.traversal,
        time=tw,
        measure=body.measure,
        filters=filters,
        group_by=body.group_by,
        order_by=order_by,
        limit=body.limit,
    )

    result = sas_compile(query, conn)

    return QueryResultOut(
        series=[
            SeriesOut(key=s.key, x=[str(d) for d in s.x], y=s.y)
            for s in result.series
        ],
        dimensions=result.dimensions,
        snapshot_range=[str(result.snapshot_range[0]), str(result.snapshot_range[1])],
        missing_days=[str(d) for d in result.missing_days],
        exec_time_ms=result.exec_time_ms,
    )
