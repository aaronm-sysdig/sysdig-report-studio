"""Core dataclasses for the Query primitive. No DB imports."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class TimeWindow:
    mode: str                       # "last_n_snapshots" | "date_range" | "all_time"
    n: int | None = None            # for last_n_snapshots
    start: date | None = None       # for date_range
    end: date | None = None         # for date_range
    granularity: str = "day"        # "day" | "week" | "month"


@dataclass
class Filter:
    field: str
    operator: str                   # "eq" | "neq" | "in" | "gte" | "lte"
    value: Any


@dataclass
class Ordering:
    field: str
    direction: str = "asc"          # "asc" | "desc"


@dataclass
class Query:
    lens: str                       # registry key e.g. "Image"
    traversal: list[str]            # list of Edge registry keys
    time: TimeWindow
    measure: str                    # registry key e.g. "count_open"
    filters: list[Filter]
    group_by: list[str] = field(default_factory=list)
    order_by: Ordering | None = None
    limit: int | None = None


@dataclass
class Series:
    key: dict[str, Any]             # e.g. {"severity": "Critical"}
    x: list[date]
    y: list[float | int]


@dataclass
class QueryResult:
    series: list[Series]
    dimensions: dict[str, list]
    snapshot_range: tuple[date, date]
    missing_days: list[date]
    exec_time_ms: int
