# SAS Phase 2 — Query Engine + API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Query primitive, SQL compiler, rollup router, and FastAPI HTTP layer that Phase 3 (Next.js frontend) will consume. End state: `POST /api/query` accepts a serialised `Query`, compiles it to DuckDB SQL, and returns a typed `QueryResult`. All 6 measures work. The 3 API endpoints are testable with FastAPI TestClient.

**Architecture:** `sas/query/` is pure Python (no FastAPI import). `sas/api/` depends on `sas/query/`. DuckDB connection is injected via FastAPI dependency — the API opens read-only; the ingest pipeline owns the write connection.

**Design references:**
- Spec §7 (Query primitive): [`docs/superpowers/specs/2026-04-23-sas-design.md`](../specs/2026-04-23-sas-design.md) lines 196–239.
- Spec §3 (architecture): same file, lines 32–52.
- Phase 1 plan (structural template): [`2026-04-23-sas-phase1-data-foundation.md`](./2026-04-23-sas-phase1-data-foundation.md).

**Collaboration note:** Tasks 1–4 (primitives, registry, measures, rollup router) can be dispatched to Sonnet workers in parallel. Task 5 (compiler) and Task 7 (routes) benefit from Opus review.

---

## File Structure

```
sas/
├── query/
│   ├── __init__.py          # exports Query, QueryResult, Series, Lens, Measure, Edge, TimeWindow, Filter
│   ├── primitives.py        # dataclasses: Query, QueryResult, Series, TimeWindow, Filter, Ordering
│   ├── registry.py          # Lens, Measure, Edge as dicts (not enums)
│   ├── measures.py          # 6 concrete Measure implementations
│   ├── compiler.py          # compile(query, conn) -> QueryResult
│   └── rollup_router.py     # can_use_rollup(query) -> bool | str
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, /healthz
│   ├── deps.py              # DuckDB read-only connection dependency
│   ├── run.py               # uvicorn one-liner
│   └── routes/
│       ├── __init__.py
│       ├── query.py         # POST /api/query
│       ├── widgets.py       # GET /api/widgets/catalog
│       └── entities.py      # GET /api/entities/{lens}
tests/
├── test_query_primitives.py
├── test_compiler.py
├── test_rollup_router.py
└── test_api.py
```

---

## Task 1: Package scaffolding

**Files:**
- Create: `sas/query/__init__.py`
- Create: `sas/api/__init__.py`
- Create: `sas/api/routes/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add FastAPI and uvicorn to requirements.txt**

Append these two lines (don't replace existing contents):

```
fastapi>=0.110.0
uvicorn>=0.27.0
httpx>=0.27.0
```

(`httpx` is required by FastAPI's TestClient.)

- [ ] **Step 2: Install into the existing venv**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: `Successfully installed fastapi-... uvicorn-... httpx-...`

- [ ] **Step 3: Create empty package files**

Files (all empty — they only mark the package boundary):

```
sas/query/__init__.py
sas/api/__init__.py
sas/api/routes/__init__.py
```

- [ ] **Step 4: Verify FastAPI is importable**

Run: `.venv/bin/python -c "import fastapi; print(fastapi.__version__)"`
Expected: prints a version string, no errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt sas/query/__init__.py sas/api/__init__.py sas/api/routes/__init__.py
git commit -m "feat(sas): phase 2 package scaffolding + fastapi dependency"
```

---

## Task 2: Query primitives (dataclasses)

**Files:**
- Create: `sas/query/primitives.py`
- Create: `tests/test_query_primitives.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_query_primitives.py`

```python
from datetime import date
from sas.query.primitives import (
    Query, QueryResult, Series, TimeWindow, Filter, Ordering
)


def test_timewindow_last_n():
    tw = TimeWindow(mode="last_n_snapshots", n=30, granularity="day")
    assert tw.n == 30
    assert tw.start is None


def test_timewindow_date_range():
    tw = TimeWindow(mode="date_range", start=date(2026, 1, 1), end=date(2026, 4, 1), granularity="week")
    assert tw.start < tw.end


def test_filter_equality():
    f = Filter(field="severity", operator="eq", value="Critical")
    assert f.field == "severity"
    assert f.value == "Critical"


def test_query_minimal():
    tw = TimeWindow(mode="last_n_snapshots", n=7, granularity="day")
    q = Query(lens="Image", traversal=[], time=tw, measure="count_open", filters=[])
    assert q.lens == "Image"
    assert q.limit is None


def test_query_result_empty():
    qr = QueryResult(series=[], dimensions={}, snapshot_range=(date(2026, 1, 1), date(2026, 4, 1)), missing_days=[], exec_time_ms=0)
    assert qr.series == []


def test_series_shape():
    s = Series(key={"severity": "Critical"}, x=[date(2026, 4, 1)], y=[42])
    assert len(s.x) == len(s.y)


def test_ordering():
    o = Ordering(field="date", direction="asc")
    assert o.direction in ("asc", "desc")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_query_primitives.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sas.query.primitives'`

- [ ] **Step 3: Write primitives.py**

File: `sas/query/primitives.py`

```python
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
```

- [ ] **Step 4: Update sas/query/__init__.py to export public API**

File: `sas/query/__init__.py`

```python
from .primitives import Query, QueryResult, Series, TimeWindow, Filter, Ordering
from .registry import LENSES, MEASURES, EDGES

__all__ = [
    "Query", "QueryResult", "Series", "TimeWindow", "Filter", "Ordering",
    "LENSES", "MEASURES", "EDGES",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_query_primitives.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add sas/query/primitives.py sas/query/__init__.py tests/test_query_primitives.py
git commit -m "feat(sas): query primitives dataclasses"
```

---

## Task 3: Registry + 6 measures

**Files:**
- Create: `sas/query/registry.py`
- Create: `sas/query/measures.py`
- Create: `tests/test_measures.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_measures.py`

```python
import pytest
from sas.query.registry import LENSES, MEASURES, EDGES
from sas.query.measures import (
    CountOpen, CountNew, CountFixed, CountRegressed, CountDistinctCve, Mttr
)


def test_all_v1_lenses_registered():
    expected = {"Image", "CVE", "Workload", "Cluster", "Namespace", "Package", "Repository", "Team", "Owner"}
    assert expected == set(LENSES.keys())


def test_all_v1_measures_registered():
    expected = {"count_open", "count_new", "count_fixed", "count_regressed", "count_distinct_cve", "mttr"}
    assert expected == set(MEASURES.keys())


def test_count_open_sql():
    m = CountOpen()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "state" in sql and "OPEN" in sql


def test_count_new_sql():
    m = CountNew()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "first_seen" in sql


def test_count_fixed_sql():
    m = CountFixed()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "closed_at" in sql


def test_count_regressed_sql():
    m = CountRegressed()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "reopened_at" in sql


def test_count_distinct_cve_sql():
    m = CountDistinctCve()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "DISTINCT" in sql and "cve_id" in sql


def test_mttr_sql():
    m = Mttr()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "AVG" in sql and "days_open" in sql


def test_all_measures_have_required_columns():
    for name, cls in [
        ("count_open", CountOpen), ("count_new", CountNew), ("count_fixed", CountFixed),
        ("count_regressed", CountRegressed), ("count_distinct_cve", CountDistinctCve), ("mttr", Mttr),
    ]:
        m = cls()
        assert isinstance(m.required_columns, list)
        assert len(m.required_columns) > 0
```

- [ ] **Step 2: Write registry.py**

File: `sas/query/registry.py`

```python
"""Lens, Measure, and Edge registries. Dicts, not enums — additive by design."""

# Keys are the canonical string names used in Query.lens, Query.measure, Query.traversal.

LENSES: dict[str, dict] = {
    "Image":      {"primary_table": "image",      "pk": "image_id"},
    "CVE":        {"primary_table": "cve",         "pk": "cve_id"},
    "Workload":   {"primary_table": "workload",    "pk": ("cluster_name", "namespace_name", "workload_type", "workload_name")},
    "Cluster":    {"primary_table": "cluster",     "pk": "cluster_name"},
    "Namespace":  {"primary_table": "namespace",   "pk": ("cluster_name", "namespace_name")},
    "Package":    {"primary_table": "package",     "pk": ("package_name", "package_type")},
    "Repository": {"primary_table": "repository",  "pk": "repository"},
    "Team":       {"primary_table": "team",        "pk": "team_id"},
    "Owner":      {"primary_table": "owner",       "pk": "owner_id"},
}

MEASURES: dict[str, type] = {}  # populated by measures.py via register_measure()

EDGES: dict[str, dict] = {
    "image_in_repository":   {"from": "Image",    "to": "Repository", "table": "image_in_repository",   "join_on": "image_id"},
    "workload_runs_image":   {"from": "Workload",  "to": "Image",      "table": "workload_runs_image_daily", "join_on": "image_id"},
    "workload_in_namespace": {"from": "Workload",  "to": "Namespace",  "table": "workload_in_namespace", "join_on": ("cluster_name", "namespace_name")},
    "namespace_in_cluster":  {"from": "Namespace", "to": "Cluster",    "table": "namespace_in_cluster",  "join_on": ("cluster_name", "namespace_name")},
    "workload_owned_by":     {"from": "Workload",  "to": "Team",       "table": "workload_owned_by",     "join_on": "team_id"},
}


def register_measure(name: str, cls: type) -> None:
    MEASURES[name] = cls
```

- [ ] **Step 3: Write measures.py**

File: `sas/query/measures.py`

```python
"""Six v1 Measure implementations. Each produces a SQL fragment and lists required columns."""

from sas.query.registry import register_measure


class CountOpen:
    name = "count_open"
    required_columns = ["state", "last_seen"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE state = 'OPEN' AND CAST(last_seen AS DATE) <= '{target_date}')"
        )


class CountNew:
    name = "count_new"
    required_columns = ["state", "first_seen"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE state = 'OPEN' AND CAST(first_seen AS DATE) = '{target_date}')"
        )


class CountFixed:
    name = "count_fixed"
    required_columns = ["state", "closed_at"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE state = 'CLOSED' AND CAST(closed_at AS DATE) = '{target_date}')"
        )


class CountRegressed:
    name = "count_regressed"
    required_columns = ["reopened_at"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE reopened_at IS NOT NULL AND CAST(reopened_at AS DATE) = '{target_date}')"
        )


class CountDistinctCve:
    name = "count_distinct_cve"
    required_columns = ["cve_id", "state"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(DISTINCT cve_id) FILTER (WHERE state = 'OPEN' AND CAST(last_seen AS DATE) <= '{target_date}')"
        )


class Mttr:
    name = "mttr"
    required_columns = ["days_open", "state"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"AVG(days_open) FILTER (WHERE state = 'CLOSED' AND CAST(closed_at AS DATE) = '{target_date}')"
        )


# Registration — must run at import time
for _cls in [CountOpen, CountNew, CountFixed, CountRegressed, CountDistinctCve, Mttr]:
    register_measure(_cls.name, _cls)
```

- [ ] **Step 4: Update sas/query/__init__.py to import measures (triggers registration)**

Add this import at the top of `sas/query/__init__.py`:

```python
from . import measures as _measures  # noqa: F401 — side-effect: registers all measures
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_measures.py -v`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add sas/query/registry.py sas/query/measures.py sas/query/__init__.py tests/test_measures.py
git commit -m "feat(sas): query registry + 6 measure implementations"
```

---

## Task 4: Rollup router

**Files:**
- Create: `sas/query/rollup_router.py`
- Create: `tests/test_rollup_router.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_rollup_router.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_rollup_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sas.query.rollup_router'`

- [ ] **Step 3: Write rollup_router.py**

File: `sas/query/rollup_router.py`

```python
"""Decides whether a Query can be served from a pre-aggregated rollup table.

Returns the rollup table name (str) if yes, False otherwise.
The API/compiler never needs to know the routing logic — it just calls this.
"""

from sas.query.primitives import Query

_ROLLUP_MEASURES = {"count_open", "count_new", "count_fixed", "count_regressed"}

_LENS_TO_ROLLUP = {
    "Image":      "daily_metrics_by_image",
    "Workload":   "daily_metrics_by_workload",
    "Team":       "daily_metrics_by_team",
    "Repository": "daily_metrics_by_repository",
    "Cluster":    "daily_metrics_by_cluster_severity",
}


def can_use_rollup(query: Query) -> bool | str:
    """Return rollup table name if eligible, False otherwise.

    Eligible when ALL of:
    - granularity == 'day'
    - measure is in _ROLLUP_MEASURES
    - lens has a corresponding rollup table
    """
    if query.time.granularity != "day":
        return False
    if query.measure not in _ROLLUP_MEASURES:
        return False
    table = _LENS_TO_ROLLUP.get(query.lens)
    if table is None:
        return False
    return table
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_rollup_router.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add sas/query/rollup_router.py tests/test_rollup_router.py
git commit -m "feat(sas): rollup router — routes day-granularity queries to pre-agg tables"
```

---

## Task 5: SQL compiler (the core)

**Files:**
- Create: `sas/query/compiler.py`
- Create: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_compiler.py`

```python
"""Compiler tests. Use the in-memory `db` fixture from conftest.py.
Schema is created and a tiny fixture CSV is ingested before each test.
"""
import duckdb
import pytest
from datetime import date
from sas.ingest.schema import create_schema
from sas.query.primitives import Query, TimeWindow, Filter
from sas.query.compiler import compile as sas_compile


@pytest.fixture
def seeded_db():
    """In-memory DuckDB with schema + 3 findings across 2 snapshot dates."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    # Seed image
    conn.execute("INSERT INTO image VALUES ('sha256:aaa', 'linux', NOW(), NOW(), 'myrepo', 'v1')")
    conn.execute("INSERT INTO cve VALUES ('CVE-2024-0001', NULL, NULL, 'v3', 'Critical', NULL, NULL, FALSE, NOW(), NOW())")
    # Seed finding_state rows
    conn.execute("""
        INSERT INTO finding_state VALUES
        (1, 'sha256:aaa', 'CVE-2024-0001', 'openssl', '1.0', '/usr/lib', 'Critical', 9.8,
         TRUE, TRUE, '1.1', FALSE, TRUE,
         '2026-04-01'::TIMESTAMP, '2026-04-10'::TIMESTAMP, 'OPEN', 'NEW', NULL, NULL, 0, 9, FALSE),
        (2, 'sha256:aaa', 'CVE-2024-0002', 'curl', '7.0', '/usr/bin', 'High', 7.5,
         FALSE, FALSE, NULL, FALSE, FALSE,
         '2026-04-05'::TIMESTAMP, '2026-04-10'::TIMESTAMP, 'OPEN', 'NEW', NULL, NULL, 0, 5, FALSE),
        (3, 'sha256:aaa', 'CVE-2024-0003', 'zlib', '1.2', '/lib', 'Medium', 5.0,
         FALSE, FALSE, NULL, FALSE, FALSE,
         '2026-04-03'::TIMESTAMP, '2026-04-08'::TIMESTAMP, 'CLOSED', 'PATCHED', '2026-04-08'::TIMESTAMP, NULL, 0, 5, FALSE)
    """)
    # Seed rollup table for image
    conn.execute("""
        INSERT INTO daily_metrics_by_image VALUES
        ('2026-04-10'::DATE, 'sha256:aaa', 2, 0, 0, 0, 2, NULL)
    """)
    yield conn
    conn.close()


def _tw(n=30):
    return TimeWindow(mode="last_n_snapshots", n=n, granularity="day")


# One test per measure
def test_count_open(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_open", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
    assert len(result.series) > 0
    total = sum(p for s in result.series for p in s.y)
    assert total >= 0


def test_count_new(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_new", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


def test_count_fixed(seeded_db):
    q = Query(lens="CVE", traversal=[], time=_tw(), measure="count_fixed", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


def test_count_regressed(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_regressed", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


def test_count_distinct_cve(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_distinct_cve", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


def test_mttr(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="mttr", filters=[])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0


# Rollup routing test: Image + count_open + day granularity → daily_metrics_by_image
def test_count_open_image_uses_rollup(seeded_db, monkeypatch):
    from sas.query import rollup_router
    calls = []
    original = rollup_router.can_use_rollup
    def spy(q):
        result = original(q)
        calls.append(result)
        return result
    monkeypatch.setattr(rollup_router, "can_use_rollup", spy)
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_open", filters=[])
    sas_compile(q, seeded_db)
    assert calls[0] == "daily_metrics_by_image"


# group_by test
def test_group_by_severity(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_open",
              filters=[], group_by=["severity"])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
    # each series key must include "severity"
    for s in result.series:
        assert "severity" in s.key


# filter test
def test_filter_by_severity(seeded_db):
    q = Query(lens="Image", traversal=[], time=_tw(), measure="count_open",
              filters=[Filter(field="severity", operator="eq", value="Critical")])
    result = sas_compile(q, seeded_db)
    assert result.exec_time_ms >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_compiler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sas.query.compiler'`

- [ ] **Step 3: Write compiler.py**

The compiler has two paths: rollup and direct. SQL templates below are the exact patterns to use.

**Rollup path** (e.g., `count_open` on `Image`, day granularity):
```sql
SELECT date, image_id, count_open AS value
FROM daily_metrics_by_image
WHERE date BETWEEN ? AND ?
ORDER BY date
```

**Direct path** — each measure maps to a WHERE clause against `finding_state`. The date spine is generated in Python (all dates in range), then LEFT JOINed to make missing days explicit.

For `count_open` on `CVE` (no rollup):
```sql
SELECT CAST(last_seen AS DATE) AS date, cve_id, COUNT(*) AS value
FROM finding_state
WHERE state = 'OPEN' AND CAST(last_seen AS DATE) BETWEEN ? AND ?
GROUP BY date, cve_id
ORDER BY date
```

File: `sas/query/compiler.py`

```python
"""Compiles a Query dataclass into DuckDB SQL and returns a QueryResult.

Two paths:
1. Rollup path — fast, pre-aggregated daily_metrics_* tables.
2. Direct path — query finding_state, compute measure predicate per date.

The caller (FastAPI route) owns the connection lifecycle.
"""

from __future__ import annotations
import time
from datetime import date, timedelta
from typing import Any

from sas.query.primitives import Query, QueryResult, Series
from sas.query.registry import LENSES, MEASURES
from sas.query.rollup_router import can_use_rollup


_OPERATOR_MAP = {"eq": "=", "neq": "!=", "gte": ">=", "lte": "<=", "in": "IN"}

_ROLLUP_MEASURE_COL = {
    "count_open":      "count_open",
    "count_new":       "count_new",
    "count_fixed":     "count_fixed",
    "count_regressed": "count_regressed",
}

_ROLLUP_LENS_PK = {
    "daily_metrics_by_image":            "image_id",
    "daily_metrics_by_workload":         "workload_name",
    "daily_metrics_by_team":             "team_id",
    "daily_metrics_by_repository":       "repository",
    "daily_metrics_by_cluster_severity": "cluster_name",
}

_DIRECT_DATE_COL = {
    "count_open":        ("last_seen",   "state = 'OPEN'"),
    "count_new":         ("first_seen",  "state = 'OPEN'"),
    "count_fixed":       ("closed_at",   "state = 'CLOSED'"),
    "count_regressed":   ("reopened_at", "reopened_at IS NOT NULL"),
    "count_distinct_cve": ("last_seen",  "state = 'OPEN'"),
    "mttr":              ("closed_at",   "state = 'CLOSED'"),
}

_DIRECT_AGGREGATE = {
    "count_open":        "COUNT(*)",
    "count_new":         "COUNT(*)",
    "count_fixed":       "COUNT(*)",
    "count_regressed":   "COUNT(*)",
    "count_distinct_cve": "COUNT(DISTINCT cve_id)",
    "mttr":              "AVG(days_open)",
}


def _resolve_window(query: Query) -> tuple[date, date]:
    tw = query.time
    if tw.mode == "date_range":
        return tw.start, tw.end
    if tw.mode == "last_n_snapshots":
        end = date.today()
        start = end - timedelta(days=tw.n - 1)
        return start, end
    # all_time — pull full range from DB (caller supplies conn, handled in compile)
    return date(2020, 1, 1), date.today()


def _date_spine(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _build_filter_clause(filters) -> tuple[str, list]:
    if not filters:
        return "", []
    clauses, params = [], []
    for f in filters:
        op = _OPERATOR_MAP.get(f.operator, "=")
        if f.operator == "in":
            placeholders = ", ".join(["?"] * len(f.value))
            clauses.append(f"{f.field} IN ({placeholders})")
            params.extend(f.value)
        else:
            clauses.append(f"{f.field} {op} ?")
            params.append(f.value)
    return " AND " + " AND ".join(clauses), params


def _compile_rollup(query: Query, conn, table: str, start: date, end: date) -> QueryResult:
    t0 = time.monotonic()
    pk_col = _ROLLUP_LENS_PK[table]
    measure_col = _ROLLUP_MEASURE_COL[query.measure]
    filter_sql, filter_params = _build_filter_clause(query.filters)

    if query.group_by:
        group_cols = ", ".join(query.group_by)
        sql = (
            f"SELECT date, {pk_col}, {group_cols}, {measure_col} AS value "
            f"FROM {table} "
            f"WHERE date BETWEEN ? AND ?{filter_sql} "
            f"ORDER BY date"
        )
    else:
        sql = (
            f"SELECT date, {pk_col}, {measure_col} AS value "
            f"FROM {table} "
            f"WHERE date BETWEEN ? AND ?{filter_sql} "
            f"ORDER BY date"
        )

    rows = conn.execute(sql, [start, end] + filter_params).fetchall()
    col_names = [d[0] for d in conn.description]

    # Group rows into Series by (pk_col, group_by values)
    groups: dict[tuple, tuple[list, list]] = {}
    all_dates = set(_date_spine(start, end))
    seen_dates: set[date] = set()
    for row in rows:
        row_dict = dict(zip(col_names, row))
        key_vals = {pk_col: row_dict[pk_col]}
        if query.group_by:
            for g in query.group_by:
                key_vals[g] = row_dict.get(g)
        group_key = tuple(sorted(key_vals.items()))
        if group_key not in groups:
            groups[group_key] = ([], [])
        groups[group_key][0].append(row_dict["date"])
        groups[group_key][1].append(row_dict["value"])
        seen_dates.add(row_dict["date"])

    series = [Series(key=dict(k), x=v[0], y=v[1]) for k, v in groups.items()]
    missing = sorted(all_dates - seen_dates)
    exec_ms = int((time.monotonic() - t0) * 1000)
    return QueryResult(series=series, dimensions={}, snapshot_range=(start, end), missing_days=missing, exec_time_ms=exec_ms)


def _compile_direct(query: Query, conn, start: date, end: date) -> QueryResult:
    t0 = time.monotonic()
    lens_meta = LENSES[query.lens]
    pk = lens_meta["pk"]
    pk_col = pk if isinstance(pk, str) else pk[0]

    date_col, base_predicate = _DIRECT_DATE_COL[query.measure]
    aggregate = _DIRECT_AGGREGATE[query.measure]
    filter_sql, filter_params = _build_filter_clause(query.filters)

    group_cols_sql = ""
    if query.group_by:
        group_cols_sql = ", " + ", ".join(query.group_by)

    # Build select: date bucket, entity pk, optional group_by cols, aggregate
    sql = (
        f"SELECT CAST({date_col} AS DATE) AS date, {pk_col}{group_cols_sql}, {aggregate} AS value "
        f"FROM finding_state "
        f"WHERE {base_predicate} "
        f"  AND CAST({date_col} AS DATE) BETWEEN ? AND ?{filter_sql} "
        f"GROUP BY date, {pk_col}{group_cols_sql} "
        f"ORDER BY date"
    )

    rows = conn.execute(sql, [start, end] + filter_params).fetchall()
    col_names = [d[0] for d in conn.description]

    all_dates = set(_date_spine(start, end))
    seen_dates: set[date] = set()
    groups: dict[tuple, tuple[list, list]] = {}
    for row in rows:
        row_dict = dict(zip(col_names, row))
        key_vals = {pk_col: row_dict[pk_col]}
        if query.group_by:
            for g in query.group_by:
                key_vals[g] = row_dict.get(g)
        group_key = tuple(sorted(key_vals.items()))
        if group_key not in groups:
            groups[group_key] = ([], [])
        groups[group_key][0].append(row_dict["date"])
        groups[group_key][1].append(row_dict["value"])
        seen_dates.add(row_dict["date"])

    series = [Series(key=dict(k), x=v[0], y=v[1]) for k, v in groups.items()]
    missing = sorted(all_dates - seen_dates)
    exec_ms = int((time.monotonic() - t0) * 1000)
    return QueryResult(series=series, dimensions={}, snapshot_range=(start, end), missing_days=missing, exec_time_ms=exec_ms)


def compile(query: Query, conn) -> QueryResult:
    """Compile a Query to SQL and execute against conn. Returns QueryResult."""
    start, end = _resolve_window(query)
    rollup_table = can_use_rollup(query)
    if rollup_table:
        return _compile_rollup(query, conn, rollup_table, start, end)
    return _compile_direct(query, conn, start, end)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_compiler.py -v`
Expected: 9 passed. (Rollup spy test requires the rollup table rows seeded in fixture — verify `daily_metrics_by_image` schema matches the INSERT in the fixture.)

- [ ] **Step 5: Commit**

```bash
git add sas/query/compiler.py tests/test_compiler.py
git commit -m "feat(sas): sql compiler — rollup + direct paths, all 6 measures"
```

---

## Task 6: FastAPI app + deps

**Files:**
- Create: `sas/api/main.py`
- Create: `sas/api/deps.py`
- Create: `tests/test_api.py` (health check only for now — routes added in Task 7)

- [ ] **Step 1: Write failing test**

File: `tests/test_api.py` (initial — health check only)

```python
from fastapi.testclient import TestClient
from sas.api.main import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sas.api.main'`

- [ ] **Step 3: Write deps.py**

File: `sas/api/deps.py`

```python
"""FastAPI dependency: provides a read-only DuckDB connection per request.

Reads SAS_DATA_DIR from environment (default: ~/sysdig-vuln-data/).
The ingest pipeline owns the write connection; the API is read-only.
"""

import os
from pathlib import Path
from typing import Generator

import duckdb
from fastapi import HTTPException


def _db_path() -> Path:
    data_dir = Path(os.environ.get("SAS_DATA_DIR", Path.home() / "sysdig-vuln-data"))
    return data_dir / "sas.duckdb"


def get_db() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """FastAPI dependency — yields a read-only DuckDB connection, closes on exit."""
    path = _db_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Database not found at {path}. Run ingest first.")
    conn = duckdb.connect(str(path), read_only=True)
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 4: Write main.py**

File: `sas/api/main.py`

```python
"""FastAPI application entry point for Sysdig Analytics Studio API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sas.api.routes import query, widgets, entities

app = FastAPI(
    title="Sysdig Analytics Studio API",
    description="Query engine for vulnerability trend analytics.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tightened in production via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router,    prefix="/api")
app.include_router(widgets.router,  prefix="/api")
app.include_router(entities.router, prefix="/api")


@app.get("/healthz", tags=["ops"])
def healthz():
    return {"status": "ok"}
```

- [ ] **Step 5: Create stub route files so main.py can import**

File: `sas/api/routes/query.py` (stub — full implementation in Task 7)

```python
from fastapi import APIRouter
router = APIRouter()
```

File: `sas/api/routes/widgets.py` (stub)

```python
from fastapi import APIRouter
router = APIRouter()
```

File: `sas/api/routes/entities.py` (stub)

```python
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add sas/api/main.py sas/api/deps.py sas/api/routes/query.py sas/api/routes/widgets.py sas/api/routes/entities.py tests/test_api.py
git commit -m "feat(sas): fastapi app skeleton + /healthz + read-only db dependency"
```

---

## Task 7: API routes

**Files:**
- Modify: `sas/api/routes/query.py`
- Modify: `sas/api/routes/widgets.py`
- Modify: `sas/api/routes/entities.py`
- Modify: `tests/test_api.py`

**The 10 starter widget definitions** (hardcoded in `widgets.py`):

| # | title | lens | measure | granularity | widget_type | notes |
|---|---|---|---|---|---|---|
| 1 | Image Remediation Story | Image | count_open | day | line | group_by severity |
| 2 | Fleet Critical Trend | Image | count_open | day | line | filter severity=Critical |
| 3 | New Findings This Week | Image | count_new | day | bar | last_n=7 |
| 4 | Fixed Findings This Week | Image | count_fixed | day | bar | last_n=7 |
| 5 | Regressed Findings | Image | count_regressed | day | bar | last_n=30 |
| 6 | MTTR by Team | Team | mttr | day | bar | last_n=90 |
| 7 | Unique CVEs Open | Image | count_distinct_cve | day | line | last_n=90 |
| 8 | Team Leaderboard — Open | Team | count_open | day | bar | last_n=30 |
| 9 | Repository Risk Trend | Repository | count_open | day | line | last_n=60 |
| 10 | Cluster Open Findings | Cluster | count_open | day | bar | last_n=30 |

- [ ] **Step 1: Write route tests (add to tests/test_api.py)**

Append these tests to `tests/test_api.py`:

```python
import duckdb
import pytest
from unittest.mock import patch
from sas.query.primitives import Query, TimeWindow
from sas.ingest.schema import create_schema


@pytest.fixture
def mock_db():
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    return conn


def test_query_endpoint(mock_db):
    with patch("sas.api.routes.query.get_db", return_value=iter([mock_db])):
        payload = {
            "lens": "Image",
            "traversal": [],
            "time": {"mode": "last_n_snapshots", "n": 7, "granularity": "day"},
            "measure": "count_open",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        }
        response = client.post("/api/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "series" in data
        assert "exec_time_ms" in data


def test_widgets_catalog():
    response = client.get("/api/widgets/catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert len(catalog) == 10
    for w in catalog:
        assert "title" in w
        assert "widget_type" in w
        assert "query" in w


def test_entities_image(mock_db):
    with patch("sas.api.routes.entities.get_db", return_value=iter([mock_db])):
        response = client.get("/api/entities/Image")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_entities_unknown_lens():
    response = client.get("/api/entities/UnknownLens")
    assert response.status_code == 422
```

- [ ] **Step 2: Write query route**

File: `sas/api/routes/query.py`

```python
from fastapi import APIRouter, Depends
from sas.api.deps import get_db
from sas.query.primitives import Query, QueryResult
from sas.query.compiler import compile as sas_compile

router = APIRouter()


@router.post("/query", response_model=None, tags=["query"])
def run_query(query: Query, conn=Depends(get_db)) -> dict:
    """Execute a Query against DuckDB and return a QueryResult."""
    result: QueryResult = sas_compile(query, conn)
    return {
        "series": [
            {"key": s.key, "x": [str(d) for d in s.x], "y": s.y}
            for s in result.series
        ],
        "dimensions": result.dimensions,
        "snapshot_range": [str(result.snapshot_range[0]), str(result.snapshot_range[1])],
        "missing_days": [str(d) for d in result.missing_days],
        "exec_time_ms": result.exec_time_ms,
    }
```

- [ ] **Step 3: Write widgets route**

File: `sas/api/routes/widgets.py`

```python
"""Hardcoded catalog of 10 starter widget definitions."""

from fastapi import APIRouter

router = APIRouter()

_CATALOG = [
    {"title": "Image Remediation Story",    "widget_type": "line", "query": {"lens": "Image",      "traversal": [], "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"}, "measure": "count_open",          "filters": [],                                              "group_by": ["severity"], "order_by": None, "limit": None}},
    {"title": "Fleet Critical Trend",       "widget_type": "line", "query": {"lens": "Image",      "traversal": [], "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"}, "measure": "count_open",          "filters": [{"field": "severity", "operator": "eq", "value": "Critical"}], "group_by": [], "order_by": None, "limit": None}},
    {"title": "New Findings This Week",     "widget_type": "bar",  "query": {"lens": "Image",      "traversal": [], "time": {"mode": "last_n_snapshots", "n": 7,  "granularity": "day"}, "measure": "count_new",           "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
    {"title": "Fixed Findings This Week",   "widget_type": "bar",  "query": {"lens": "Image",      "traversal": [], "time": {"mode": "last_n_snapshots", "n": 7,  "granularity": "day"}, "measure": "count_fixed",         "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
    {"title": "Regressed Findings",         "widget_type": "bar",  "query": {"lens": "Image",      "traversal": [], "time": {"mode": "last_n_snapshots", "n": 30, "granularity": "day"}, "measure": "count_regressed",     "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
    {"title": "MTTR by Team",               "widget_type": "bar",  "query": {"lens": "Team",       "traversal": [], "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"}, "measure": "mttr",                "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
    {"title": "Unique CVEs Open",           "widget_type": "line", "query": {"lens": "Image",      "traversal": [], "time": {"mode": "last_n_snapshots", "n": 90, "granularity": "day"}, "measure": "count_distinct_cve",  "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
    {"title": "Team Leaderboard — Open",    "widget_type": "bar",  "query": {"lens": "Team",       "traversal": [], "time": {"mode": "last_n_snapshots", "n": 30, "granularity": "day"}, "measure": "count_open",          "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
    {"title": "Repository Risk Trend",      "widget_type": "line", "query": {"lens": "Repository", "traversal": [], "time": {"mode": "last_n_snapshots", "n": 60, "granularity": "day"}, "measure": "count_open",          "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
    {"title": "Cluster Open Findings",      "widget_type": "bar",  "query": {"lens": "Cluster",    "traversal": [], "time": {"mode": "last_n_snapshots", "n": 30, "granularity": "day"}, "measure": "count_open",          "filters": [],                                              "group_by": [],           "order_by": None, "limit": None}},
]


@router.get("/widgets/catalog", tags=["widgets"])
def get_catalog() -> list[dict]:
    """Return the 10 hardcoded starter widget definitions."""
    return _CATALOG
```

- [ ] **Step 4: Write entities route**

File: `sas/api/routes/entities.py`

```python
"""Entity picker endpoint — returns values for UI dropdowns."""

from fastapi import APIRouter, Depends, HTTPException
from sas.api.deps import get_db
from sas.query.registry import LENSES

router = APIRouter()

_ENTITY_QUERIES = {
    "Image":      "SELECT image_id AS id, COALESCE(current_repository || ':' || current_tag, image_id) AS label, current_repository AS repository, current_tag AS tag FROM image ORDER BY label",
    "CVE":        "SELECT cve_id AS id, cve_id AS label, initial_severity AS severity FROM cve ORDER BY cve_id",
    "Workload":   "SELECT workload_name AS id, cluster_name || '/' || namespace_name || '/' || workload_name AS label, cluster_name, namespace_name FROM workload ORDER BY label",
    "Cluster":    "SELECT cluster_name AS id, cluster_name AS label FROM cluster ORDER BY label",
    "Namespace":  "SELECT cluster_name || '/' || namespace_name AS id, cluster_name || '/' || namespace_name AS label, cluster_name, namespace_name FROM namespace ORDER BY label",
    "Package":    "SELECT package_name AS id, package_name AS label, package_type FROM package ORDER BY label",
    "Repository": "SELECT repository AS id, repository AS label FROM repository ORDER BY label",
    "Team":       "SELECT team_id AS id, display_name AS label FROM team ORDER BY label",
    "Owner":      "SELECT owner_id AS id, display_name AS label FROM owner ORDER BY label",
}


@router.get("/entities/{lens}", tags=["entities"])
def get_entities(lens: str, conn=Depends(get_db)) -> list[dict]:
    """Return entity values for a given lens, for use in UI pickers."""
    if lens not in LENSES:
        raise HTTPException(status_code=422, detail=f"Unknown lens: {lens}. Valid: {list(LENSES.keys())}")
    sql = _ENTITY_QUERIES[lens]
    rows = conn.execute(sql).fetchall()
    col_names = [d[0] for d in conn.description]
    return [dict(zip(col_names, row)) for row in rows]
```

- [ ] **Step 5: Run all API tests**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: 5 passed.

- [ ] **Step 6: Run full test suite to confirm no regressions**

Run: `.venv/bin/pytest -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add sas/api/routes/query.py sas/api/routes/widgets.py sas/api/routes/entities.py tests/test_api.py
git commit -m "feat(sas): api routes — /query, /widgets/catalog, /entities/{lens}"
```

---

## Task 8: Startup script + README snippet

**Files:**
- Create: `sas/api/run.py`
- Create: `sas/README.md`

- [ ] **Step 1: Write run.py**

File: `sas/api/run.py`

```python
"""Start the SAS API server. Usage: python -m sas.api.run"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("sas.api.main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 2: Write sas/README.md**

File: `sas/README.md`

```markdown
# Sysdig Analytics Studio (SAS)

```bash
# 1. Ingest a CSV export
python -m sas.ingest ~/downloads/phoenix-vuln-findings-2026_04_23.csv

# 2. Start the API (reads ~/sysdig-vuln-data/sas.duckdb by default)
python -m sas.api.run
# → http://localhost:8000/docs  (OpenAPI)
# → http://localhost:8000/healthz

# 3. Run tests
.venv/bin/pytest -v
```
```

- [ ] **Step 3: Verify the server starts (smoke test)**

Run: `.venv/bin/python -m sas.api.run &`
Run: `curl -s http://localhost:8000/healthz`
Expected: `{"status":"ok"}`
Kill the background process.

- [ ] **Step 4: Commit**

```bash
git add sas/api/run.py sas/README.md
git commit -m "feat(sas): api startup script + sas/README"
```

---

## Summary Checklist

| Task | Deliverable | Tests |
|---|---|---|
| 1 | Package scaffolding, fastapi/uvicorn installed | import smoke test |
| 2 | `primitives.py` — 6 dataclasses | 7 unit tests |
| 3 | `registry.py` + `measures.py` — 9 lenses, 6 measures | 10 unit tests |
| 4 | `rollup_router.py` | 12 unit tests |
| 5 | `compiler.py` — rollup + direct paths | 9 tests (1 per measure + rollup spy + group_by + filter) |
| 6 | `main.py` + `deps.py` | 1 health check test |
| 7 | 3 route files | 4 API tests |
| 8 | `run.py` + `sas/README.md` | curl smoke test |

**Total new tests: ~43. Total new files: 14.**
