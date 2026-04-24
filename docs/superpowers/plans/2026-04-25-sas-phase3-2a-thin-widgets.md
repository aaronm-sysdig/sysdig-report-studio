# SAS Phase 3.2a — Thin Widgets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver 5 new dashboard widgets plus a prerequisite hotfix that denormalises `cisa_kev_known_ransomware` onto `finding_state`. Widgets use the shared chart primitives locked in Phase 3.1. Phase 3.2a ends with all 6 widgets (Fleet Critical Trend + 5 new) visible on `/dashboard` and smoke-tested.

**Architecture:** Same as Phase 3.1. Every new widget follows the `FleetCriticalTrend.tsx` pattern exactly — see that file as the canonical reference. Backend hotfix is 3 file touches in `sas/ingest/` and one scenario test.

**Design references:**
- Spec: [`docs/superpowers/specs/2026-04-24-sas-phase3-frontend-design.md`](../specs/2026-04-24-sas-phase3-frontend-design.md) — §2 tenets (especially "Flowing lines, anchored observations"), §8 Widget card shell, §9 Widget catalog.
- Phase 3.1 plan (structural template): [`docs/superpowers/plans/2026-04-24-sas-phase3-1-foundation.md`](./2026-04-24-sas-phase3-1-foundation.md).
- Shared chart primitives: `sas/web/lib/charts/defaults.ts` — `CHART_COLORS`, `flowingLineSeries`, `standardXAxis`, `STANDARD_Y_AXIS`, `STANDARD_TOOLTIP_STYLE`, `standardGrid`.
- Reference widget: `sas/web/components/widgets/FleetCriticalTrend.tsx`.
- Phase 2 measures: `sas/query/measures.py` — `count_open`, `count_new`, `count_fixed`, `count_regressed`, `count_open_critical` already registered.
- Phase 2 API: `sas/api/routes/query.py`, `sas/api/routes/entities.py`.
- Scenario test helper: `tests/scenarios/_builder.py`.

**Budget estimate:** ~$15–20. All 8 tasks dispatch to Sonnet 4.6. Opus reviews after Task 1 (hotfix) and at Task 8 (smoke test).

**Collaboration note:** Tasks 2–6 are independently parallelisable after Task 1 completes. Task 7 depends on Tasks 2–6. Task 8 depends on Task 7.

---

## Critical warnings for Sonnet workers

1. **Read `sas/web/AGENTS.md` before writing any Next.js code.** The installed Next.js version has breaking changes from training data. Check `node_modules/next/dist/docs/` for the relevant guide.
2. **Tailwind v4 has no `tailwind.config.ts`.** Theme tokens live in `app/globals.css` via `@theme inline`. Do not use utility classes that don't exist — use `style={{ ... }}` with CSS vars instead (e.g. `style={{ borderRadius: "var(--radius)" }}`).
3. **CSS vars do not resolve inside ECharts canvas.** Use `CHART_COLORS.*` hex constants from `@/lib/charts/defaults` — never `var(--...)` in ECharts option objects.
4. **Never use `step: "end"` or `step: "start"` on line series.** Use `flowingLineSeries` from shared primitives, which enforces `smooth: 0.4` with visible dots.
5. **Measures already registered:** `count_open_critical`, `count_open_high`, `count_open_medium`, `count_open_low`. Use these instead of `count_open + severity filter` for severity-specific counts.
6. **British English in all user-facing strings.** "colour", "organise", "centre", "analyse". Code identifiers stay American English.

---

## File structure

Files created or modified in Phase 3.2a:

```
sas/ingest/
├── schema.py                         MODIFY — add cisa_kev_known_ransomware col to finding_state
├── entity_upsert.py                  MODIFY — pass kev flag through to finding insert
└── finding_diff.py                   MODIFY — include kev flag in INSERT + RESEEN UPDATE

sas/api/routes/
└── findings.py                       CREATE — GET /api/findings endpoint for FindingsTable

sas/web/components/widgets/
├── NewFixedRegressed.tsx             CREATE — task 2
├── KevRansomwareExposure.tsx         CREATE — task 3
├── ImageInventoryGrid.tsx            CREATE — task 4
├── RepositoryTagHygiene.tsx          CREATE — task 5
└── FindingsTable.tsx                 CREATE — task 6

sas/web/lib/api/
└── client.ts                         MODIFY — add getFindingRows() helper

sas/web/app/dashboard/
└── page.tsx                          MODIFY — 4-row 12-col grid with all 6 widgets

tests/scenarios/kev_ransomware/
├── generate.py                       CREATE — fixture generator for Task 1 scenario
└── test_kev_ransomware_denorm.py     CREATE — scenario test for Task 1
```

---

## Task 1 — Phase 2.2 hotfix: denormalise cisa_kev_known_ransomware onto finding_state

**Opus review gate: Opus reviews this task's output before Tasks 2–6 start.**

This hotfix denormalises `cisa_kev_known_ransomware` from the `cve` table onto `finding_state` at ingest time. The column already exists on `cve` and is populated during entity upsert. This task adds it to `finding_state` so widgets can filter on it directly without compiler joins.

`public_exploit` is the precedent: it is also a CVE-derived boolean already on `finding_state`. This hotfix follows the same pattern.

**Files:**
- Modify: `sas/ingest/schema.py`
- Modify: `sas/ingest/finding_diff.py`
- Modify: `sas/ingest/entity_upsert.py` (no-op — `cisa_kev_known_ransomware` already flows through `df`; finding_diff reads it from `df` via `_row_to_fs_values`)
- Create: `tests/scenarios/kev_ransomware/generate.py`
- Create: `tests/scenarios/kev_ransomware/test_kev_ransomware_denorm.py`

- [ ] **Step 1: Add column to finding_state DDL in schema.py**

Open `sas/ingest/schema.py`. In the `finding_state` CREATE TABLE statement, add `cisa_kev_known_ransomware` after `public_exploit`:

```python
# Before (line ~211):
        public_exploit BOOLEAN,
        first_seen TIMESTAMPTZ,

# After:
        public_exploit BOOLEAN,
        cisa_kev_known_ransomware BOOLEAN DEFAULT FALSE,
        first_seen TIMESTAMPTZ,
```

The full `finding_state` DDL block after the change (only the affected excerpt shown):

```sql
CREATE TABLE IF NOT EXISTS finding_state (
    finding_id BIGINT PRIMARY KEY,
    image_id VARCHAR,
    cve_id VARCHAR,
    package_name VARCHAR,
    package_version VARCHAR,
    package_path VARCHAR,
    severity VARCHAR,
    cvss_score DOUBLE,
    in_use BOOLEAN,
    fix_available BOOLEAN,
    fix_version VARCHAR,
    risk_accepted BOOLEAN,
    public_exploit BOOLEAN,
    cisa_kev_known_ransomware BOOLEAN DEFAULT FALSE,
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    state VARCHAR,
    reason_code VARCHAR,
    closed_at TIMESTAMPTZ,
    reopened_at TIMESTAMPTZ,
    reopen_count INTEGER DEFAULT 0,
    days_open INTEGER,
    is_regression BOOLEAN DEFAULT FALSE
)
```

- [ ] **Step 2: Add schema migration helper to schema.py**

Add this function below `create_schema()`:

```python
def migrate_schema(conn) -> None:
    """Apply schema migrations for databases created before Phase 2.2.

    Safe to call on a fresh DB (columns already present) or an existing DB
    (ALTER TABLE ADD COLUMN IF NOT EXISTS is idempotent in DuckDB).
    """
    conn.execute(
        "ALTER TABLE finding_state "
        "ADD COLUMN IF NOT EXISTS cisa_kev_known_ransomware BOOLEAN DEFAULT FALSE"
    )
```

- [ ] **Step 3: Call migrate_schema from the ingest pipeline**

Open `sas/ingest/pipeline.py`. Find where `create_schema(conn)` is called. Add a call to `migrate_schema(conn)` immediately after it:

```python
from sas.ingest.schema import create_schema, migrate_schema   # update existing import

# In the pipeline function, after create_schema:
create_schema(conn)
migrate_schema(conn)
```

- [ ] **Step 4: Thread cisa_kev_known_ransomware through _row_to_fs_values in finding_diff.py**

Open `sas/ingest/finding_diff.py`. Update `_row_to_fs_values` to include the new field:

```python
def _row_to_fs_values(r, snapshot_at: datetime):  # noqa: ARG001
    return {
        "image_id": r["image_id"],
        "cve_id": r["vulnerability_name"],
        "package_name": r["package_name"],
        "package_version": r["package_version"],
        "package_path": r["package_path"],
        "severity": r["vulnerability_severity"],
        "cvss_score": (
            float(r["cvss_score"]) if pd.notna(r["cvss_score"]) else None
        ),
        "in_use": bool(r["package_in_use"]),
        "fix_available": bool(r["fix_available"]),
        "fix_version": (
            r["fix_version"] if pd.notna(r["fix_version"]) else None
        ),
        "risk_accepted": bool(r["risk_accepted"]),
        "public_exploit": bool(r["public_exploit"]),
        "cisa_kev_known_ransomware": bool(r["cisa_kev_known_ransomware"]),  # NEW
    }
```

- [ ] **Step 5: Include cisa_kev_known_ransomware in _insert_finding_row**

In `finding_diff.py`, update `_insert_finding_row` to include the new column in both the column list and the values list:

```python
def _insert_finding_row(
    conn, v, snapshot_at, *, reopened_at, reopen_count, is_regression
):
    conn.execute(
        """
        INSERT INTO finding_state (
          finding_id, image_id, cve_id, package_name, package_version,
          package_path, severity, cvss_score, in_use, fix_available,
          fix_version, risk_accepted, public_exploit, cisa_kev_known_ransomware,
          first_seen, last_seen,
          state, reason_code, closed_at, reopened_at, reopen_count,
          days_open, is_regression
        ) VALUES (
          nextval('seq_finding_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, 'OPEN', NULL, NULL, ?, ?, 0, ?
        )
        """,
        [v["image_id"], v["cve_id"], v["package_name"], v["package_version"],
         v["package_path"], v["severity"], v["cvss_score"], v["in_use"],
         v["fix_available"], v["fix_version"], v["risk_accepted"],
         v["public_exploit"], v["cisa_kev_known_ransomware"],
         snapshot_at, snapshot_at,
         reopened_at, reopen_count, is_regression],
    )
```

- [ ] **Step 6: Include cisa_kev_known_ransomware in RESEEN UPDATE**

In `finding_diff.py`, in the RESEEN branch of `diff_and_apply_findings`, extend the UPDATE statement to also refresh `cisa_kev_known_ransomware` (KEV flag can change as CISA updates the catalogue):

```python
            conn.execute(
                """
                UPDATE finding_state SET
                  last_seen = ?, severity = ?, cvss_score = ?, in_use = ?,
                  fix_available = ?, fix_version = ?, risk_accepted = ?,
                  public_exploit = ?, cisa_kev_known_ransomware = ?, days_open = ?
                WHERE finding_id = ?
                """,
                [snapshot_at, v["severity"], v["cvss_score"], v["in_use"],
                 v["fix_available"], v["fix_version"], v["risk_accepted"],
                 v["public_exploit"], v["cisa_kev_known_ransomware"],
                 days_open, prior["finding_id"]],
            )
```

Also add `"cisa_kev_known_ransomware"` to `_DRIFT_COLUMNS` at the top of the file:

```python
_DRIFT_COLUMNS = {
    "severity", "cvss_score", "fix_available", "fix_version",
    "risk_accepted", "public_exploit", "in_use", "cisa_kev_known_ransomware",
}
```

- [ ] **Step 7: Create scenario fixture generator**

File: `tests/scenarios/kev_ransomware/generate.py`

```python
"""Generate two-day fixture for KEV ransomware denormalisation scenario.

Day 1: one ransomware-flagged CVE, one non-ransomware CVE, same image.
Day 2: both CVEs still open (reseen path exercises the UPDATE branch).
"""
from pathlib import Path

from tests.scenarios._builder import ScenarioBuilder

OUT = Path(__file__).parent

def main() -> None:
    b = ScenarioBuilder()
    # Ransomware-flagged finding
    b.add_finding(
        vulnerability_name="CVE-2024-RANSOM-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="true",
        vulnerability_severity="Critical",
        cvss_score=9.8,
    )
    # Non-ransomware finding on same image
    b.add_finding(
        vulnerability_name="CVE-2024-NORMAL-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="false",
        vulnerability_severity="High",
        cvss_score=7.5,
    )
    b.write_csv(OUT / "day1_2026-05-01.csv")

    # Day 2 — same findings reseen
    b.clear()
    b.add_finding(
        vulnerability_name="CVE-2024-RANSOM-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="true",
        vulnerability_severity="Critical",
        cvss_score=9.8,
    )
    b.add_finding(
        vulnerability_name="CVE-2024-NORMAL-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="false",
        vulnerability_severity="High",
        cvss_score=7.5,
    )
    b.write_csv(OUT / "day2_2026-05-02.csv")
    print("Generated day1_2026-05-01.csv and day2_2026-05-02.csv")

if __name__ == "__main__":
    main()
```

Run:
```bash
cd /path/to/repo && python tests/scenarios/kev_ransomware/generate.py
```

- [ ] **Step 8: Create scenario test**

File: `tests/scenarios/kev_ransomware/test_kev_ransomware_denorm.py`

```python
"""Scenario: cisa_kev_known_ransomware propagates to finding_state at ingest.

Verifies:
1. NEW path: ransomware flag written to finding_state on first insert.
2. Non-ransomware finding on same image has flag = False.
3. RESEEN path: flag refreshed on second ingest (UPDATE branch).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from sas.ingest.pipeline import run_pipeline
from sas.ingest.schema import create_schema, migrate_schema

FIXTURES = Path(__file__).parent


@pytest.fixture()
def db():
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    migrate_schema(conn)
    return conn


def test_kev_flag_propagates_on_new(db) -> None:
    """Ransomware flag must be True on finding_state after first ingest."""
    run_pipeline(db, FIXTURES / "day1_2026-05-01.csv",
                 snapshot_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    row = db.execute(
        "SELECT cisa_kev_known_ransomware FROM finding_state "
        "WHERE cve_id = 'CVE-2024-RANSOM-1'"
    ).fetchone()
    assert row is not None, "Finding for CVE-2024-RANSOM-1 not found"
    assert row[0] is True, f"Expected True, got {row[0]}"


def test_non_kev_flag_is_false(db) -> None:
    """Non-ransomware finding must have flag = False."""
    run_pipeline(db, FIXTURES / "day1_2026-05-01.csv",
                 snapshot_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    row = db.execute(
        "SELECT cisa_kev_known_ransomware FROM finding_state "
        "WHERE cve_id = 'CVE-2024-NORMAL-1'"
    ).fetchone()
    assert row is not None, "Finding for CVE-2024-NORMAL-1 not found"
    assert row[0] is False, f"Expected False, got {row[0]}"


def test_kev_flag_persists_on_reseen(db) -> None:
    """Ransomware flag must remain True after the RESEEN UPDATE on day 2."""
    run_pipeline(db, FIXTURES / "day1_2026-05-01.csv",
                 snapshot_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    run_pipeline(db, FIXTURES / "day2_2026-05-02.csv",
                 snapshot_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    rows = db.execute(
        "SELECT cisa_kev_known_ransomware FROM finding_state "
        "WHERE cve_id = 'CVE-2024-RANSOM-1'"
    ).fetchall()
    # Both the original OPEN row should exist — reseen, not re-inserted
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    assert rows[0][0] is True


def test_filter_on_kev_column_works(db) -> None:
    """Direct boolean filter on finding_state.cisa_kev_known_ransomware returns only ransomware rows."""
    run_pipeline(db, FIXTURES / "day1_2026-05-01.csv",
                 snapshot_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    kev_rows = db.execute(
        "SELECT COUNT(*) FROM finding_state WHERE cisa_kev_known_ransomware = TRUE"
    ).fetchone()[0]
    total_rows = db.execute("SELECT COUNT(*) FROM finding_state").fetchone()[0]

    assert kev_rows == 1, f"Expected 1 KEV row, got {kev_rows}"
    assert total_rows == 2, f"Expected 2 total rows, got {total_rows}"
```

- [ ] **Step 9: Generate fixtures and run tests**

```bash
# Generate CSV fixtures
python tests/scenarios/kev_ransomware/generate.py

# Run scenario tests
python -m pytest tests/scenarios/kev_ransomware/ -v
```

Expected output: 4 tests pass.

- [ ] **Step 10: Re-ingest 7 days of lab data**

Wipe and re-ingest to pick up the new column on existing data:

```bash
# Wipe the lab DB
rm -f ~/sysdig-vuln-data/sas.duckdb

# Re-ingest all 7 days (adjust filenames to your actual lab files)
for f in ~/sysdig-vuln-data/day*.csv; do
    python -m sas.ingest.cli "$f"
done
```

- [ ] **Step 11: Verify filter works against live DB**

Start the API server:
```bash
.venv/bin/python -m sas.api.run &
```

Run:
```bash
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "lens": "Image",
    "traversal": [],
    "time": {"mode": "last_n_snapshots", "n": 7, "granularity": "day"},
    "measure": "count_open",
    "filters": [{"field": "cisa_kev_known_ransomware", "operator": "eq", "value": true}],
    "group_by": [],
    "order_by": null,
    "limit": null
  }' | python -m json.tool
```

Expected: JSON response with `series` array (may be empty if no KEV ransomware CVEs in lab data — that is acceptable; no HTTP 500 error is the pass criterion).

- [ ] **Step 12: Commit**

```bash
git add sas/ingest/schema.py sas/ingest/finding_diff.py sas/ingest/pipeline.py \
        tests/scenarios/kev_ransomware/
git commit -m "$(cat <<'EOF'
feat(sas): denormalise kev ransomware flag onto finding_state

Phase 2.2 hotfix. Adds cisa_kev_known_ransomware BOOLEAN to finding_state,
populated at INSERT and refreshed on RESEEN. Enables KEV-Ransomware widget
to filter directly on finding_state without compiler joins. Includes schema
migration (ADD COLUMN IF NOT EXISTS) for existing databases. Four scenario
tests verify NEW, RESEEN, and filter paths.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Widget: New vs Fixed vs Regressed

**Depends on:** Task 1 complete (uses `count_new`, `count_fixed`, `count_regressed` — no KEV dependency, but wait for Task 1 before starting so the DB state is clean).

**Files:**
- Create: `sas/web/components/widgets/NewFixedRegressed.tsx`

This widget overlays three series on one chart: new findings (red line), fixed findings (green bars below zero), regressed findings (orange bars above zero). The flowing line for `count_new` enforces the "flowing lines, anchored observations" tenet. The bar series for fixed/regressed are standard ECharts bars.

Three separate `POST /api/query` calls are made in parallel (`Promise.all`) to fetch the three measures. Results are aligned by date before rendering.

- [ ] **Step 1: Create the component**

File: `sas/web/components/widgets/NewFixedRegressed.tsx`

```tsx
"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { WidgetCard } from "./WidgetCard";
import { runQuery } from "@/lib/api/client";
import type { QueryIn, QueryResult } from "@/lib/api/client";
import {
  CHART_COLORS,
  flowingLineSeries,
  standardGrid,
  standardXAxis,
  STANDARD_Y_AXIS,
  STANDARD_TOOLTIP_STYLE,
} from "@/lib/charts/defaults";

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// ---------------------------------------------------------------------------
// Query definitions — three measures, same time window
// ---------------------------------------------------------------------------
const BASE_TIME = { mode: "last_n_snapshots", n: 90, granularity: "day" } as const;

const NEW_QUERY: QueryIn = {
  lens: "Image", traversal: [], time: BASE_TIME,
  measure: "count_new", filters: [], group_by: [], order_by: null, limit: null,
};
const FIXED_QUERY: QueryIn = {
  lens: "Image", traversal: [], time: BASE_TIME,
  measure: "count_fixed", filters: [], group_by: [], order_by: null, limit: null,
};
const REGRESSED_QUERY: QueryIn = {
  lens: "Image", traversal: [], time: BASE_TIME,
  measure: "count_regressed", filters: [], group_by: [], order_by: null, limit: null,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function ChartSkeleton() {
  return (
    <div
      className="w-full h-[220px] animate-pulse"
      style={{ backgroundColor: "var(--bg-surface)", borderRadius: "var(--radius)" }}
      aria-label="Loading chart data…"
      role="status"
    />
  );
}

/** Aggregate all series in a QueryResult by summing y values per date. */
function aggregateByDate(result: QueryResult): Map<string, number> {
  const map = new Map<string, number>();
  for (const s of result.series) {
    for (let i = 0; i < s.x.length; i++) {
      const d = s.x[i];
      const v = typeof s.y[i] === "number" ? (s.y[i] as number) : 0;
      map.set(d, (map.get(d) ?? 0) + v);
    }
  }
  return map;
}

function buildChartOption(
  newResult: QueryResult,
  fixedResult: QueryResult,
  regressedResult: QueryResult,
  axisLabels: boolean,
): object {
  // Union of all dates across the three results
  const allDates = new Set<string>();
  for (const r of [newResult, fixedResult, regressedResult]) {
    for (const s of r.series) { for (const d of s.x) allDates.add(d); }
  }
  const dates = Array.from(allDates).sort();

  const newMap = aggregateByDate(newResult);
  const fixedMap = aggregateByDate(fixedResult);
  const regressedMap = aggregateByDate(regressedResult);

  const newCounts = dates.map((d) => newMap.get(d) ?? null);
  // Fixed displayed below zero (negative), regressed above zero (positive)
  const fixedCounts = dates.map((d) => {
    const v = fixedMap.get(d);
    return v != null ? -v : null;
  });
  const regressedCounts = dates.map((d) => regressedMap.get(d) ?? null);

  return {
    backgroundColor: "transparent",
    grid: standardGrid(axisLabels ? 32 : 0),
    xAxis: standardXAxis(dates, axisLabels),
    yAxis: {
      ...STANDARD_Y_AXIS,
      axisLabel: {
        ...STANDARD_Y_AXIS.axisLabel,
        formatter: (v: number) => {
          const abs = Math.abs(v);
          return abs >= 1000 ? `${(abs / 1000).toFixed(1)}k` : String(abs);
        },
      },
    },
    legend: {
      bottom: axisLabels ? 48 : 4,
      textStyle: { fontSize: 10, color: CHART_COLORS.greyMuted },
      itemHeight: 8,
      data: ["New", "Fixed", "Regressed"],
    },
    series: [
      {
        name: "New",
        ...flowingLineSeries({ color: CHART_COLORS.severityCritical }),
        data: newCounts,
        z: 3,
      },
      {
        name: "Fixed",
        type: "bar" as const,
        data: fixedCounts,
        itemStyle: { color: CHART_COLORS.lumin },
        barMaxWidth: 6,
        z: 2,
      },
      {
        name: "Regressed",
        type: "bar" as const,
        data: regressedCounts,
        itemStyle: { color: CHART_COLORS.severityHigh },
        barMaxWidth: 6,
        z: 2,
      },
    ],
    tooltip: {
      trigger: "axis",
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const ps = params as Array<{ seriesName: string; value: number | null; axisValue: string }>;
        if (!ps.length) return "";
        const date = ps[0].axisValue;
        const lines = ps.map((p) => {
          const abs = p.value != null ? Math.abs(p.value) : "—";
          return `<div><b>${abs}</b> ${p.seriesName.toLowerCase()}</div>`;
        });
        return `<div style="font-size:11px"><div style="color:${CHART_COLORS.greyMuted};margin-bottom:2px">${date}</div>${lines.join("")}</div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
type TripleResult = [QueryResult, QueryResult, QueryResult];

export function NewFixedRegressed() {
  const [results, setResults] = useState<TripleResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [axisLabels, setAxisLabels] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([runQuery(NEW_QUERY), runQuery(FIXED_QUERY), runQuery(REGRESSED_QUERY)])
      .then((r) => { if (!cancelled) setResults(r as TripleResult); })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load data.");
      });
    return () => { cancelled = true; };
  }, []);

  const footer = results
    ? (() => {
        const [nr, fr, rr] = results;
        const totalNew = Array.from(aggregateByDate(nr).values()).reduce((a, b) => a + b, 0);
        const totalFixed = Array.from(aggregateByDate(fr).values()).reduce((a, b) => a + b, 0);
        const totalRegressed = Array.from(aggregateByDate(rr).values()).reduce((a, b) => a + b, 0);
        return `Over the last 90 snapshots: ${totalNew.toLocaleString("en-GB")} new, ${totalFixed.toLocaleString("en-GB")} fixed, ${totalRegressed.toLocaleString("en-GB")} regressed.`;
      })()
    : undefined;

  let body: React.ReactNode;
  if (error) {
    body = (
      <div className="flex items-center justify-center h-[220px] text-sm"
        style={{ color: "var(--severity-critical)" }} role="alert">
        Unable to load data: {error}
      </div>
    );
  } else if (results === null) {
    body = <ChartSkeleton />;
  } else {
    const [nr, fr, rr] = results;
    const hasData = nr.series.length > 0 || fr.series.length > 0 || rr.series.length > 0;
    if (!hasData) {
      body = (
        <div className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--fg-muted)" }}>
          No findings data in this window.
        </div>
      );
    } else {
      body = (
        <ReactECharts
          option={buildChartOption(nr, fr, rr, axisLabels)}
          style={{ height: "220px", width: "100%" }}
          notMerge
          lazyUpdate={false}
        />
      );
    }
  }

  return (
    <WidgetCard
      label="Fleet Metrics"
      title="New vs Fixed vs Regressed"
      footer={footer}
      axisLabels={axisLabels}
      onAxisLabelsChange={setAxisLabels}
    >
      {body}
    </WidgetCard>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```

Expected: no errors for `NewFixedRegressed.tsx`.

- [ ] **Step 3: Commit**

```bash
git add sas/web/components/widgets/NewFixedRegressed.tsx
git commit -m "$(cat <<'EOF'
feat(sas): add New vs Fixed vs Regressed composite widget (3.2a task 2)

Three-series ECharts combo: flowing line for new findings, bar series for
fixed (below zero) and regressed (above zero). Footer narrative summarises
90-snapshot totals. Uses shared chart primitives throughout.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Widget: KEV-Ransomware Exposure Trend

**Depends on:** Task 1 complete (filters on `cisa_kev_known_ransomware` column added to `finding_state`).

**Files:**
- Create: `sas/web/components/widgets/KevRansomwareExposure.tsx`

Single flowing line chart in `CHART_COLORS.severityCritical` (alarming red). Uses `count_open` measure with a boolean filter `cisa_kev_known_ransomware == true`. Footer states the current count.

- [ ] **Step 1: Create the component**

File: `sas/web/components/widgets/KevRansomwareExposure.tsx`

```tsx
"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { WidgetCard } from "./WidgetCard";
import { runQuery } from "@/lib/api/client";
import type { QueryIn, QueryResult } from "@/lib/api/client";
import {
  CHART_COLORS,
  flowingLineSeries,
  standardGrid,
  standardXAxis,
  STANDARD_Y_AXIS,
  STANDARD_TOOLTIP_STYLE,
} from "@/lib/charts/defaults";

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// ---------------------------------------------------------------------------
// Query definition
// ---------------------------------------------------------------------------
const KEV_RANSOMWARE_QUERY: QueryIn = {
  lens: "Image",
  traversal: [],
  time: { mode: "last_n_snapshots", n: 90, granularity: "day" },
  measure: "count_open",
  filters: [
    { field: "cisa_kev_known_ransomware", operator: "eq", value: true },
  ],
  group_by: [],
  order_by: null,
  limit: null,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function ChartSkeleton() {
  return (
    <div
      className="w-full h-[220px] animate-pulse"
      style={{ backgroundColor: "var(--bg-surface)", borderRadius: "var(--radius)" }}
      aria-label="Loading chart data…"
      role="status"
    />
  );
}

function buildChartOption(result: QueryResult, axisLabels: boolean): object {
  const allDates = new Set<string>();
  for (const s of result.series) { for (const d of s.x) allDates.add(d); }
  const dates = Array.from(allDates).sort();

  const counts = dates.map((date) => {
    let total = 0;
    for (const s of result.series) {
      const idx = s.x.indexOf(date);
      if (idx >= 0 && typeof s.y[idx] === "number") total += s.y[idx] as number;
    }
    return total;
  });

  return {
    backgroundColor: "transparent",
    grid: standardGrid(axisLabels ? 32 : 0),
    xAxis: standardXAxis(dates, axisLabels),
    yAxis: STANDARD_Y_AXIS,
    series: [
      {
        ...flowingLineSeries({ color: CHART_COLORS.severityCritical, width: 2 }),
        data: counts,
        areaStyle: {
          color: {
            type: "linear",
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: `${CHART_COLORS.severityCritical}33` },
              { offset: 1, color: `${CHART_COLORS.severityCritical}00` },
            ],
          },
        },
      },
    ],
    tooltip: {
      trigger: "axis",
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const p = (params as Array<{ axisValue: string; value: number }>)[0];
        if (!p) return "";
        return `<div style="font-size:11px">
          <div style="color:${CHART_COLORS.greyMuted};margin-bottom:2px">${p.axisValue}</div>
          <div><b style="color:${CHART_COLORS.severityCritical}">${p.value?.toLocaleString("en-GB") ?? "—"}</b> ransomware-associated open</div>
        </div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function KevRansomwareExposure() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [axisLabels, setAxisLabels] = useState(false);

  useEffect(() => {
    let cancelled = false;
    runQuery(KEV_RANSOMWARE_QUERY)
      .then((r) => { if (!cancelled) setResult(r); })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load data.");
      });
    return () => { cancelled = true; };
  }, []);

  const footer = result
    ? (() => {
        const allDates = new Set<string>();
        for (const s of result.series) { for (const d of s.x) allDates.add(d); }
        const dates = Array.from(allDates).sort();
        if (dates.length === 0) return "No ransomware-associated CVEs currently open in your fleet.";
        const latest = dates[dates.length - 1];
        let count = 0;
        for (const s of result.series) {
          const idx = s.x.indexOf(latest);
          if (idx >= 0 && typeof s.y[idx] === "number") count += s.y[idx] as number;
        }
        return count === 0
          ? "No ransomware-associated CVEs currently open in your fleet."
          : `${count.toLocaleString("en-GB")} open findings linked to CISA KEV ransomware CVEs as of the latest snapshot.`;
      })()
    : undefined;

  let body: React.ReactNode;
  if (error) {
    body = (
      <div className="flex items-center justify-center h-[220px] text-sm"
        style={{ color: "var(--severity-critical)" }} role="alert">
        Unable to load data: {error}
      </div>
    );
  } else if (result === null) {
    body = <ChartSkeleton />;
  } else {
    const totalPoints = result.series.reduce((sum, s) => sum + s.y.length, 0);
    if (totalPoints === 0) {
      body = (
        <div className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--fg-muted)" }}>
          No ransomware-associated CVEs in this window.
        </div>
      );
    } else {
      body = (
        <ReactECharts
          option={buildChartOption(result, axisLabels)}
          style={{ height: "220px", width: "100%" }}
          notMerge
          lazyUpdate={false}
        />
      );
    }
  }

  return (
    <WidgetCard
      label="Security Intelligence"
      title="KEV-Ransomware Exposure Trend"
      footer={footer}
      axisLabels={axisLabels}
      onAxisLabelsChange={setAxisLabels}
    >
      {body}
    </WidgetCard>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```

Expected: no errors for `KevRansomwareExposure.tsx`.

- [ ] **Step 3: Commit**

```bash
git add sas/web/components/widgets/KevRansomwareExposure.tsx
git commit -m "$(cat <<'EOF'
feat(sas): add KEV-Ransomware Exposure Trend widget (3.2a task 3)

Single alarming-red flowing line filtered to cisa_kev_known_ransomware=true.
Depends on Phase 2.2 hotfix (cisa_kev_known_ransomware on finding_state).
Area fill reinforces urgency. Footer states current ransomware-exposure count.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Widget: Image Inventory Grid (TanStack Table)

**Depends on:** Task 1 complete (clean DB state). No KEV column dependency.

**Files:**
- Create: `sas/web/components/widgets/ImageInventoryGrid.tsx`

Filterable, sortable TanStack Table inside a WidgetCard. Columns: image, repository, critical, high, total open, last seen. Data merged from two sources: `GET /api/entities/Image` (provides image_id, repository, tag, label) and `POST /api/query` (provides `count_open_critical` and `count_open_high` at latest snapshot). No ECharts — no shared chart primitives needed for this widget.

- [ ] **Step 1: Install TanStack Table if not already present**

```bash
cd sas/web && npm list @tanstack/react-table 2>/dev/null || npm install @tanstack/react-table
```

- [ ] **Step 2: Create the component**

File: `sas/web/components/widgets/ImageInventoryGrid.tsx`

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { WidgetCard } from "./WidgetCard";
import { runQuery, getEntities } from "@/lib/api/client";
import type { QueryIn } from "@/lib/api/client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface ImageRow {
  imageId: string;
  label: string;
  repository: string;
  tag: string;
  critical: number;
  high: number;
  totalOpen: number;
  lastSeen: string;
}

// ---------------------------------------------------------------------------
// Query: critical + high counts per image at latest snapshot
// ---------------------------------------------------------------------------
const CRITICAL_QUERY: QueryIn = {
  lens: "Image",
  traversal: [],
  time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
  measure: "count_open_critical",
  filters: [],
  group_by: ["image_id"],
  order_by: null,
  limit: null,
};

const HIGH_QUERY: QueryIn = {
  lens: "Image",
  traversal: [],
  time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
  measure: "count_open_high",
  filters: [],
  group_by: ["image_id"],
  order_by: null,
  limit: null,
};

// ---------------------------------------------------------------------------
// Table skeleton
// ---------------------------------------------------------------------------
function TableSkeleton() {
  return (
    <div className="w-full animate-pulse space-y-1" role="status" aria-label="Loading image inventory…">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          style={{
            height: "var(--h-row)",
            backgroundColor: "var(--bg-surface)",
            borderRadius: "var(--radius)",
          }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Column helper
// ---------------------------------------------------------------------------
const col = createColumnHelper<ImageRow>();

const COLUMNS = [
  col.accessor("label", {
    header: "Image",
    cell: (info) => (
      <span className="font-mono text-[11px]" style={{ color: "var(--fg-primary)" }}>
        {info.getValue()}
      </span>
    ),
    size: 260,
  }),
  col.accessor("repository", {
    header: "Repository",
    cell: (info) => <span className="text-[11px]">{info.getValue()}</span>,
    size: 200,
  }),
  col.accessor("critical", {
    header: "Critical",
    cell: (info) => (
      <span className="font-medium text-[11px]" style={{ color: info.getValue() > 0 ? "var(--severity-critical)" : "var(--fg-muted)" }}>
        {info.getValue()}
      </span>
    ),
    size: 72,
  }),
  col.accessor("high", {
    header: "High",
    cell: (info) => (
      <span className="text-[11px]" style={{ color: info.getValue() > 0 ? "var(--severity-high)" : "var(--fg-muted)" }}>
        {info.getValue()}
      </span>
    ),
    size: 60,
  }),
  col.accessor("totalOpen", {
    header: "Total Open",
    cell: (info) => <span className="text-[11px]">{info.getValue()}</span>,
    size: 80,
  }),
  col.accessor("lastSeen", {
    header: "Last Seen",
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {info.getValue()}
      </span>
    ),
    size: 100,
  }),
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ImageInventoryGrid() {
  const [rows, setRows] = useState<ImageRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [globalFilter, setGlobalFilter] = useState("");
  const [sorting, setSorting] = useState<SortingState>([
    { id: "critical", desc: true },
  ]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const [entities, critResult, highResult] = await Promise.all([
        getEntities("Image") as Promise<Array<{
          id: string; label: string; repository: string; tag: string;
        }>>,
        runQuery(CRITICAL_QUERY),
        runQuery(HIGH_QUERY),
      ]);

      if (cancelled) return;

      // Build maps: image_id → count (last snapshot only)
      const critMap = new Map<string, number>();
      for (const s of critResult.series) {
        const imageId = s.key["image_id"] as string | undefined;
        if (!imageId) continue;
        const last = s.y[s.y.length - 1];
        critMap.set(imageId, typeof last === "number" ? last : 0);
      }

      const highMap = new Map<string, number>();
      for (const s of highResult.series) {
        const imageId = s.key["image_id"] as string | undefined;
        if (!imageId) continue;
        const last = s.y[s.y.length - 1];
        highMap.set(imageId, typeof last === "number" ? last : 0);
      }

      const built: ImageRow[] = entities.map((e) => {
        const critical = critMap.get(e.id) ?? 0;
        const high = highMap.get(e.id) ?? 0;
        return {
          imageId: e.id,
          label: e.label,
          repository: e.repository ?? "",
          tag: e.tag ?? "",
          critical,
          high,
          totalOpen: critical + high,
          lastSeen: new Date().toISOString().slice(0, 10), // TODO Phase 3.3: wire from entity last_seen
        };
      });

      setRows(built);
    }

    load().catch((e: unknown) => {
      if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load data.");
    });

    return () => { cancelled = true; };
  }, []);

  const data = useMemo(() => rows ?? [], [rows]);

  const table = useReactTable({
    data,
    columns: COLUMNS,
    state: { globalFilter, sorting },
    onGlobalFilterChange: setGlobalFilter,
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    globalFilterFn: "includesString",
  });

  let body: React.ReactNode;

  if (error) {
    body = (
      <div className="flex items-center justify-center h-[240px] text-sm"
        style={{ color: "var(--severity-critical)" }} role="alert">
        Unable to load data: {error}
      </div>
    );
  } else if (rows === null) {
    body = <TableSkeleton />;
  } else if (rows.length === 0) {
    body = (
      <div className="flex items-center justify-center h-[240px] text-sm"
        style={{ color: "var(--fg-muted)" }}>
        No images found.
      </div>
    );
  } else {
    body = (
      <div>
        {/* Search input */}
        <input
          type="text"
          placeholder="Filter images…"
          value={globalFilter}
          onChange={(e) => setGlobalFilter(e.target.value)}
          className="w-full mb-2 px-2 py-1 text-[11px] rounded"
          style={{
            border: "1px solid var(--border-subtle)",
            background: "var(--bg-surface)",
            color: "var(--fg-primary)",
            borderRadius: "var(--radius)",
            outline: "none",
          }}
        />

        {/* Table */}
        <div style={{ overflowX: "auto" }}>
          <table className="w-full border-collapse text-[11px]">
            <thead>
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id}>
                  {hg.headers.map((header) => (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="text-left font-medium select-none"
                      style={{
                        color: "var(--fg-muted)",
                        height: "var(--h-row)",
                        paddingRight: 8,
                        borderBottom: "1px solid var(--border-subtle)",
                        cursor: header.column.getCanSort() ? "pointer" : "default",
                        whiteSpace: "nowrap",
                        width: header.getSize(),
                      }}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === "asc" ? " ↑" : header.column.getIsSorted() === "desc" ? " ↓" : ""}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.slice(0, 50).map((row) => (
                <tr
                  key={row.id}
                  style={{ borderBottom: "1px solid var(--border-subtle)" }}
                  className="hover:bg-[var(--bg-surface)] transition-colors"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      style={{ height: "var(--h-row)", paddingRight: 8 }}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rows.length > 50 && (
          <p className="text-[10px] mt-1" style={{ color: "var(--fg-muted)" }}>
            Showing 50 of {rows.length.toLocaleString("en-GB")} images.
          </p>
        )}
      </div>
    );
  }

  return (
    <WidgetCard label="Asset Inventory" title="Image Inventory">
      {body}
    </WidgetCard>
  );
}
```

- [ ] **Step 3: Add getEntities type to client.ts**

Open `sas/web/lib/api/client.ts`. The `getEntities` function already exists. Ensure the return type is `Promise<unknown[]>` (it is). No change required — the widget casts the result inline.

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```

Expected: no errors for `ImageInventoryGrid.tsx`.

- [ ] **Step 5: Commit**

```bash
git add sas/web/components/widgets/ImageInventoryGrid.tsx
git commit -m "$(cat <<'EOF'
feat(sas): add Image Inventory Grid widget (3.2a task 4)

TanStack Table showing images ranked by critical count. Global search input,
column sort, severity colour-coding. Merges GET /api/entities/Image with
POST /api/query (count_open_critical + count_open_high at latest snapshot).
Paginates to 50 visible rows. Phase 3.3 will wire row-click drill-in.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Widget: Repository Tag Hygiene (horizontal bars)

**Files:**
- Create: `sas/web/components/widgets/RepositoryTagHygiene.tsx`

Horizontal bar chart. Each bar = one repository, length = `count_open_critical` at latest snapshot. Sorted descending, top 15 repositories. ECharts horizontal bar: `series[0].type: "bar"`, `yAxis.type: "category"`. Colour: `CHART_COLORS.severityCritical`.

- [ ] **Step 1: Create the component**

File: `sas/web/components/widgets/RepositoryTagHygiene.tsx`

```tsx
"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { WidgetCard } from "./WidgetCard";
import { runQuery } from "@/lib/api/client";
import type { QueryIn, QueryResult } from "@/lib/api/client";
import {
  CHART_COLORS,
  STANDARD_TOOLTIP_STYLE,
} from "@/lib/charts/defaults";

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// ---------------------------------------------------------------------------
// Query: critical count per repository at latest snapshot
// ---------------------------------------------------------------------------
const REPO_HYGIENE_QUERY: QueryIn = {
  lens: "Repository",
  traversal: [],
  time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
  measure: "count_open_critical",
  filters: [],
  group_by: ["repository"],
  order_by: { field: "count_open_critical", direction: "desc" },
  limit: 15,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function ChartSkeleton() {
  return (
    <div
      className="w-full h-[280px] animate-pulse"
      style={{ backgroundColor: "var(--bg-surface)", borderRadius: "var(--radius)" }}
      aria-label="Loading chart data…"
      role="status"
    />
  );
}

function buildChartOption(result: QueryResult): object {
  // Each series has key: { repository: "..." }, y: [count] (single snapshot)
  type RepoEntry = { repository: string; count: number };
  const entries: RepoEntry[] = result.series
    .map((s) => ({
      repository: (s.key["repository"] as string) ?? "unknown",
      count: typeof s.y[0] === "number" ? (s.y[0] as number) : 0,
    }))
    .filter((e) => e.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, 15);

  // Horizontal bar: categories on y-axis, values on x-axis
  const categories = entries.map((e) => {
    // Truncate long repository names for display
    const r = e.repository;
    return r.length > 40 ? `…${r.slice(-37)}` : r;
  });
  const values = entries.map((e) => e.count);

  const chartHeight = Math.max(200, entries.length * 28);

  return {
    backgroundColor: "transparent",
    grid: { top: 8, right: 48, bottom: 8, left: 8, containLabel: true },
    xAxis: {
      type: "value" as const,
      minInterval: 1,
      axisLabel: { fontSize: 10, color: CHART_COLORS.greyMuted },
      splitLine: {
        lineStyle: { color: CHART_COLORS.greyBorder, type: "dashed" as const },
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    yAxis: {
      type: "category" as const,
      data: categories,
      axisLabel: {
        fontSize: 10,
        color: CHART_COLORS.greyMuted,
        width: 160,
        overflow: "truncate" as const,
      },
      axisLine: { lineStyle: { color: CHART_COLORS.greyBorder } },
      axisTick: { show: false },
    },
    series: [
      {
        type: "bar" as const,
        data: values,
        itemStyle: {
          color: CHART_COLORS.severityCritical,
          borderRadius: [0, 4, 4, 0],
        },
        barMaxWidth: 20,
        label: {
          show: true,
          position: "right" as const,
          fontSize: 10,
          color: CHART_COLORS.greyMuted,
          formatter: (p: { value: number }) => p.value.toLocaleString("en-GB"),
        },
      },
    ],
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "none" as const },
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const p = (params as Array<{ name: string; value: number }>)[0];
        if (!p) return "";
        return `<div style="font-size:11px">
          <div style="color:${CHART_COLORS.greyMuted};margin-bottom:2px">${p.name}</div>
          <div><b style="color:${CHART_COLORS.severityCritical}">${p.value.toLocaleString("en-GB")}</b> critical open</div>
        </div>`;
      },
    },
    _chartHeight: chartHeight, // passed to style below, not part of ECharts spec
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function RepositoryTagHygiene() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    runQuery(REPO_HYGIENE_QUERY)
      .then((r) => { if (!cancelled) setResult(r); })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load data.");
      });
    return () => { cancelled = true; };
  }, []);

  const footer = result
    ? (() => {
        const withCritical = result.series.filter(
          (s) => typeof s.y[0] === "number" && (s.y[0] as number) > 0
        ).length;
        if (withCritical === 0) return "No critical findings across repositories.";
        return `${withCritical.toLocaleString("en-GB")} ${withCritical === 1 ? "repository has" : "repositories have"} critical open findings.`;
      })()
    : undefined;

  let body: React.ReactNode;
  if (error) {
    body = (
      <div className="flex items-center justify-center h-[280px] text-sm"
        style={{ color: "var(--severity-critical)" }} role="alert">
        Unable to load data: {error}
      </div>
    );
  } else if (result === null) {
    body = <ChartSkeleton />;
  } else {
    const hasData = result.series.some(
      (s) => typeof s.y[0] === "number" && (s.y[0] as number) > 0
    );
    if (!hasData) {
      body = (
        <div className="flex items-center justify-center h-[280px] text-sm"
          style={{ color: "var(--fg-muted)" }}>
          No critical findings in any repository.
        </div>
      );
    } else {
      const option = buildChartOption(result);
      const { _chartHeight, ...echartOption } = option as typeof option & { _chartHeight: number };
      const h = Math.min(320, Math.max(200, _chartHeight));
      body = (
        <ReactECharts
          option={echartOption}
          style={{ height: `${h}px`, width: "100%" }}
          notMerge
          lazyUpdate={false}
        />
      );
    }
  }

  return (
    <WidgetCard label="Asset Health" title="Repository Tag Hygiene" footer={footer}>
      {body}
    </WidgetCard>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```

Expected: no errors for `RepositoryTagHygiene.tsx`.

- [ ] **Step 3: Commit**

```bash
git add sas/web/components/widgets/RepositoryTagHygiene.tsx
git commit -m "$(cat <<'EOF'
feat(sas): add Repository Tag Hygiene horizontal bar chart (3.2a task 5)

Top-15 repositories ranked by critical open findings. Horizontal ECharts bar
chart in severity-critical red. Inline count labels. Dynamic chart height
scales with visible repository count. Footer summarises how many repos have
critical findings.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Widget: Findings Table with new /api/findings endpoint

This is the most complex widget in Phase 3.2a. It requires a new FastAPI endpoint for raw finding_state queries, an update to the API client, and the TanStack Table component.

**Files:**
- Create: `sas/api/routes/findings.py`
- Modify: `sas/api/main.py` (register the new router)
- Modify: `sas/web/lib/api/client.ts` (add `getFindingRows()`)
- Create: `sas/web/components/widgets/FindingsTable.tsx`

- [ ] **Step 1: Create the findings API route**

File: `sas/api/routes/findings.py`

```python
"""GET /api/findings — raw paginated finding_state rows for the Findings Table widget."""

from fastapi import APIRouter, Depends, Query as QParam
from pydantic import BaseModel

from sas.api.deps import get_db

router = APIRouter()


class FindingRow(BaseModel):
    finding_id: int
    cve_id: str
    severity: str
    image_id: str
    package_name: str
    package_version: str
    state: str
    reason_code: str | None
    first_seen: str
    last_seen: str
    days_open: int | None
    fix_available: bool | None
    public_exploit: bool | None
    cisa_kev_known_ransomware: bool | None
    in_use: bool | None


class FindingsResponse(BaseModel):
    rows: list[FindingRow]
    total: int
    page: int
    page_size: int


@router.get("/findings", tags=["findings"], response_model=FindingsResponse)
def get_findings(
    severity: str | None = QParam(default=None, description="Filter by severity: Critical, High, Medium, Low"),
    state: str | None = QParam(default="OPEN", description="Filter by state: OPEN, CLOSED, or ALL"),
    kev_only: bool = QParam(default=False, description="Return only KEV ransomware-associated findings"),
    has_fix: bool | None = QParam(default=None, description="True = fix available only, False = no fix only"),
    page: int = QParam(default=1, ge=1),
    page_size: int = QParam(default=100, ge=1, le=500),
    conn=Depends(get_db),
) -> FindingsResponse:
    """Return paginated raw finding_state rows with optional filters."""
    conditions: list[str] = []
    params: list = []

    if state and state.upper() != "ALL":
        conditions.append("state = ?")
        params.append(state.upper())

    if severity:
        conditions.append("severity = ?")
        params.append(severity)

    if kev_only:
        conditions.append("cisa_kev_known_ransomware = TRUE")

    if has_fix is True:
        conditions.append("fix_available = TRUE")
    elif has_fix is False:
        conditions.append("fix_available = FALSE")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    count_row = conn.execute(
        f"SELECT COUNT(*) FROM finding_state {where}", params
    ).fetchone()
    total = count_row[0] if count_row else 0

    offset = (page - 1) * page_size
    rows = conn.execute(
        f"""
        SELECT finding_id, cve_id, severity, image_id, package_name, package_version,
               state, reason_code, first_seen, last_seen, days_open,
               fix_available, public_exploit, cisa_kev_known_ransomware, in_use
        FROM finding_state
        {where}
        ORDER BY
          CASE severity
            WHEN 'Critical' THEN 1
            WHEN 'High' THEN 2
            WHEN 'Medium' THEN 3
            WHEN 'Low' THEN 4
            ELSE 5
          END,
          last_seen DESC
        LIMIT ? OFFSET ?
        """,
        params + [page_size, offset],
    ).fetchall()

    col_names = [
        "finding_id", "cve_id", "severity", "image_id", "package_name", "package_version",
        "state", "reason_code", "first_seen", "last_seen", "days_open",
        "fix_available", "public_exploit", "cisa_kev_known_ransomware", "in_use",
    ]

    finding_rows = [
        FindingRow(**{
            k: (str(v) if k in ("first_seen", "last_seen") and v is not None else v)
            for k, v in zip(col_names, row)
        })
        for row in rows
    ]

    return FindingsResponse(
        rows=finding_rows,
        total=total,
        page=page,
        page_size=page_size,
    )
```

- [ ] **Step 2: Register the findings router in main.py**

Open `sas/api/main.py`. Find where other routers are registered (e.g. `app.include_router(query.router, prefix="/api")`). Add:

```python
from sas.api.routes import findings  # add to existing imports

app.include_router(findings.router, prefix="/api")
```

- [ ] **Step 3: Add getFindingRows to client.ts**

Open `sas/web/lib/api/client.ts`. Add these types and function after the existing exports:

```typescript
// ---------------------------------------------------------------------------
// Findings endpoint — raw finding_state rows for FindingsTable widget
// ---------------------------------------------------------------------------

export interface FindingRow {
  finding_id: number;
  cve_id: string;
  severity: string;
  image_id: string;
  package_name: string;
  package_version: string;
  state: string;
  reason_code: string | null;
  first_seen: string;
  last_seen: string;
  days_open: number | null;
  fix_available: boolean | null;
  public_exploit: boolean | null;
  cisa_kev_known_ransomware: boolean | null;
  in_use: boolean | null;
}

export interface FindingsResponse {
  rows: FindingRow[];
  total: number;
  page: number;
  page_size: number;
}

export interface FindingsFilter {
  severity?: string;
  state?: string;  // "OPEN" | "CLOSED" | "ALL"
  kev_only?: boolean;
  has_fix?: boolean;
  page?: number;
  page_size?: number;
}

export async function getFindingRows(filter: FindingsFilter = {}): Promise<FindingsResponse> {
  const params = new URLSearchParams();
  if (filter.severity) params.set("severity", filter.severity);
  if (filter.state !== undefined) params.set("state", filter.state);
  if (filter.kev_only) params.set("kev_only", "true");
  if (filter.has_fix !== undefined) params.set("has_fix", String(filter.has_fix));
  if (filter.page !== undefined) params.set("page", String(filter.page));
  if (filter.page_size !== undefined) params.set("page_size", String(filter.page_size));
  const qs = params.toString();
  return apiFetch<FindingsResponse>(`/api/findings${qs ? "?" + qs : ""}`);
}
```

- [ ] **Step 4: Create the Findings Table component**

File: `sas/web/components/widgets/FindingsTable.tsx`

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { WidgetCard } from "./WidgetCard";
import { getFindingRows } from "@/lib/api/client";
import type { FindingRow } from "@/lib/api/client";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function ChartSkeleton() {
  return (
    <div className="w-full animate-pulse space-y-1" role="status" aria-label="Loading findings…">
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          style={{
            height: "var(--h-row)",
            backgroundColor: "var(--bg-surface)",
            borderRadius: "var(--radius)",
          }}
        />
      ))}
    </div>
  );
}

const SEVERITY_COLOUR: Record<string, string> = {
  Critical: "var(--severity-critical)",
  High: "var(--severity-high)",
  Medium: "var(--severity-medium)",
  Low: "var(--severity-low)",
};

// ---------------------------------------------------------------------------
// Column helper
// ---------------------------------------------------------------------------
const col = createColumnHelper<FindingRow>();

const COLUMNS = [
  col.accessor("cve_id", {
    header: "CVE",
    cell: (info) => (
      <span className="font-mono text-[11px] font-medium" style={{ color: "var(--fg-primary)" }}>
        {info.getValue()}
      </span>
    ),
    size: 160,
  }),
  col.accessor("severity", {
    header: "Severity",
    cell: (info) => {
      const v = info.getValue();
      return (
        <span
          className="text-[10px] font-medium uppercase tracking-wider px-1.5 py-0.5 rounded"
          style={{
            color: SEVERITY_COLOUR[v] ?? "var(--fg-muted)",
            backgroundColor: `${SEVERITY_COLOUR[v] ?? "#888"}1a`,
            borderRadius: "4px",
          }}
        >
          {v}
        </span>
      );
    },
    size: 80,
  }),
  col.accessor("package_name", {
    header: "Package",
    cell: (info) => <span className="font-mono text-[11px]">{info.getValue()}</span>,
    size: 140,
  }),
  col.accessor("package_version", {
    header: "Version",
    cell: (info) => <span className="font-mono text-[11px]">{info.getValue()}</span>,
    size: 100,
  }),
  col.accessor("state", {
    header: "State",
    cell: (info) => (
      <span className="text-[11px]" style={{ color: info.getValue() === "OPEN" ? "var(--severity-critical)" : "var(--fg-muted)" }}>
        {info.getValue()}
      </span>
    ),
    size: 64,
  }),
  col.accessor("days_open", {
    header: "Days Open",
    cell: (info) => {
      const v = info.getValue();
      return <span className="text-[11px]">{v != null ? v : "—"}</span>;
    },
    size: 80,
  }),
  col.accessor("fix_available", {
    header: "Fix",
    cell: (info) => {
      const v = info.getValue();
      return (
        <span className="text-[11px]" style={{ color: v ? "var(--fg-primary)" : "var(--fg-muted)" }}>
          {v == null ? "—" : v ? "Yes" : "No"}
        </span>
      );
    },
    size: 48,
  }),
  col.accessor("reason_code", {
    header: "Reason",
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {info.getValue() ?? "—"}
      </span>
    ),
    size: 96,
  }),
];

// ---------------------------------------------------------------------------
// Severity filter bar
// ---------------------------------------------------------------------------
const SEVERITIES = ["All", "Critical", "High", "Medium", "Low"] as const;
type SeverityFilter = typeof SEVERITIES[number];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function FindingsTable() {
  const [rows, setRows] = useState<FindingRow[] | null>(null);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>("All");
  const [globalFilter, setGlobalFilter] = useState("");
  const [sorting, setSorting] = useState<SortingState>([
    { id: "severity", desc: false },
  ]);

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    getFindingRows({
      severity: severityFilter === "All" ? undefined : severityFilter,
      state: "OPEN",
      page_size: 200,
    })
      .then((resp) => {
        if (!cancelled) {
          setRows(resp.rows);
          setTotal(resp.total);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load data.");
      });
    return () => { cancelled = true; };
  }, [severityFilter]);

  const data = useMemo(() => rows ?? [], [rows]);

  const table = useReactTable({
    data,
    columns: COLUMNS,
    state: { globalFilter, sorting },
    onGlobalFilterChange: setGlobalFilter,
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    globalFilterFn: "includesString",
  });

  const footer = rows != null
    ? `Showing ${table.getFilteredRowModel().rows.length.toLocaleString("en-GB")} of ${total.toLocaleString("en-GB")} open findings.`
    : undefined;

  let body: React.ReactNode;

  if (error) {
    body = (
      <div className="flex items-center justify-center h-[300px] text-sm"
        style={{ color: "var(--severity-critical)" }} role="alert">
        Unable to load data: {error}
      </div>
    );
  } else if (rows === null) {
    body = <ChartSkeleton />;
  } else {
    body = (
      <div>
        {/* Severity filter pills + search row */}
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          <div className="flex gap-1">
            {SEVERITIES.map((s) => (
              <button
                key={s}
                onClick={() => setSeverityFilter(s)}
                className="text-[10px] px-2 py-0.5 rounded-full transition-colors"
                style={{
                  border: "1px solid",
                  borderColor: severityFilter === s ? "var(--border-strong)" : "var(--border-subtle)",
                  background: severityFilter === s ? "var(--bg-surface)" : "transparent",
                  color: severityFilter === s ? "var(--fg-primary)" : "var(--fg-muted)",
                  borderRadius: "999px",
                }}
              >
                {s}
              </button>
            ))}
          </div>
          <input
            type="text"
            placeholder="Search CVE, package…"
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="flex-1 min-w-[160px] px-2 py-1 text-[11px]"
            style={{
              border: "1px solid var(--border-subtle)",
              background: "var(--bg-surface)",
              color: "var(--fg-primary)",
              borderRadius: "var(--radius)",
              outline: "none",
            }}
          />
        </div>

        {/* Table */}
        <div style={{ overflowX: "auto", maxHeight: "360px", overflowY: "auto" }}>
          <table className="w-full border-collapse text-[11px]">
            <thead style={{ position: "sticky", top: 0, background: "var(--bg-base)", zIndex: 1 }}>
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id}>
                  {hg.headers.map((header) => (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="text-left font-medium select-none"
                      style={{
                        color: "var(--fg-muted)",
                        height: "var(--h-row)",
                        paddingRight: 8,
                        borderBottom: "1px solid var(--border-subtle)",
                        cursor: header.column.getCanSort() ? "pointer" : "default",
                        whiteSpace: "nowrap",
                        width: header.getSize(),
                      }}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === "asc" ? " ↑" : header.column.getIsSorted() === "desc" ? " ↓" : ""}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  style={{ borderBottom: "1px solid var(--border-subtle)" }}
                  className="hover:bg-[var(--bg-surface)] transition-colors"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      style={{ height: "var(--h-row)", paddingRight: 8 }}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <WidgetCard label="Findings" title="Open Findings" footer={footer}>
      {body}
    </WidgetCard>
  );
}
```

- [ ] **Step 5: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```

Expected: no errors for `FindingsTable.tsx` or `client.ts`.

- [ ] **Step 6: Smoke test the new API endpoint**

Ensure the API server is running, then:

```bash
# All open findings (first page)
curl -s "http://localhost:8000/api/findings?state=OPEN&page_size=5" | python -m json.tool

# Critical findings only
curl -s "http://localhost:8000/api/findings?severity=Critical&page_size=5" | python -m json.tool

# KEV-only findings
curl -s "http://localhost:8000/api/findings?kev_only=true&page_size=5" | python -m json.tool
```

Expected: JSON with `rows`, `total`, `page`, `page_size` fields. No HTTP 500.

- [ ] **Step 7: Commit**

```bash
git add sas/api/routes/findings.py sas/api/main.py \
        sas/web/lib/api/client.ts \
        sas/web/components/widgets/FindingsTable.tsx
git commit -m "$(cat <<'EOF'
feat(sas): add Findings Table widget and /api/findings endpoint (3.2a task 6)

New GET /api/findings route: paginated finding_state rows with optional
severity, state, kev_only, and has_fix filters. TanStack Table in WidgetCard
with severity filter pills, global search, sticky headers, sort on all
columns. Phase 3.3 will wire row-click drill-in.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Update dashboard page to show all 6 widgets

**Depends on:** Tasks 2–6 all committed.

**Files:**
- Modify: `sas/web/app/dashboard/page.tsx`

Replace the existing two-column placeholder layout with a 4-row 12-column grid. Table widgets span 12 columns (full width). Chart widgets span 6 columns each.

- [ ] **Step 1: Update dashboard/page.tsx**

Replace the entire file with:

```tsx
import { AppShell } from "@/components/app-shell/AppShell";
import { FleetCriticalTrend } from "@/components/widgets/FleetCriticalTrend";
import { NewFixedRegressed } from "@/components/widgets/NewFixedRegressed";
import { KevRansomwareExposure } from "@/components/widgets/KevRansomwareExposure";
import { RepositoryTagHygiene } from "@/components/widgets/RepositoryTagHygiene";
import { ImageInventoryGrid } from "@/components/widgets/ImageInventoryGrid";
import { FindingsTable } from "@/components/widgets/FindingsTable";

export default function DashboardPage() {
  return (
    <AppShell pageTitle="Dashboard">
      {/* 12-column CSS grid — widgets span 4, 6, or 12 columns */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(12, 1fr)",
          gap: "var(--gap-widget)",
        }}
      >
        {/* Row 1: time-series trend widgets */}
        <div style={{ gridColumn: "span 6" }}>
          <FleetCriticalTrend />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <NewFixedRegressed />
        </div>

        {/* Row 2: KEV exposure + repository hygiene */}
        <div style={{ gridColumn: "span 6" }}>
          <KevRansomwareExposure />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <RepositoryTagHygiene />
        </div>

        {/* Row 3: Image Inventory Grid — full width */}
        <div style={{ gridColumn: "span 12" }}>
          <ImageInventoryGrid />
        </div>

        {/* Row 4: Findings Table — full width */}
        <div style={{ gridColumn: "span 12" }}>
          <FindingsTable />
        </div>
      </div>
    </AppShell>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Start dev server and verify page loads**

```bash
cd sas/web && npm run dev &
```

Open `http://localhost:3000/dashboard`. Expected: 4 rows of widgets visible. No console errors. All 6 widgets render (may show skeletons briefly, then real data or empty-state messages).

Stop dev server with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add sas/web/app/dashboard/page.tsx
git commit -m "$(cat <<'EOF'
feat(sas): populate dashboard with all 6 Phase 3.2a widgets (task 7)

Four-row 12-column grid: Fleet Critical Trend + New/Fixed/Regressed on row 1,
KEV-Ransomware + Repository Tag Hygiene on row 2, Image Inventory Grid
(full width) on row 3, Findings Table (full width) on row 4.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — Full smoke test

**Opus review gate: Opus reviews this task's output before Phase 3.2a is marked complete.**

**Depends on:** Tasks 1–7 all committed and dev server running.

This task has no file changes. It verifies the complete Phase 3.2a implementation end-to-end.

- [ ] **Step 1: Ensure backend and frontend are running**

```bash
# Terminal 1 — FastAPI backend
.venv/bin/python -m sas.api.run

# Terminal 2 — Next.js dev server
cd sas/web && npm run dev
```

- [ ] **Step 2: Verify /api/findings endpoint**

```bash
# Basic paginated response
curl -s "http://localhost:8000/api/findings" | python -m json.tool | head -30

# Critical only
curl -s "http://localhost:8000/api/findings?severity=Critical&page_size=10" | python -m json.tool | grep '"total"'

# KEV filter (tests Task 1 hotfix end-to-end)
curl -s "http://localhost:8000/api/findings?kev_only=true" | python -m json.tool | grep '"total"'
```

Expected: all three return valid JSON with `rows`, `total`, `page`, `page_size`. No 500 errors.

- [ ] **Step 3: Verify POST /api/query for each widget's query**

```bash
# NewFixedRegressed — count_new
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"lens":"Image","traversal":[],"time":{"mode":"last_n_snapshots","n":90,"granularity":"day"},"measure":"count_new","filters":[],"group_by":[],"order_by":null,"limit":null}' \
  | python -m json.tool | grep '"exec_time_ms"'

# KevRansomwareExposure — count_open filtered by kev flag
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"lens":"Image","traversal":[],"time":{"mode":"last_n_snapshots","n":90,"granularity":"day"},"measure":"count_open","filters":[{"field":"cisa_kev_known_ransomware","operator":"eq","value":true}],"group_by":[],"order_by":null,"limit":null}' \
  | python -m json.tool | grep '"exec_time_ms"'

# RepositoryTagHygiene — count_open_critical per repository
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"lens":"Repository","traversal":[],"time":{"mode":"last_n_snapshots","n":1,"granularity":"day"},"measure":"count_open_critical","filters":[],"group_by":["repository"],"order_by":{"field":"count_open_critical","direction":"desc"},"limit":15}' \
  | python -m json.tool | grep '"exec_time_ms"'
```

Expected: all three return `exec_time_ms` field. No HTTP 500.

- [ ] **Step 4: Run Python unit tests**

```bash
python -m pytest tests/scenarios/kev_ransomware/ -v
python -m pytest tests/ -v --ignore=tests/scenarios/kev_ransomware/
```

Expected: all existing tests pass, plus the 4 new kev_ransomware scenario tests pass.

- [ ] **Step 5: Run TypeScript type check**

```bash
cd sas/web && npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 6: Run Next.js build**

```bash
cd sas/web && npm run build 2>&1 | tail -20
```

Expected: `✓ Compiled` (or equivalent success message). Zero errors. Warnings about `useEffect` dependencies are acceptable but note them.

- [ ] **Step 7: Visual check in browser**

Open `http://localhost:3000/dashboard`. Verify:
- All 6 widget cards visible without layout overlap.
- Fleet Critical Trend and New vs Fixed vs Regressed on row 1 (equal width).
- KEV-Ransomware Exposure and Repository Tag Hygiene on row 2 (equal width).
- Image Inventory Grid spans full width row 3.
- Findings Table spans full width row 4.
- Severity filter pills on Findings Table function (clicking "Critical" re-fetches).
- Search input on Findings Table filters visible rows.
- Search input on Image Inventory Grid filters visible rows.
- Dark mode toggle (if tested) does not break chart colours (ECharts uses CHART_COLORS hex, not CSS vars — expected to stay constant).
- No JavaScript console errors.

- [ ] **Step 8: Record any deviations**

If any widget shows an error state (red "Unable to load data") against the live lab DB:
1. Note which widget and the error message.
2. Check the backend log for the specific SQL error.
3. Do not block the phase completion on empty-data (the lab DB may have no KEV ransomware findings — that renders the empty-state message, not an error).

- [ ] **Step 9: Final commit (smoke test notes)**

```bash
git commit --allow-empty -m "$(cat <<'EOF'
chore(sas): phase 3.2a smoke test complete

All 6 widgets render on /dashboard. GET /api/findings passes. POST /api/query
for all new widget queries passes. 4 KEV scenario tests pass. TypeScript
type-check clean. Build succeeds.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Summary checklist

| Task | Deliverable | Gate |
|---|---|---|
| 1 | `cisa_kev_known_ransomware` on `finding_state`, 4 scenario tests | Opus review |
| 2 | `NewFixedRegressed.tsx` — 3-series combo chart | tsc --noEmit |
| 3 | `KevRansomwareExposure.tsx` — single alarming-red line | tsc --noEmit |
| 4 | `ImageInventoryGrid.tsx` — TanStack Table, search + sort | tsc --noEmit |
| 5 | `RepositoryTagHygiene.tsx` — horizontal bar chart | tsc --noEmit |
| 6 | `FindingsTable.tsx` + `GET /api/findings` endpoint | tsc --noEmit + curl |
| 7 | `dashboard/page.tsx` updated with 4-row grid | npm run dev visual |
| 8 | End-to-end smoke test — all queries, all widgets, build | Opus review |

**Total budget estimate:** ~$15–20. Tasks 2–6 parallelise across up to 5 Sonnet workers after Task 1 completes.
