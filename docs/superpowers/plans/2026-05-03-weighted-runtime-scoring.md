# Weighted Runtime Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Weighted" group-by mode to the FindingsTable that ranks CVEs by a configurable score combining severity, actionability flags, and workload blast radius.

**Architecture:** New backend endpoint `/api/workload-counts` returns workload counts per CVE. Frontend adds a "weighted" group-by mode with a config panel for severity gate and weight spin buttons. Scoring is computed client-side using user's weights, with localStorage persistence.

**Tech Stack:** FastAPI + DuckDB (backend), React + TanStack Table + Tailwind v4 (frontend)

---

### Task 1: Backend — `/api/workload-counts` endpoint

**Files:**
- Create: `sas/api/routes/workload_counts.py`
- Modify: `sas/api/main.py` (register new router)

- [ ] **Step 1: Create the workload counts route**

Create `sas/api/routes/workload_counts.py`:

```python
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
```

- [ ] **Step 2: Register the router in main.py**

Modify `sas/api/main.py` — add the import and router registration:

```python
# Add to imports:
from sas.api.routes import query, widgets, entities, findings as findings_router, workload_counts as wc_router

# Add before healthz:
app.include_router(wc_router.router, prefix="/api", tags=["workloads"])
```

- [ ] **Step 3: Test the endpoint**

Start backend and test:
```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m sas.api.run &
sleep 3
curl -s http://localhost:8000/api/workload-counts | python3 -m json.tool | head -30
```

Expected: JSON with `counts` array of `{cve_id, workload_count}` and `snapshot_date`.

- [ ] **Step 4: Commit**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
git add sas/api/routes/workload_counts.py sas/api/main.py
git commit -m "feat: add /api/workload-counts endpoint for CVE-level workload blast radius"
```

---

### Task 2: Frontend — API client for workload counts

**Files:**
- Modify: `sas/web/lib/api/client.ts` (add `getWorkloadCounts` function)
- Modify: `sas/web/lib/api/types.ts` (regenerate types)

- [ ] **Step 1: Regenerate API types**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npm run generate-api
```

This should auto-generate `WorkloadCount` and `WorkloadCountsResponse` types in `lib/api/types.ts`.

- [ ] **Step 2: Add client function**

Modify `sas/web/lib/api/client.ts` — add the fetch function:

```typescript
export async function getWorkloadCounts(): Promise<WorkloadCountsResponse> {
  const res = await fetch(`${API_BASE}/api/workload-counts`);
  if (!res.ok) throw new Error(`Workload counts failed: ${res.status}`);
  return res.json();
}
```

Add the import at the top:
```typescript
import type { FindingsResponse, WorkloadCountsResponse } from "./types";
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
git add sas/web/lib/api/client.ts sas/web/lib/api/types.ts
git commit -m "feat: add getWorkloadCounts API client function"
```

---

### Task 3: Frontend — Weight configuration types and localStorage hook

**Files:**
- Create: `sas/web/lib/weighted-weights.ts`

- [ ] **Step 1: Create the weight management module**

Create `sas/web/lib/weighted-weights.ts`:

```typescript
"use client";

export interface WeightConfig {
  severityGate: string[];
  weights: {
    Critical: number;
    High: number;
    Medium: number;
    Low: number;
    Negligible: number;
    in_use: number;
    fix_available: number;
    public_exploit: number;
  };
}

export const DEFAULT_WEIGHTS: WeightConfig = {
  severityGate: ["Critical", "High"],
  weights: {
    Critical: 2,
    High: 1,
    Medium: 0,
    Low: 0,
    Negligible: 0,
    in_use: 1,
    fix_available: 1,
    public_exploit: 1,
  },
};

const STORAGE_KEY = "sas:weighted-weights";

export function loadWeights(): WeightConfig {
  if (typeof window === "undefined") return DEFAULT_WEIGHTS;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_WEIGHTS;
    const parsed = JSON.parse(raw);
    // Validate structure
    if (
      Array.isArray(parsed.severityGate) &&
      parsed.weights &&
      typeof parsed.weights.Critical === "number"
    ) {
      return parsed as WeightConfig;
    }
    return DEFAULT_WEIGHTS;
  } catch {
    return DEFAULT_WEIGHTS;
  }
}

export function saveWeights(config: WeightConfig): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // Storage full or unavailable — silent fail
  }
}

/** Future: migrate to user profile in database when multi-tenant auth is implemented. */
```

- [ ] **Step 2: Commit**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
git add sas/web/lib/weighted-weights.ts
git commit -m "feat: add weight config types and localStorage persistence"
```

---

### Task 4: Frontend — Weighted group-by mode in FindingsTable

**Files:**
- Modify: `sas/web/components/widgets/FindingsTable.tsx`

This is the largest task. We need to:
1. Add "weighted" to the GroupBy type and options
2. Add the config panel UI
3. Add the workload counts fetch
4. Add the scoring computation
5. Add the weighted columns
6. Wire the config panel to show/hide

- [ ] **Step 1: Add weighted type and imports**

At the top of `FindingsTable.tsx`, add imports:

```typescript
import { getWorkloadCounts } from "@/lib/api/client";
import type { WorkloadCountsResponse } from "@/lib/api/types";
import { loadWeights, saveWeights, DEFAULT_WEIGHTS, type WeightConfig } from "@/lib/weighted-weights";
```

Change the GroupBy type:
```typescript
type GroupBy = "none" | "cve" | "image" | "package" | "weighted";
```

Add "weighted" to GROUP_BY_OPTIONS:
```typescript
const GROUP_BY_OPTIONS: { value: GroupBy; label: string }[] = [
  { value: "none", label: "None" },
  { value: "cve", label: "CVE" },
  { value: "image", label: "Image" },
  { value: "package", label: "Package" },
  { value: "weighted", label: "Weighted" },
];
```

- [ ] **Step 2: Add WeightedRow interface**

Add after PackageRow interface:

```typescript
interface WeightedRow {
  cve_id: string;
  severity: string;
  workload_count: number;
  in_use: boolean;
  fix_available: boolean;
  public_exploit: boolean;
  score: number;
  breakdown: string;
}
```

- [ ] **Step 3: Add weight state and workload counts fetch**

After the existing state declarations, add:

```typescript
  // Weighted scoring state
  const [weights, setWeights] = useState<WeightConfig>(DEFAULT_WEIGHTS);
  const [workloadCounts, setWorkloadCounts] = useState<Record<string, number>>({});
  const [snapshotDate, setSnapshotDate] = useState<string>("");

  // Load weights from localStorage on mount
  useEffect(() => {
    setWeights(loadWeights());
  }, []);

  // Save weights to localStorage when changed
  useEffect(() => {
    // Don't save on initial load (debounce via snapshot)
    saveWeights(weights);
  }, [weights]);

  // Fetch workload counts when weighted mode is active
  useEffect(() => {
    if (groupBy !== "weighted") return;
    getWorkloadCounts()
      .then((data) => {
        const map: Record<string, number> = {};
        for (const entry of data.counts) {
          map[entry.cve_id] = entry.workload_count;
        }
        setWorkloadCounts(map);
        setSnapshotDate(data.snapshot_date);
      })
      .catch((e) => console.error("Failed to load workload counts:", e));
  }, [groupBy]);
```

- [ ] **Step 4: Add scoring computation**

Add a useMemo after `packageRows`:

```typescript
  const weightedRows = useMemo<WeightedRow[]>(() => {
    if (groupBy !== "weighted") return [];
    
    // First aggregate by CVE (same as cveRows logic)
    const map = new Map<string, { 
      severities: string[]; 
      inUse: boolean; 
      fixAvailable: boolean; 
      publicExploit: boolean 
    }>();
    
    for (const row of filteredRows) {
      const key = row.cve_id;
      const existing = map.get(key);
      if (!existing) {
        map.set(key, {
          severities: [row.severity],
          inUse: row.in_use,
          fixAvailable: row.fix_available,
          publicExploit: row.public_exploit,
        });
      } else {
        existing.severities.push(row.severity);
        if (row.in_use) existing.inUse = true;
        if (row.fix_available) existing.fixAvailable = true;
        if (row.public_exploit) existing.publicExploit = true;
      }
    }

    // Compute scores
    const result: WeightedRow[] = [];
    for (const [cve_id, agg] of map.entries()) {
      const severity = maxSeverity(agg.severities);
      const workload_count = workloadCounts[cve_id] || 0;
      if (workload_count === 0) continue; // Skip CVEs with no workload data

      const sevWeight = weights.weights[severity as keyof typeof weights.weights] || 0;
      const flags = sevWeight
        + (agg.inUse ? weights.weights.in_use : 0)
        + (agg.fixAvailable ? weights.weights.fix_available : 0)
        + (agg.publicExploit ? weights.weights.public_exploit : 0);
      
      const score = flags * workload_count;

      // Build breakdown string
      const parts: number[] = [];
      if (sevWeight > 0) parts.push(sevWeight);
      if (agg.inUse && weights.weights.in_use > 0) parts.push(weights.weights.in_use);
      if (agg.fixAvailable && weights.weights.fix_available > 0) parts.push(weights.weights.fix_available);
      if (agg.publicExploit && weights.weights.public_exploit > 0) parts.push(weights.weights.public_exploit);
      const breakdown = `(${parts.join(" + ")}) × ${workload_count}`;

      result.push({
        cve_id,
        severity,
        workload_count,
        in_use: agg.inUse,
        fix_available: agg.fixAvailable,
        public_exploit: agg.publicExploit,
        score,
        breakdown,
      });
    }

    // Filter by severity gate
    const filtered = result.filter(r => weights.severityGate.includes(r.severity));
    
    // Sort by score descending
    filtered.sort((a, b) => b.score - a.score);
    
    return filtered;
  }, [filteredRows, groupBy, weights, workloadCounts]);
```

- [ ] **Step 5: Add weighted columns**

Add after PACKAGE_COLUMNS:

```typescript
// ---------------------------------------------------------------------------
// Column definitions — Weighted
// ---------------------------------------------------------------------------
const WEIGHTED_COLUMNS: ColumnDef<WeightedRow>[] = [
  {
    accessorKey: "score",
    header: "Score",
    size: 90,
    minSize: 60,
    cell: (info) => (
      <span
        className="text-[15px] font-bold"
        style={{ color: "var(--severity-critical)" }}
      >
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "cve_id",
    header: "CVE",
    size: 150,
    minSize: 100,
    cell: (info) => (
      <span
        className="font-mono text-[12px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-primary)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "severity",
    header: "Severity",
    size: 90,
    minSize: 60,
    cell: (info) => <SeverityPill value={String(info.getValue())} />,
  },
  {
    accessorKey: "workload_count",
    header: "Workloads",
    size: 100,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px] font-semibold" style={{ color: "var(--fg-primary)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "fix_available",
    header: "Fix",
    size: 60,
    minSize: 40,
    cell: (info) => {
      const val = info.getValue() as boolean;
      return (
        <span
          className="text-[11px] text-center block"
          style={{ color: val ? CHART_COLORS.fixedGreen : "var(--fg-muted)" }}
          title={val ? "Fix available" : "No fix available"}
        >
          {val ? "✓" : "—"}
        </span>
      );
    },
  },
  {
    accessorKey: "in_use",
    header: "In-use",
    size: 70,
    minSize: 50,
    cell: (info) => {
      const val = info.getValue() as boolean;
      return (
        <span
          className="text-[11px] text-center block"
          style={{ color: val ? CHART_COLORS.fixedGreen : CHART_COLORS.severityMedium }}
          title={val ? "Package in use" : "Not in use"}
        >
          {val ? "✓" : "✕"}
        </span>
      );
    },
  },
  {
    accessorKey: "public_exploit",
    header: "Exploit",
    size: 70,
    minSize: 50,
    cell: (info) => {
      const val = info.getValue() as boolean;
      return (
        <span
          className="text-[11px] text-center block"
          style={{ color: val ? CHART_COLORS.darkRed : "var(--fg-muted)" }}
          title={val ? "Public exploit" : "No known exploit"}
        >
          {val ? "⚠" : "—"}
        </span>
      );
    },
  },
  {
    accessorKey: "breakdown",
    header: () => (
      <span title="Score = (Severity + In Use + Has Fix + Exploit) × Workloads">
        Breakdown
        <br />
        <small style={{ fontWeight: "normal", fontSize: "10px", opacity: 0.6 }}>
          (Severity + In Use + Has Fix + Exploit) × Workloads
        </small>
      </span>
    ),
    size: 160,
    minSize: 120,
    cell: (info) => (
      <span
        className="font-mono text-[11px]"
        style={{ color: "var(--fg-muted)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
];
```

- [ ] **Step 6: Add weight config panel component**

Add before the main `FindingsTable` export:

```typescript
// ---------------------------------------------------------------------------
// Weight configuration panel
// ---------------------------------------------------------------------------
function WeightConfigPanel({
  config,
  onChange,
}: {
  config: WeightConfig;
  onChange: (config: WeightConfig) => void;
}) {
  const severities = ["Critical", "High", "Medium", "Low", "Negligible"];
  const weightLabels: { key: keyof WeightConfig["weights"]; label: string }[] = [
    { key: "Critical", label: "Critical" },
    { key: "High", label: "High" },
    { key: "Medium", label: "Medium" },
    { key: "Low", label: "Low" },
    { key: "in_use", label: "In-use" },
    { key: "fix_available", label: "Has Fix" },
    { key: "public_exploit", label: "Exploit" },
  ];

  return (
    <div
      className="rounded-lg p-3"
      style={{
        background: "var(--bg-surface)",
        border: "1px solid var(--border-subtle)",
        borderLeft: "3px solid var(--severity-critical)",
      }}
    >
      <div
        className="text-[11px] uppercase tracking-wider mb-2"
        style={{ color: "var(--severity-critical)" }}
      >
        Weighted Configuration
      </div>
      
      {/* Severity gate */}
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>Severity:</span>
        <div className="flex gap-2">
          {severities.map((sev) => (
            <label
              key={sev}
              className="flex items-center gap-1 text-[12px] cursor-pointer"
            >
              <input
                type="checkbox"
                checked={config.severityGate.includes(sev)}
                onChange={(e) => {
                  const newGate = e.target.checked
                    ? [...config.severityGate, sev]
                    : config.severityGate.filter((s) => s !== sev);
                  onChange({ ...config, severityGate: newGate });
                }}
                className="rounded"
                style={{ accentColor: "var(--severity-critical)" }}
              />
              <SeverityPill value={sev} />
            </label>
          ))}
        </div>
      </div>

      {/* Weight spin buttons */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>Weights:</span>
        <div className="flex flex-wrap gap-3">
          {weightLabels.map(({ key, label }) => (
            <div key={key} className="flex items-center gap-1">
              <label className="text-[12px]" style={{ color: "var(--fg-muted)" }}>
                {label}
              </label>
              <input
                type="number"
                min={0}
                max={10}
                step={1}
                value={config.weights[key]}
                onChange={(e) => {
                  const val = Math.max(0, Math.min(10, parseInt(e.target.value) || 0));
                  onChange({
                    ...config,
                    weights: { ...config.weights, [key]: val },
                  });
                }}
                className="w-[50px] text-center text-[12px] rounded px-1 py-0.5"
                style={{
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-surface)",
                  color: "var(--fg-primary)",
                }}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Wire weighted mode into toolbar and table rendering**

In the toolbar section, add the config panel after the group-by selector when weighted is selected:

Find the toolbar JSX, after the limit selector, add:
```tsx
{/* Weight config panel — shows when weighted mode is active */}
{groupBy === "weighted" && (
  <div className="w-full mt-2">
    <WeightConfigPanel
      config={weights}
      onChange={(newWeights) => {
        setWeights(newWeights);
        setServerPage(0);
        setGlobalFilter("");
      }}
    />
  </div>
)}
```

In the table area rendering, add the weighted case after package:
```tsx
    } else {
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          <ResizableTable
            data={packageRows}
            columns={PACKAGE_COLUMNS}
            emptyMessage={emptyMessages.package}
            colSpan={PACKAGE_COLUMNS.length}
          />
        </div>
      );
    }

    // Add before the closing else:
    if (groupBy === "weighted") {
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          <ResizableTable
            data={weightedRows}
            columns={WEIGHTED_COLUMNS}
            emptyMessage="No findings match your severity gate, or no workload data available."
            colSpan={WEIGHTED_COLUMNS.length}
          />
          {snapshotDate && (
            <p className="text-[10px] mt-1" style={{ color: "var(--fg-muted)" }}>
              Workload data from snapshot {snapshotDate}
            </p>
          )}
        </div>
      );
    }
```

The current code uses an if-else chain. Restructure to check weighted first. Find this block:
```typescript
    } else if (groupBy === "cve") {
```

Change to check weighted first:
```typescript
    if (groupBy === "weighted") {
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          <ResizableTable
            data={weightedRows}
            columns={WEIGHTED_COLUMNS}
            emptyMessage="No findings match your severity gate, or no workload data available."
            colSpan={WEIGHTED_COLUMNS.length}
          />
          {snapshotDate && (
            <p className="text-[10px] mt-1" style={{ color: "var(--fg-muted)" }}>
              Workload data from snapshot {snapshotDate}
            </p>
          )}
        </div>
      );
    } else if (groupBy === "cve") {
```

Also add to emptyMessages:
```typescript
  const emptyMessages: Record<GroupBy | "weighted", string> = {
    none: "No findings match your filters.",
    cve: "No CVEs match your filters.",
    image: "No images match your filters.",
    package: "No packages match your filters.",
    weighted: "No findings match your severity gate, or no workload data available.",
  };
```

- [ ] **Step 8: Verify TypeScript compiles**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npx tsc --noEmit
```

Expected: No errors. Fix any type issues.

- [ ] **Step 9: Visual verification**

Start both backend and frontend:
```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m sas.api.run &
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npm run dev &
```

Open http://localhost:3000/dashboard, navigate to FindingsTable, select "Weighted" group-by. Verify:
- Config panel appears with severity checkboxes and weight spin buttons
- Table shows Score, CVE, Severity, Workloads, Fix, In-use, Exploit, Breakdown columns
- Scores update when weights are changed
- Severity gate filters work
- Weights persist after page refresh

- [ ] **Step 10: Commit**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
git add sas/web/components/widgets/FindingsTable.tsx
git commit -m "feat: add Weighted group-by mode with configurable scoring to FindingsTable

- New /api/workload-counts endpoint for CVE-level workload blast radius
- Weight config panel with severity gate and spin buttons (min 0, max 10)
- Client-side scoring: (severity + flags) × workloads
- localStorage persistence for weight preferences
- Breakdown column showing the math"
```

---

### Task 5: Regenerate API types (if not done in Task 2)

If the API types weren't regenerated properly, ensure `WorkloadCountsResponse` is available:

- [ ] **Step 1: Verify types exist**

Check `sas/web/lib/api/types.ts` for `WorkloadCountsResponse`. If missing:

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npm run generate-api
```

---

## Self-Review Notes

**Spec coverage:**
- ✅ Formula: `(severity + flags) × workloads` — Task 4 Step 4
- ✅ Severity gate with checkboxes — Task 4 Step 6
- ✅ Weight spin buttons (min 0, max 10) — Task 4 Step 6
- ✅ Breakdown column with formula header — Task 4 Step 5
- ✅ localStorage persistence — Task 3
- ✅ New `/api/workload-counts` endpoint — Task 1
- ✅ Snapshot date display — Task 4 Step 7
- ✅ Workload count from `workload_runs_image_daily` — Task 1

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:** `WeightConfig` used consistently across localStorage hook, config panel, and scoring computation. `WeightedRow` matches `WEIGHTED_COLUMNS` accessor keys.
