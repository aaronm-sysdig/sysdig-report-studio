# Fleet Severity Snapshot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full-width traffic-light widget at the top of the dashboard showing latest open findings by severity

**Architecture:** New React component making 5 parallel API queries on mount, rendering coloured blocks via CSS flexbox. No ECharts dependency. Wrapped in existing `WidgetCard` shell.

**Tech Stack:** Next.js 16, React, TypeScript, Tailwind v4

**Files:**
- Create: `sas/web/components/widgets/FleetSeveritySnapshot.tsx`
- Modify: `sas/web/app/dashboard/page.tsx`

---

### Task 1: Create FleetSeveritySnapshot Component

**Files:**
- Create: `sas/web/components/widgets/FleetSeveritySnapshot.tsx`

- [ ] **Step 1: Write the component file**

Create the file with the following content. Key design decisions:
- 5 parallel queries defined as constants at module scope
- Single `useEffect` with `Promise.allSettled` to handle partial failures
- Counts extracted by summing `y[]` across all series for the single returned date
- Latest date extracted from first successful query's `snapshot_range`
- Skeleton shows 5 grey blocks matching the final layout

```tsx
"use client";

import { useEffect, useState } from "react";
import { WidgetCard } from "./WidgetCard";
import { runQuery } from "@/lib/api/client";
import type { QueryIn, QueryResult } from "@/lib/api/client";
import { CHART_COLORS } from "@/lib/charts/defaults";

// ---------------------------------------------------------------------------
// Severity configuration
// ---------------------------------------------------------------------------
type SeverityKey = "Critical" | "High" | "Medium" | "Low" | "Negligible";

interface SeverityConfig {
  key: SeverityKey;
  measure: string;
  background: string;
  color: string;
}

const SEVERITIES: SeverityConfig[] = [
  { key: "Critical", measure: "count_open_critical", background: CHART_COLORS.severityCritical, color: "#FFFFFF" },
  { key: "High", measure: "count_open_high", background: CHART_COLORS.severityHigh, color: "#000000" },
  { key: "Medium", measure: "count_open_medium", background: CHART_COLORS.severityMedium, color: "#000000" },
  { key: "Low", measure: "count_open_low", background: CHART_COLORS.severityLow, color: "#000000" },
  { key: "Negligible", measure: "count_open_negligible", background: CHART_COLORS.severityNegligible, color: "#FFFFFF" },
];

// ---------------------------------------------------------------------------
// Query builder
// ---------------------------------------------------------------------------
function buildQuery(measure: string): QueryIn {
  return {
    lens: "Image",
    traversal: [],
    time: {
      mode: "last_n_snapshots",
      n: 1,
      granularity: "day",
    },
    measure,
    filters: [],
    group_by: [],
    order_by: null,
    limit: null,
  };
}

// ---------------------------------------------------------------------------
// Helper — extract total count and date from a QueryResult
// ---------------------------------------------------------------------------
function extractCount(result: QueryResult): { count: number | null; date: string | null } {
  let total = 0;
  let hasData = false;
  for (const series of result.series) {
    for (const v of series.y) {
      if (typeof v === "number") {
        total += v;
        hasData = true;
      }
    }
  }
  const date = result.snapshot_range?.[0] ?? null;
  return { count: hasData ? total : null, date };
}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------
function SkeletonBlocks() {
  return (
    <div
      className="w-full flex gap-2"
      style={{ minHeight: "100px" }}
      aria-label="Loading severity data…"
      role="status"
    >
      {SEVERITIES.map((s) => (
        <div
          key={s.key}
          className="flex-1 animate-pulse"
          style={{
            backgroundColor: "var(--bg-surface)",
            borderRadius: "var(--radius)",
            minHeight: "100px",
          }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single severity block
// ---------------------------------------------------------------------------
function SeverityBlock({ config, count }: { config: SeverityConfig; count: number | null }) {
  return (
    <div
      className="flex-1 flex flex-col items-center justify-center"
      style={{
        backgroundColor: config.background,
        color: config.color,
        borderRadius: "var(--radius)",
        minHeight: "100px",
        padding: "16px 12px",
      }}
    >
      <div style={{ fontSize: "28px", fontWeight: 700 }}>
        {count !== null ? count.toLocaleString("en-GB") : "—"}
      </div>
      <div
        style={{
          fontSize: "10px",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginTop: "2px",
          opacity: 0.85,
        }}
      >
        {config.key}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
interface CountsState {
  [key: string]: number | null;
}

export function FleetSeveritySnapshot() {
  const [counts, setCounts] = useState<CountsState>({});
  const [latestDate, setLatestDate] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const queries = SEVERITIES.map((s) => buildQuery(s.measure));
    const promises = queries.map((q) => runQuery(q));

    Promise.allSettled(promises).then((results) => {
      if (cancelled) return;

      const countsMap: CountsState = {};
      let firstDate: string | null = null;
      let hasAnyError = false;

      results.forEach((r, idx) => {
        const key = SEVERITIES[idx].key;
        if (r.status === "fulfilled") {
          const { count, date } = extractCount(r.value);
          countsMap[key] = count;
          if (date && !firstDate) firstDate = date;
        } else {
          countsMap[key] = null;
          hasAnyError = true;
        }
      });

      setCounts(countsMap);
      if (firstDate) setLatestDate(firstDate);
      if (hasAnyError) {
        setError("Some severity data could not be loaded.");
      }
      setLoading(false);
    });

    return () => { cancelled = true; };
  }, []);

  // Build footer narrative
  const footer = (() => {
    if (loading || Object.keys(counts).length === 0) return undefined;
    const total = SEVERITIES.reduce((sum, s) => {
      const c = counts[s.key];
      return sum + (c !== null ? c : 0);
    }, 0);
    const date = latestDate ?? "unknown date";
    return `As of ${date} · ${total.toLocaleString("en-GB")} total open findings`;
  })();

  // Decide what to render
  let body: React.ReactNode;
  if (error && Object.values(counts).every((c) => c === null)) {
    body = (
      <div
        className="flex items-center justify-center"
        style={{ minHeight: "100px", color: "var(--severity-critical)" }}
        role="alert"
      >
        Unable to load severity data.
      </div>
    );
  } else if (loading) {
    body = <SkeletonBlocks />;
  } else {
    body = (
      <div className="w-full flex gap-2">
        {SEVERITIES.map((s) => (
          <SeverityBlock key={s.key} config={s} count={counts[s.key]} />
        ))}
      </div>
    );
  }

  // Show error banner if partial failure
  const errorBanner = error && Object.values(counts).some((c) => c !== null) ? (
    <div
      className="w-full text-xs px-2 py-1"
      style={{ color: "var(--severity-critical)" }}
      role="alert"
    >
      {error}
    </div>
  ) : null;

  return (
    <WidgetCard
      label="Fleet Metrics"
      title="Fleet Severity Snapshot"
      footer={footer}
    >
      {errorBanner}
      {body}
    </WidgetCard>
  );
}
```

- [ ] **Step 2: Verify the component compiles**

Run the TypeScript check to ensure no type errors:

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npx tsc --noEmit --pretty 2>&1 | head -50
```

Expected: No errors related to `FleetSeveritySnapshot.tsx`. Fix any type errors before proceeding.

- [ ] **Step 3: Commit the component**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio && git add sas/web/components/widgets/FleetSeveritySnapshot.tsx && git commit -m "feat(sas): add FleetSeveritySnapshot widget component

Static traffic-light display showing latest open findings by severity.
Five parallel queries, no ECharts dependency, pure CSS flexbox."
```

---

### Task 2: Add Widget to Dashboard

**Files:**
- Modify: `sas/web/app/dashboard/page.tsx`

- [ ] **Step 1: Add import and widget to dashboard**

Add the import at the top of the file, and add the widget as Row 1 (before ImageRemediationStory):

```tsx
// Add to imports
import { FleetSeveritySnapshot } from "@/components/widgets/FleetSeveritySnapshot";

// Add as first row in the grid, before ImageRemediationStory:
{/* Row 1 — severity snapshot, full width */}
<div style={{ gridColumn: "span 12" }}>
  <FleetSeveritySnapshot />
</div>
```

The full file should read:

```tsx
import { AppShell } from "@/components/app-shell/AppShell";
import { FleetSeveritySnapshot } from "@/components/widgets/FleetSeveritySnapshot";
import { FleetCriticalTrend } from "@/components/widgets/FleetCriticalTrend";
import { NewFixedRegressed } from "@/components/widgets/NewFixedRegressed";
import { KevRansomwareExposure } from "@/components/widgets/KevRansomwareExposure";
import { RepositoryTagHygiene } from "@/components/widgets/RepositoryTagHygiene";
import { ImageInventoryGrid } from "@/components/widgets/ImageInventoryGrid";
import { FindingsTable } from "@/components/widgets/FindingsTable";
import { ImageRemediationStory } from "@/components/widgets/ImageRemediationStory";

export default function DashboardPage() {
  return (
    <AppShell pageTitle="Dashboard">
      {/* 12-column CSS grid — widgets span 6 or 12 columns */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(12, 1fr)",
          gap: "var(--gap-widget)",
        }}
      >
        {/* Row 1 — severity snapshot, full width */}
        <div style={{ gridColumn: "span 12" }}>
          <FleetSeveritySnapshot />
        </div>

        {/* Row 2 — flagship widget, full width */}
        <div style={{ gridColumn: "span 12" }}>
          <ImageRemediationStory />
        </div>

        {/* Row 3 — fleet metrics side by side */}
        <div style={{ gridColumn: "span 6" }}>
          <FleetCriticalTrend />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <NewFixedRegressed />
        </div>

        {/* Row 4 — exposure + repository hygiene */}
        <div style={{ gridColumn: "span 6" }}>
          <KevRansomwareExposure />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <RepositoryTagHygiene />
        </div>

        {/* Row 5 — image inventory full width */}
        <div style={{ gridColumn: "span 12" }}>
          <ImageInventoryGrid />
        </div>

        {/* Row 6 — findings table full width */}
        <div style={{ gridColumn: "span 12" }}>
          <FindingsTable />
        </div>
      </div>
    </AppShell>
  );
}
```

- [ ] **Step 2: Verify the build passes**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npx tsc --noEmit --pretty 2>&1 | head -50
```

Expected: No errors.

- [ ] **Step 3: Commit the dashboard change**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio && git add sas/web/app/dashboard/page.tsx && git commit -m "feat(sas): place FleetSeveritySnapshot at top of dashboard

Full-width widget before ImageRemediationStory. Gives immediate
'where are we at' view without scrolling."
```

---

### Task 3: Visual Verification

- [ ] **Step 1: Start the backend** (if not already running)

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m sas.api.run
```

Confirm it's running: `curl http://localhost:8000/healthz` should return `ok`.

- [ ] **Step 2: Start the frontend** (if not already running)

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npm run dev
```

- [ ] **Step 3: Open http://localhost:3000 and verify**

Checklist:
- [ ] Widget appears at the very top of the dashboard
- [ ] Five coloured blocks are visible side-by-side
- [ ] Each block shows a number and severity label
- [ ] Colours match the severity palette (purple=critical, red=high, orange=medium, yellow=low, grey=negligible)
- [ ] Footer shows "As of {date}" with total count
- [ ] Widget title reads "Fleet Severity Snapshot"
- [ ] Label reads "Fleet Metrics"
- [ ] No console errors in browser DevTools

---

## Self-Review

**Spec coverage:**
- [x] Five severity blocks with correct colours
- [x] Parallel queries for each severity
- [x] Skeleton loading state
- [x] Error handling (full and partial)
- [x] Footer with date and total
- [x] WidgetCard wrapper with label/title
- [x] Dashboard placement at Row 1, span 12
- [x] No ECharts dependency
- [x] Static (no click handlers)

**Placeholder scan:** No TBDs, no "add validation", no vague steps. All code is complete.

**Type consistency:** `CountsState` uses string keys matching `SeverityKey`. `extractCount` returns `{ count, date }` matching usage in `useEffect`. `CHART_COLORS` imports match existing pattern from `FleetCriticalTrend`.
