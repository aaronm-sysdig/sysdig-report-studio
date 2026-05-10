# Fleet Remediation Story — Implementation Plan

**Goal:** Fleet-wide version of ImageRemediationStory — no repo picker, no tag lineage, just totals across all images.

**Architecture:** New React component making 8 parallel queries (count_open + 4 severities + count_new + count_fixed + count_regressed) for 90-day history, plus 4 reason-code queries for closures breakdown. ECharts for stacked bar + flow charts.

**Files:**
- Create: `sas/web/components/widgets/FleetRemediationStory.tsx`
- Modify: `sas/web/app/dashboard/page.tsx`

**Key design decisions:**
- Dynamic window: derive actual days from series `x` arrays (not hardcoded "90 days")
- `snapshot_range` is min/max only — use `alignToSharedDates` pattern from ImageRemediationStory
- Fleet-level = `lens: "Image"`, `filters: []`, sum across all series
- No tag lineage panel, no repo picker
- Same StatCard, ReasonCodeBar, ECharts patterns as ImageRemediationStory

---

### Task 1: Create FleetRemediationStory Component

**Files:**
- Create: `sas/web/components/widgets/FleetRemediationStory.tsx`

- [ ] **Step 1: Write the component**

```tsx
"use client";

import { useEffect, useState, useMemo } from "react";
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
// Types
// ---------------------------------------------------------------------------
interface RemediationData {
  dates: string[];
  critical: number[];
  high: number[];
  medium: number[];
  low: number[];
  totals: number[];
  newCounts: number[];
  fixedCounts: number[];
  regressedCounts: number[];
}

interface HeadlineMetrics {
  currentOpen: number;
  totalNew: number;
  totalFixed: number;
  totalRegressed: number;
  deltaVs7Days: number | null;
  windowDays: number;
}

interface ReasonTotals {
  patched: number;
  retired: number;
  accepted: number;
  other: number;
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------
const COMMON_TIME: QueryIn["time"] = {
  mode: "last_n_snapshots",
  n: 90,
  granularity: "day",
};

function fleetQuery(measure: string): QueryIn {
  return {
    lens: "Image",
    traversal: [],
    time: COMMON_TIME,
    measure,
    filters: [],
    group_by: [],
    order_by: null,
    limit: null,
  };
}

/** Sum across ALL series into {dates, values} — fleet-wide aggregation. */
function extractSeries(result: QueryResult): { dates: string[]; values: number[] } {
  if (!result.series.length) return { dates: [], values: [] };

  const dateSet = new Set<string>();
  for (const s of result.series) {
    for (const d of s.x) dateSet.add(d);
  }
  const dates = Array.from(dateSet).sort();

  const values = dates.map((d) => {
    let total = 0;
    for (const s of result.series) {
      const idx = (s.x as string[]).indexOf(d);
      if (idx >= 0) {
        const v = (s.y as number[])[idx];
        if (typeof v === "number") total += v;
      }
    }
    return total;
  });

  return { dates, values };
}

/** Align multiple value arrays to a shared dates array */
function alignToSharedDates(
  allSeries: Array<{ dates: string[]; values: number[] }>,
): { dates: string[]; aligned: number[][] } {
  const dateSet = new Set<string>();
  for (const s of allSeries) {
    for (const d of s.dates) dateSet.add(d);
  }
  const dates = Array.from(dateSet).sort();
  const aligned = allSeries.map(({ dates: sDates, values }) =>
    dates.map((d) => {
      const idx = sDates.indexOf(d);
      return idx >= 0 ? (values[idx] ?? 0) : 0;
    }),
  );
  return { dates, aligned };
}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------
function ChartSkeleton({ height }: { height: number }) {
  return (
    <div
      className="w-full animate-pulse"
      style={{
        height,
        backgroundColor: "var(--bg-surface)",
        borderRadius: "var(--radius)",
      }}
      aria-label="Loading chart data…"
      role="status"
    />
  );
}

// ---------------------------------------------------------------------------
// Stat card (reused from ImageRemediationStory pattern)
// ---------------------------------------------------------------------------
function StatCard({ label, value, delta }: {
  label: string;
  value: number;
  delta?: { value: number; label: string; positive: boolean } | null;
}) {
  return (
    <div
      className="flex flex-col gap-0.5 px-3 py-2 rounded"
      style={{
        backgroundColor: "var(--bg-surface)",
        border: "1px solid var(--border-subtle)",
        flex: "1 1 0",
        minWidth: 0,
      }}
    >
      <span
        className="text-[10px] font-medium tracking-widest uppercase"
        style={{ color: "var(--fg-muted)" }}
      >
        {label}
      </span>
      <span
        className="text-[28px] font-semibold leading-none tabular-nums"
        style={{ color: "var(--fg-primary)" }}
      >
        {value.toLocaleString("en-GB")}
      </span>
      {delta !== null && delta !== undefined && (
        <span
          className="text-[10px] leading-none mt-0.5"
          style={{
            color: delta.positive ? CHART_COLORS.fixedGreen : CHART_COLORS.severityCritical,
          }}
        >
          {delta.positive ? "↓" : "↑"} {Math.abs(delta.value).toLocaleString("en-GB")}{" "}
          {delta.label}
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ECharts: main severity-stack + flowing total line
// ---------------------------------------------------------------------------
function buildMainChartOption(data: RemediationData, axisLabels: boolean): object {
  const { dates, critical, high, medium, low, totals } = data;

  return {
    backgroundColor: "transparent",
    grid: standardGrid(axisLabels ? 32 : 4),
    xAxis: standardXAxis(dates, axisLabels),
    yAxis: STANDARD_Y_AXIS,
    series: [
      { type: "bar", name: "Low", data: low, stack: "severity", itemStyle: { color: CHART_COLORS.severityLow }, barMaxWidth: 16 },
      { type: "bar", name: "Medium", data: medium, stack: "severity", itemStyle: { color: CHART_COLORS.severityMedium }, barMaxWidth: 16 },
      { type: "bar", name: "High", data: high, stack: "severity", itemStyle: { color: CHART_COLORS.severityHigh }, barMaxWidth: 16 },
      { type: "bar", name: "Critical", data: critical, stack: "severity", itemStyle: { color: CHART_COLORS.severityCritical }, barMaxWidth: 16 },
      { ...flowingLineSeries({ color: CHART_COLORS.greyMuted, width: 1.5, symbolSize: 3 }), name: "Total", data: totals, z: 10 },
    ],
    tooltip: {
      trigger: "axis",
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const arr = params as Array<{ axisValue: string; seriesName: string; value: number; color: string }>;
        if (!arr.length) return "";
        const date = arr[0].axisValue;
        const rows = arr.filter((p) => p.seriesName !== "Total").map(
          (p) => `<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:${p.color}">&#9632;</span><span style="color:${CHART_COLORS.greyMuted};flex:1;margin-left:4px">${p.seriesName}:</span><b>${(p.value ?? 0).toLocaleString("en-GB")}</b></div>`,
        );
        const totalEntry = arr.find((p) => p.seriesName === "Total");
        const total = totalEntry ? totalEntry.value : 0;
        return `<div style="font-size:11px;min-width:160px"><div style="color:${CHART_COLORS.greyMuted};margin-bottom:4px">${date}</div>${rows.join("")}<div style="border-top:1px solid ${CHART_COLORS.greyBorder};margin-top:4px;padding-top:4px;display:flex;justify-content:space-between"><span style="color:${CHART_COLORS.greyMuted}">Total:</span><b>${total.toLocaleString("en-GB")}</b></div></div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// ECharts: net daily flow chart
// ---------------------------------------------------------------------------
function buildFlowChartOption(
  dates: string[],
  newCounts: number[],
  fixedCounts: number[],
  axisLabels: boolean,
): object {
  return {
    backgroundColor: "transparent",
    grid: { top: 8, right: 16, bottom: axisLabels ? 36 : 8, left: 48, containLabel: false },
    xAxis: { ...standardXAxis(dates, axisLabels) },
    yAxis: {
      type: "value" as const,
      minInterval: 1,
      axisLabel: {
        fontSize: 9,
        color: CHART_COLORS.greyMuted,
        formatter: (v: number) => (v === 0 ? "0" : v > 0 ? `+${v}` : String(v)),
      },
      splitLine: { lineStyle: { color: CHART_COLORS.greyBorder, type: "dashed" as const } },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      { type: "bar", name: "New", data: newCounts, itemStyle: { color: CHART_COLORS.darkRed }, barMaxWidth: 16 },
      { type: "bar", name: "Closed", data: fixedCounts.map((v) => -v), itemStyle: { color: CHART_COLORS.fixedGreen }, barMaxWidth: 16 },
    ],
    tooltip: {
      trigger: "axis",
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const arr = params as Array<{ axisValue: string; seriesName: string; value: number }>;
        if (!arr.length) return "";
        const date = arr[0].axisValue;
        const lines = arr.map((p) => {
          const v = Math.abs(p.value ?? 0);
          const col = p.seriesName === "New" ? CHART_COLORS.severityCritical : CHART_COLORS.fixedGreen;
          return `<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:${col}">${p.seriesName}:</span><b>${v.toLocaleString("en-GB")}</b></div>`;
        });
        return `<div style="font-size:11px;min-width:120px"><div style="color:${CHART_COLORS.greyMuted};margin-bottom:4px">${date}</div>${lines.join("")}</div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Reason-code decomposition bar
// ---------------------------------------------------------------------------
function ReasonCodeBar({ patched, retired, accepted, other }: {
  patched: number; retired: number; accepted: number; other: number;
}) {
  const total = patched + retired + accepted + other;

  if (total === 0) {
    return (
      <div className="text-xs italic" style={{ color: "var(--fg-muted)" }}>
        No findings closed in this window across the fleet.
      </div>
    );
  }

  const segments = [
    { label: "PATCHED", count: patched, color: CHART_COLORS.fixedGreen },
    { label: "RETIRED", count: retired, color: CHART_COLORS.greyMuted },
    { label: "ACCEPTED", count: accepted, color: CHART_COLORS.severityMedium },
    { label: "OTHER", count: other, color: CHART_COLORS.greyBorder },
  ].filter((s) => s.count > 0);

  return (
    <div>
      <div
        className="text-[10px] font-medium tracking-widest uppercase mb-1.5"
        style={{ color: "var(--fg-muted)" }}
      >
        Why {total.toLocaleString("en-GB")} closed?
      </div>
      <div className="flex w-full h-7 overflow-hidden" style={{ borderRadius: "var(--radius)" }}>
        {segments.map((s) => (
          <div
            key={s.label}
            style={{ width: `${(s.count / total) * 100}%`, backgroundColor: s.color, minWidth: 24 }}
            className="flex items-center justify-center text-[10px] font-semibold text-white"
            title={`${s.label}: ${s.count}`}
          >
            {((s.count / total) * 100) >= 8 && `${s.count}`}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]" style={{ color: "var(--fg-muted)" }}>
        {segments.map((s) => (
          <div key={s.label} className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ backgroundColor: s.color }} />
            <span>{s.label} {s.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auto-narrative — fleet-centric
// ---------------------------------------------------------------------------
function buildNarrative(metrics: HeadlineMetrics, reason?: ReasonTotals): string {
  const { totalNew, totalFixed, totalRegressed, windowDays } = metrics;
  const windowLabel = `${windowDays}-day`;

  if (totalNew === 0 && totalFixed === 0 && totalRegressed === 0) {
    return `No new findings, closures, or regressions recorded across the fleet in the last ${windowLabel} window.`;
  }

  const netGrowth = totalNew - totalFixed;

  if (reason && totalFixed > 0) {
    const { patched, retired } = reason;
    if (retired > patched) {
      return `The fleet has ${totalNew.toLocaleString("en-GB")} new findings against ${totalFixed.toLocaleString("en-GB")} closures in the last ${windowLabel} window — a net ${netGrowth > 0 ? "growth" : "reduction"} of ${Math.abs(netGrowth).toLocaleString("en-GB")} open findings. Only ${patched.toLocaleString("en-GB")} were patched (real fixes); ${retired.toLocaleString("en-GB")} disappeared via image retirement.${totalRegressed > 0 ? ` ${totalRegressed.toLocaleString("en-GB")} regressions recorded.` : ""}`;
    }
    if (patched > retired) {
      return `The fleet has ${totalNew.toLocaleString("en-GB")} new findings against ${totalFixed.toLocaleString("en-GB")} closures in the last ${windowLabel} window — a net ${netGrowth > 0 ? "growth" : "reduction"} of ${Math.abs(netGrowth).toLocaleString("en-GB")} open findings. ${patched.toLocaleString("en-GB")} were patched (real fixes); ${retired.toLocaleString("en-GB")} retired.${totalRegressed > 0 ? ` ${totalRegressed.toLocaleString("en-GB")} regressions recorded.` : ""}`;
    }
  }

  return `The fleet has ${totalNew.toLocaleString("en-GB")} new findings against ${totalFixed.toLocaleString("en-GB")} closures in the last ${windowLabel} window — a net ${netGrowth > 0 ? "growth" : "reduction"} of ${Math.abs(netGrowth).toLocaleString("en-GB")} open findings.${totalRegressed > 0 ? ` ${totalRegressed.toLocaleString("en-GB")} regressions recorded.` : ""}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function FleetRemediationStory() {
  const [remediationData, setRemediationData] = useState<RemediationData | null>(null);
  const [metrics, setMetrics] = useState<HeadlineMetrics | null>(null);
  const [reason, setReason] = useState<ReasonTotals | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    // Main trend queries
    const mainQueries = [
      fleetQuery("count_open"),
      fleetQuery("count_open_critical"),
      fleetQuery("count_open_high"),
      fleetQuery("count_open_medium"),
      fleetQuery("count_open_low"),
      fleetQuery("count_new"),
      fleetQuery("count_fixed"),
      fleetQuery("count_regressed"),
    ];

    // Reason code queries
    const reasonQueries = [
      fleetQuery("count_fixed_patched"),
      fleetQuery("count_fixed_retired"),
      fleetQuery("count_fixed_accepted"),
      fleetQuery("count_fixed_other"),
    ];

    Promise.allSettled(mainQueries.map((q) => runQuery(q))).then((mainResults) => {
      if (cancelled) return;

      const seriesResults = mainResults.filter((r) => r.status === "fulfilled") as PromiseFulfilledResult<QueryResult>[];
      if (seriesResults.length < mainQueries.length) {
        setError("Some trend data could not be loaded.");
      }

      const extracted = seriesResults.map((r) => extractSeries(r.value));
      const { dates, aligned } = alignToSharedDates(extracted);

      // [open, critical, high, medium, low, new, fixed, regressed]
      const [openArr, criticalArr, highArr, mediumArr, lowArr, newArr, fixedArr, regressedArr] = aligned;

      const totals = dates.map((_, i) => (criticalArr[i] || 0) + (highArr[i] || 0) + (mediumArr[i] || 0) + (lowArr[i] || 0));

      setRemediationData({
        dates,
        critical: criticalArr,
        high: highArr,
        medium: mediumArr,
        low: lowArr,
        totals,
        newCounts: newArr,
        fixedCounts: fixedArr,
        regressedCounts: regressedArr,
      });

      // Compute headline metrics
      const windowDays = dates.length > 1 ? dates.length - 1 : 0;
      const currentOpen = openArr[openArr.length - 1] || 0;
      const totalNew = newArr.reduce((a, b) => a + b, 0);
      const totalFixed = fixedArr.reduce((a, b) => a + b, 0);
      const totalRegressed = regressedArr.reduce((a, b) => a + b, 0);

      // Delta vs 7 days ago
      let deltaVs7Days: number | null = null;
      const sevenDaysAgoIdx = Math.max(0, openArr.length - 1 - 7);
      if (sevenDaysAgoIdx < openArr.length - 1) {
        deltaVs7Days = currentOpen - (openArr[sevenDaysAgoIdx] || 0);
      }

      setMetrics({ currentOpen, totalNew, totalFixed, totalRegressed, deltaVs7Days, windowDays });
    });

    // Reason codes (independent)
    Promise.allSettled(reasonQueries.map((q) => runQuery(q))).then((reasonResults) => {
      if (cancelled) return;

      const sumFromResult = (result: PromiseSettledResult<QueryResult>): number => {
        if (result.status !== "fulfilled") return 0;
        let total = 0;
        for (const series of result.value.series) {
          for (const v of series.y) {
            if (typeof v === "number") total += v;
          }
        }
        return total;
      };

      setReason({
        patched: sumFromResult(reasonResults[0]),
        retired: sumFromResult(reasonResults[1]),
        accepted: sumFromResult(reasonResults[2]),
        other: sumFromResult(reasonResults[3]),
      });
    });

    setLoading(false);

    return () => { cancelled = true; };
  }, []);

  // Chart responsiveness
  const [axisLabels, setAxisLabels] = useState(true);
  useEffect(() => {
    const mql = window.matchMedia("(min-width: 768px)");
    const handler = (e: MediaQueryListEvent) => setAxisLabels(e.matches);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, []);

  // Error state
  if (error && !remediationData) {
    return (
      <WidgetCard label="Fleet Metrics" title="Fleet Remediation Story">
        <div className="flex items-center justify-center" style={{ minHeight: "200px", color: "var(--severity-critical)" }} role="alert">
          Unable to load remediation data.
        </div>
      </WidgetCard>
    );
  }

  // Loading state
  if (loading || !remediationData || !metrics) {
    return (
      <WidgetCard label="Fleet Metrics" title="Fleet Remediation Story">
        <div className="flex flex-col gap-3">
          <div className="flex gap-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="flex-1 h-20 animate-pulse rounded" style={{ backgroundColor: "var(--bg-surface)" }} />
            ))}
          </div>
          <ChartSkeleton height={240} />
          <ChartSkeleton height={120} />
        </div>
      </WidgetCard>
    );
  }

  const { dates, critical, high, medium, low, totals, newCounts, fixedCounts } = remediationData;
  const { windowDays } = metrics;
  const windowLabel = `${windowDays}-day`;

  const narrative = buildNarrative(metrics, reason || undefined);
  const footer = remediationData.dates.length > 0
    ? `As of ${remediationData.dates[remediationData.dates.length - 1]} · Fleet-wide across all repositories`
    : undefined;

  return (
    <WidgetCard
      label="Fleet Metrics"
      title="Fleet Remediation Story"
      footer={footer}
    >
      {/* Headline cards */}
      <div className="flex gap-2 mb-3">
        <StatCard
          label="Current Open"
          value={metrics.currentOpen}
          delta={metrics.deltaVs7Days !== null ? {
            value: Math.abs(metrics.deltaVs7Days),
            label: "vs 7 days ago",
            positive: metrics.deltaVs7Days <= 0,
          } : null}
        />
        <StatCard label={`${windowLabel} New`} value={metrics.totalNew} />
        <StatCard label={`${windowLabel} Fixed`} value={metrics.totalFixed} />
        <StatCard label={`${windowLabel} Regressed`} value={metrics.totalRegressed} />
      </div>

      {/* Main severity stack chart */}
      <div className="mb-3">
        <div className="text-[10px] font-medium tracking-widest uppercase mb-1.5" style={{ color: "var(--fg-muted)" }}>
          Open findings by severity ({windowLabel} window)
        </div>
        <div style={{ height: 240 }}>
          <ReactECharts
            option={buildMainChartOption(remediationData, axisLabels)}
            style={{ height: "100%", width: "100%" }}
            opts={{ renderer: "canvas", locale: "ZH_CN" }}
          />
        </div>
      </div>

      {/* Net flow chart */}
      <div className="mb-3">
        <div className="text-[10px] font-medium tracking-widest uppercase mb-1.5" style={{ color: "var(--fg-muted)" }}>
          Daily new vs closed ({windowLabel} window)
        </div>
        <div style={{ height: 120 }}>
          <ReactECharts
            option={buildFlowChartOption(dates, newCounts, fixedCounts, axisLabels)}
            style={{ height: "100%", width: "100%" }}
            opts={{ renderer: "canvas", locale: "ZH_CN" }}
          />
        </div>
      </div>

      {/* Reason code breakdown */}
      {reason && (
        <div className="mb-3">
          <ReasonCodeBar {...reason} />
        </div>
      )}

      {/* Narrative */}
      <div
        className="rounded px-3 py-2.5 text-xs leading-relaxed"
        style={{ backgroundColor: "var(--bg-surface)" }}
      >
        {narrative}
      </div>
    </WidgetCard>
  );
}
```

---

### Task 2: Add widget to dashboard

**Files:**
- Modify: `sas/web/app/dashboard/page.tsx`

- [ ] Add import and place widget in Row 2 (between FleetSeveritySnapshot and ImageRemediationStory), full width (span 12)

---

### Task 3: Visual verification

- [ ] Confirm TypeScript compiles, lint passes, widget renders correctly
