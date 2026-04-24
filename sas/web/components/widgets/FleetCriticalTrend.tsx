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

// echarts-for-react uses browser APIs — must be loaded client-side only
const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// ---------------------------------------------------------------------------
// Query definition — Widget 2: Fleet Critical Trend
// Fields match QueryIn from sas/api/routes/query.py
// ---------------------------------------------------------------------------
const FLEET_CRITICAL_QUERY: QueryIn = {
  lens: "Image",
  traversal: [],
  time: {
    mode: "last_n_snapshots",
    n: 90,
    granularity: "day",
  },
  measure: "count_open_critical",
  filters: [],
  group_by: [],
  order_by: null,
  limit: null,
};

// ---------------------------------------------------------------------------
// Skeleton shimmer
// ---------------------------------------------------------------------------
function ChartSkeleton() {
  return (
    <div
      className="w-full h-[220px] animate-pulse"
      style={{
        backgroundColor: "var(--bg-surface)",
        borderRadius: "var(--radius)",
      }}
      aria-label="Loading chart data…"
      role="status"
    />
  );
}

// ---------------------------------------------------------------------------
// ECharts option builder
// ---------------------------------------------------------------------------
function buildChartOption(result: QueryResult, axisLabels: boolean): object {
  // Aggregate series by summing y[] across all images (result may have many series,
  // one per image_id). For a fleet-wide view we want total critical count per day.
  const series = result.series;
  const allDates = new Set<string>();
  for (const s of series) {
    for (const d of s.x) allDates.add(d);
  }
  const dates = Array.from(allDates).sort();

  const counts: number[] = dates.map((date) => {
    let total = 0;
    for (const s of series) {
      const idx = s.x.indexOf(date);
      if (idx >= 0) {
        const v = s.y[idx];
        if (typeof v === "number") total += v;
      }
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
        ...flowingLineSeries({ color: CHART_COLORS.deepSee }),
        data: counts,
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
          <div><b>${p.value?.toLocaleString("en-GB") ?? "—"}</b> critical open</div>
        </div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function FleetCriticalTrend() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [axisLabels, setAxisLabels] = useState(false);

  useEffect(() => {
    let cancelled = false;
    runQuery(FLEET_CRITICAL_QUERY)
      .then((r) => { if (!cancelled) setResult(r); })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load data.");
        }
      });
    return () => { cancelled = true; };
  }, []);

  // Build footer from aggregate of all series
  const footer = result
    ? (() => {
        const allDates = new Set<string>();
        for (const s of result.series) {
          for (const d of s.x) allDates.add(d);
        }
        const dates = Array.from(allDates).sort();
        if (dates.length === 0) return undefined;

        const totalAt = (date: string) => {
          let total = 0;
          for (const s of result.series) {
            const idx = s.x.indexOf(date);
            if (idx >= 0) {
              const v = s.y[idx];
              if (typeof v === "number") total += v;
            }
          }
          return total;
        };
        const latest = totalAt(dates[dates.length - 1]);
        const earliest = totalAt(dates[0]);
        const delta = latest - earliest;
        const direction = delta < 0 ? "down" : delta > 0 ? "up" : "unchanged";
        const abs = Math.abs(delta);
        return direction === "unchanged"
          ? `Critical open findings are unchanged over this period.`
          : `Critical open findings are ${direction} by ${abs.toLocaleString("en-GB")} vs the start of this window.`;
      })()
    : undefined;

  // Decide what to render in the chart area
  let body: React.ReactNode;
  if (error) {
    body = (
      <div
        className="flex items-center justify-center h-[220px] text-sm"
        style={{ color: "var(--severity-critical)" }}
        role="alert"
      >
        Unable to load data: {error}
      </div>
    );
  } else if (result === null) {
    body = <ChartSkeleton />;
  } else {
    // Check aggregate for empty data
    const totalPoints = result.series.reduce((sum, s) => sum + s.y.length, 0);
    if (totalPoints === 0) {
      body = (
        <div
          className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--fg-muted)" }}
        >
          No critical findings in this window.
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
      label="Fleet Metrics"
      title="Fleet Critical Trend"
      footer={footer}
      axisLabels={axisLabels}
      onAxisLabelsChange={setAxisLabels}
    >
      {body}
    </WidgetCard>
  );
}
