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
// Query definitions — Widget: New vs Fixed vs Regressed
// ---------------------------------------------------------------------------
const COMMON_TIME = { mode: "last_n_snapshots" as const, n: 90, granularity: "day" as const };

const NEW_QUERY: QueryIn = {
  lens: "Image",
  traversal: [],
  time: COMMON_TIME,
  measure: "count_new",
  filters: [],
  group_by: [],
  order_by: null,
  limit: null,
};
const FIXED_QUERY: QueryIn = { ...NEW_QUERY, measure: "count_fixed" };
const REGRESSED_QUERY: QueryIn = { ...NEW_QUERY, measure: "count_regressed" };

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
// Aggregate helper — sum y[] across all series (images) per date
// ---------------------------------------------------------------------------
function aggregateSeries(result: QueryResult): { dates: string[]; totals: number[] } {
  const allDates = new Set<string>();
  for (const s of result.series) {
    for (const d of s.x) allDates.add(d);
  }
  const dates = Array.from(allDates).sort();
  const totals: number[] = dates.map((date) => {
    let total = 0;
    for (const s of result.series) {
      const idx = s.x.indexOf(date);
      if (idx >= 0) {
        const v = s.y[idx];
        if (typeof v === "number") total += v;
      }
    }
    return total;
  });
  return { dates, totals };
}

// ---------------------------------------------------------------------------
// ECharts option builder
// ---------------------------------------------------------------------------
function buildChartOption(
  newCounts: number[],
  fixedCounts: number[],
  regressedCounts: number[],
  dates: string[],
  axisLabels: boolean,
): object {
  return {
    backgroundColor: "transparent",
    grid: standardGrid(axisLabels ? 32 : 0),
    xAxis: standardXAxis(dates, axisLabels),
    yAxis: STANDARD_Y_AXIS,
    series: [
      // Bars first so line draws on top
      {
        type: "bar",
        name: "Fixed",
        data: fixedCounts.map((v) => -v), // below zero
        itemStyle: { color: CHART_COLORS.fixedGreen },
        barWidth: "60%",
      },
      {
        type: "bar",
        name: "Regressed",
        data: regressedCounts, // above zero
        itemStyle: { color: CHART_COLORS.severityHigh },
        barWidth: "60%",
      },
      {
        ...flowingLineSeries({ color: CHART_COLORS.severityCritical }),
        name: "New",
        data: newCounts,
        z: 10, // line on top
      },
    ],
    tooltip: {
      trigger: "axis",
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const arr = params as Array<{ axisValue: string; seriesName: string; value: number }>;
        if (!arr.length) return "";
        const date = arr[0].axisValue;
        const lines = arr.map((p) => {
          const v = Math.abs(p.value);
          return `<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:${CHART_COLORS.greyMuted}">${p.seriesName}:</span><b>${v.toLocaleString("en-GB")}</b></div>`;
        });
        return `<div style="font-size:11px;min-width:140px">
          <div style="color:${CHART_COLORS.greyMuted};margin-bottom:4px">${date}</div>
          ${lines.join("")}
        </div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function NewFixedRegressed() {
  const [newResult, setNewResult] = useState<QueryResult | null>(null);
  const [fixedResult, setFixedResult] = useState<QueryResult | null>(null);
  const [regressedResult, setRegressedResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [axisLabels, setAxisLabels] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([runQuery(NEW_QUERY), runQuery(FIXED_QUERY), runQuery(REGRESSED_QUERY)])
      .then(([nr, fr, rr]) => {
        if (!cancelled) {
          setNewResult(nr);
          setFixedResult(fr);
          setRegressedResult(rr);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load data.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const isLoading = newResult === null && fixedResult === null && regressedResult === null && !error;

  // Aggregate totals per date from all three results
  const { dates: rawDates, totals: rawNewTotals } = newResult
    ? aggregateSeries(newResult)
    : { dates: [] as string[], totals: [] as number[] };
  const { totals: rawFixedTotals } = fixedResult
    ? aggregateSeries(fixedResult)
    : { totals: [] as number[] };
  const { totals: rawRegressedTotals } = regressedResult
    ? aggregateSeries(regressedResult)
    : { totals: [] as number[] };

  // Drop the first data point — it always shows every finding as "NEW" because
  // there is no prior snapshot to compare against. This is technically correct
  // but wildly misleading to a viewer.
  const dates = rawDates.slice(1);
  const newTotals = rawNewTotals.slice(1);
  const fixedTotals = rawFixedTotals.slice(1);
  const regressedTotals = rawRegressedTotals.slice(1);

  // Build narrative footer (computed from TRIMMED arrays)
  const footer = newResult && fixedResult && regressedResult
    ? (() => {
        const totalNew = newTotals.reduce((a, b) => a + b, 0);
        const totalFixed = fixedTotals.reduce((a, b) => a + b, 0);
        const totalRegressed = regressedTotals.reduce((a, b) => a + b, 0);
        const n = dates.length;
        return `Over the last ${n} snapshot${n === 1 ? "" : "s"} after the initial ingest, ${totalNew.toLocaleString("en-GB")} were newly discovered, ${totalFixed.toLocaleString("en-GB")} fixed, and ${totalRegressed.toLocaleString("en-GB")} regressed.`;
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
  } else if (isLoading) {
    body = <ChartSkeleton />;
  } else {
    // If we have ≤1 raw data point the trimmed window is empty — show a
    // friendly message rather than a 0-point chart.
    if (rawDates.length <= 1) {
      body = (
        <div
          className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--fg-muted)" }}
        >
          Trend will develop after multiple snapshots have been ingested.
        </div>
      );
    } else if (dates.length === 0) {
      body = (
        <div
          className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--fg-muted)" }}
        >
          No activity data in this window.
        </div>
      );
    } else {
      body = (
        <ReactECharts
          option={buildChartOption(newTotals, fixedTotals, regressedTotals, dates, axisLabels)}
          style={{ height: "220px", width: "100%" }}
          notMerge
          lazyUpdate={false}
        />
      );
    }
  }

  return (
    <WidgetCard
      label="Fleet Activity"
      title="New vs Fixed vs Regressed"
      footer={footer}
      axisLabels={axisLabels}
      onAxisLabelsChange={setAxisLabels}
    >
      {body}
    </WidgetCard>
  );
}
