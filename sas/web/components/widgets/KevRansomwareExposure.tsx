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
// Query definition — KEV-Ransomware Exposure widget
// Filters to findings where cisa_kev_known_ransomware = true (Task 1 denorm)
// ---------------------------------------------------------------------------
const KEV_RANSOMWARE_QUERY: QueryIn = {
  lens: "Image",
  traversal: [],
  time: {
    mode: "last_n_snapshots",
    n: 90,
    granularity: "day",
  },
  measure: "count_open",
  filters: [
    { field: "cisa_kev_known_ransomware", operator: "eq", value: true },
  ],
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
// Aggregate series by date (sum across all image-series at each date)
// ---------------------------------------------------------------------------
function aggregateSeriesByDate(result: QueryResult): {
  dates: string[];
  counts: number[];
} {
  const allDates = new Set<string>();
  for (const s of result.series) {
    for (const d of s.x) allDates.add(d);
  }
  const dates = Array.from(allDates).sort();

  const counts: number[] = dates.map((date) => {
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

  return { dates, counts };
}

// ---------------------------------------------------------------------------
// ECharts option builder — red area-fill line for urgency signalling
// ---------------------------------------------------------------------------
function buildChartOption(result: QueryResult, axisLabels: boolean): object {
  const { dates, counts } = aggregateSeriesByDate(result);

  return {
    backgroundColor: "transparent",
    grid: standardGrid(axisLabels ? 32 : 0),
    xAxis: standardXAxis(dates, axisLabels),
    yAxis: STANDARD_Y_AXIS,
    series: [
      {
        ...flowingLineSeries({ color: CHART_COLORS.severityCritical }),
        data: counts,
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(255, 119, 116, 0.25)" }, // 25% severityCritical at top
              { offset: 1, color: "rgba(255, 119, 116, 0.0)" },  // transparent at bottom
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
          <div><b>${p.value?.toLocaleString("en-GB") ?? "—"}</b> ransomware-exposed open</div>
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
      .then((r) => {
        if (!cancelled) setResult(r);
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

  // Build footer narrative from latest aggregate count
  const footer = result
    ? (() => {
        const { dates, counts } = aggregateSeriesByDate(result);
        if (dates.length === 0) return undefined;
        const latest = counts[counts.length - 1] ?? 0;
        if (latest === 0) {
          return "No findings exposed to known-ransomware CVEs in this window — the lab data may not contain any.";
        }
        return `As of latest snapshot, ${latest.toLocaleString("en-GB")} findings exposed to known-ransomware CVEs across the fleet.`;
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
    const totalPoints = result.series.reduce((sum, s) => sum + s.y.length, 0);
    if (totalPoints === 0) {
      body = (
        <div
          className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--fg-muted)" }}
        >
          No findings exposed to known-ransomware CVEs in this window.
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
      label="Threat Exposure"
      title="KEV-Ransomware Exposure"
      footer={footer}
      axisLabels={axisLabels}
      onAxisLabelsChange={setAxisLabels}
    >
      {body}
    </WidgetCard>
  );
}
