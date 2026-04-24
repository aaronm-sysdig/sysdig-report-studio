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

// echarts-for-react uses browser APIs — must be loaded client-side only
const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// ---------------------------------------------------------------------------
// Query definition — Widget: Repository Tag Hygiene
// Latest snapshot only; one series per repository with a single y value.
// ---------------------------------------------------------------------------
const REPO_HYGIENE_QUERY: QueryIn = {
  lens: "Repository",
  traversal: [],
  time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
  measure: "count_open_critical",
  filters: [],
  group_by: [],
  order_by: null,
  limit: null,
};

const TOP_N = 15;

// ---------------------------------------------------------------------------
// Skeleton shimmer
// ---------------------------------------------------------------------------
function ChartSkeleton() {
  return (
    <div
      className="w-full h-[340px] animate-pulse"
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
// Data shape after aggregation
// ---------------------------------------------------------------------------
interface RepoRow {
  repository: string;
  count: number;
}

function seriesKeyToString(key: { [k: string]: unknown }): string {
  // The key object typically has a single field (e.g. { repository: "repo/name" }).
  // Extract the first string value, falling back to JSON.
  const values = Object.values(key);
  if (values.length === 1 && typeof values[0] === "string") return values[0];
  return JSON.stringify(key);
}

function aggregateRows(result: QueryResult): RepoRow[] {
  // Each series corresponds to one repository. The series has 1 date and 1 value
  // (latest snapshot). Extract key → sum of y (should be a single value).
  const rows: RepoRow[] = result.series.map((s) => {
    const count = s.y.reduce<number>((acc, v) => acc + (typeof v === "number" ? v : 0), 0);
    return { repository: seriesKeyToString(s.key), count };
  });

  // Sort descending, take top 15
  rows.sort((a, b) => b.count - a.count);
  return rows.slice(0, TOP_N);
}

// ---------------------------------------------------------------------------
// ECharts option builder
// ---------------------------------------------------------------------------
function buildChartOption(rows: RepoRow[]): object {
  // Horizontal bar: yAxis is category (repo names), xAxis is value.
  // inverse: true puts the highest count at the top.
  const repoNames = rows.map((r) => r.repository);
  const counts = rows.map((r) => r.count);

  return {
    backgroundColor: "transparent",
    grid: {
      top: 12,
      right: 16,
      bottom: 20,
      left: 200,
      containLabel: false,
    },
    yAxis: {
      type: "category",
      data: repoNames,
      inverse: true,
      axisLabel: {
        fontSize: 11,
        color: CHART_COLORS.greyMuted,
        formatter: (v: string) => (v.length > 28 ? v.slice(0, 26) + "…" : v),
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    xAxis: {
      type: "value",
      axisLabel: { fontSize: 10, color: CHART_COLORS.greyMuted },
      splitLine: {
        lineStyle: { color: CHART_COLORS.greyBorder, type: "dashed" },
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      {
        type: "bar",
        data: counts,
        itemStyle: { color: CHART_COLORS.severityCritical },
        barWidth: "50%",
      },
    ],
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const arr = params as Array<{ axisValue: string; value: number }>;
        if (!arr.length) return "";
        const p = arr[0];
        return `<div style="font-size:11px;min-width:160px">
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
export function RepositoryTagHygiene() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    runQuery(REPO_HYGIENE_QUERY)
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

  const rows = result ? aggregateRows(result) : [];

  const footer = result
    ? `Top ${rows.length} repositories by critical CVE count at latest snapshot.`
    : undefined;

  // Decide what to render in the chart area
  let body: React.ReactNode;
  if (error) {
    body = (
      <div
        className="flex items-center justify-center h-[340px] text-sm"
        style={{ color: "var(--severity-critical)" }}
        role="alert"
      >
        Unable to load data: {error}
      </div>
    );
  } else if (result === null) {
    body = <ChartSkeleton />;
  } else if (rows.length === 0) {
    body = (
      <div
        className="flex items-center justify-center h-[340px] text-sm"
        style={{ color: "var(--fg-muted)" }}
      >
        No critical findings found.
      </div>
    );
  } else {
    body = (
      <ReactECharts
        option={buildChartOption(rows)}
        style={{ height: "340px", width: "100%" }}
        notMerge
        lazyUpdate={false}
      />
    );
  }

  return (
    <WidgetCard
      label="Repository Hygiene"
      title="Top repositories by open criticals"
      footer={footer}
    >
      {body}
    </WidgetCard>
  );
}
