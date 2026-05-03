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
    const last = series.y[series.y.length - 1];
    if (typeof last === "number") {
      total += last;
      hasData = true;
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
        {count !== null ? count.toLocaleString("en-GB") : "\u2014"}
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
type CountsState = Record<SeverityKey, number | null>;

const emptyCounts: CountsState = Object.fromEntries(
  SEVERITIES.map((s) => [s.key, null]),
) as CountsState;

export function FleetSeveritySnapshot() {
  const [counts, setCounts] = useState<CountsState>(emptyCounts);
  const [latestDate, setLatestDate] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const queries = SEVERITIES.map((s) => buildQuery(s.measure));
    const promises = queries.map((q) => runQuery(q));

    Promise.allSettled(promises).then((results) => {
      if (cancelled) return;

      const countsMap: CountsState = Object.fromEntries(
        SEVERITIES.map((s) => [s.key, null]),
      ) as CountsState;
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
