"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { WidgetCard } from "./WidgetCard";
import { runQuery, getEntities } from "@/lib/api/client";
import type { QueryIn, QueryResult } from "@/lib/api/client";
import { Input } from "@/components/ui/input";
import {
  CHART_COLORS,
  flowingLineSeries,
  standardGrid,
  standardXAxis,
  STANDARD_Y_AXIS,
  STANDARD_TOOLTIP_STYLE,
} from "@/lib/charts/defaults";

// ECharts must be loaded client-side only
const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface ImageRemediationStoryProps {
  /**
   * If provided, skips the repository picker and renders the story for this
   * repository directly. Format: "owner/name" e.g. "aaronmsysdig/sysdig-notifier"
   */
  repository?: string;
}

interface ImageEntity {
  id: string;
  label: string;
  repository: string;
  tag: string;
  last_seen: string;
  first_seen?: string;
}

interface RepositoryEntity {
  repository: string;
  imageCount: number;
}

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
}

interface ReasonTotals {
  patched: number;
  retired: number;
  accepted: number;
  other: number;
}

interface TagGenealogyRow {
  imageId: string;
  tag: string;
  digestPrefix: string;
  firstSeen: string;
  critical: number | null;
  high: number | null;
  isCurrent: boolean;
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------
const COMMON_TIME: QueryIn["time"] = {
  mode: "last_n_snapshots",
  n: 90,
  granularity: "day",
};

function repoQuery(measure: string, imageIds: string[]): QueryIn {
  return {
    lens: "Image",
    traversal: [],
    time: COMMON_TIME,
    measure,
    filters: [{ field: "image_id", operator: "in", value: imageIds }],
    group_by: [],
    order_by: null,
    limit: null,
  };
}

/** Sum across ALL series from a QueryResult into {dates, values}.
 *
 * When a query filters by image_id IN [...], the rollup path returns one
 * series per image_id — NOT a pre-aggregated single series. We must sum
 * across all series at each date to get the repository-wide total.
 */
function extractSeries(result: QueryResult): { dates: string[]; values: number[] } {
  if (!result.series.length) return { dates: [], values: [] };

  // Collect all dates across every series
  const dateSet = new Set<string>();
  for (const s of result.series) {
    for (const d of s.x as string[]) dateSet.add(d);
  }
  const dates = Array.from(dateSet).sort();

  // Sum values across all series at each date
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
// Skeleton shimmer
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
// Stat card
// ---------------------------------------------------------------------------
interface StatCardProps {
  label: string;
  value: number;
  delta?: { value: number; label: string; positive: boolean } | null;
}

function StatCard({ label, value, delta }: StatCardProps) {
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
function buildMainChartOption(
  data: RemediationData,
  axisLabels: boolean,
  tagRows?: TagGenealogyRow[] | null,
): object {
  const { dates, critical, high, medium, low, totals } = data;

  // Tag markers — vertical dashed lines with rounded-pill labels
  const tagMarkLines = tagRows && tagRows.length > 0
    ? tagRows
        .filter((r) => r.firstSeen)
        .map((r) => {
          const isCurrent = r.isCurrent;
          return {
            xAxis: r.firstSeen.slice(0, 10),
            label: {
              formatter: r.tag || r.digestPrefix.slice(0, 8),
              position: "insideStartTop" as const,
              fontSize: 12,
              fontWeight: isCurrent ? "bold" : "normal",
              color: isCurrent ? "#ffffff" : CHART_COLORS.greyMuted,
              backgroundColor: isCurrent ? CHART_COLORS.deepSee : "var(--bg-surface)",
              borderColor: isCurrent ? CHART_COLORS.deepSee : CHART_COLORS.greyBorder,
              borderWidth: 1,
              borderRadius: 10,
              padding: [2, 6, 2, 6],
              // slight shadow so pill lifts off chart bg
              shadowBlur: isCurrent ? 4 : 0,
              shadowColor: "rgba(0,0,0,0.15)",
            },
            lineStyle: {
              color: isCurrent ? CHART_COLORS.deepSee : CHART_COLORS.greyBorder,
              type: (isCurrent ? "solid" : "dashed") as "solid" | "dashed",
              width: isCurrent ? 1.5 : 1,
              opacity: isCurrent ? 0.6 : 0.4,
            },
          };
        })
    : [];

  return {
    backgroundColor: "transparent",
    grid: standardGrid(axisLabels ? 32 : 4),
    xAxis: standardXAxis(dates, axisLabels),
    yAxis: STANDARD_Y_AXIS,
    series: [
      {
        type: "bar",
        name: "Low",
        data: low,
        stack: "severity",
        itemStyle: { color: CHART_COLORS.severityLow },
        barMaxWidth: 16,
      },
      {
        type: "bar",
        name: "Medium",
        data: medium,
        stack: "severity",
        itemStyle: { color: CHART_COLORS.severityMedium },
        barMaxWidth: 16,
      },
      {
        type: "bar",
        name: "High",
        data: high,
        stack: "severity",
        itemStyle: { color: CHART_COLORS.severityHigh },
        barMaxWidth: 16,
      },
      {
        type: "bar",
        name: "Critical",
        data: critical,
        stack: "severity",
        itemStyle: { color: CHART_COLORS.severityCritical },
        barMaxWidth: 16,
        // Tag markers drawn on this series (topmost bar = best z-order)
        markLine: tagMarkLines.length > 0
          ? { silent: false, symbol: ["none", "none"], data: tagMarkLines }
          : undefined,
      },
      // Flowing total line on top — grey context line
      {
        ...flowingLineSeries({ color: CHART_COLORS.greyMuted, width: 1.5, symbolSize: 3 }),
        name: "Total",
        data: totals,
        z: 10,
      },
    ],
    tooltip: {
      trigger: "axis",
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const arr = params as Array<{
          axisValue: string;
          seriesName: string;
          value: number;
          color: string;
        }>;
        if (!arr.length) return "";
        const date = arr[0].axisValue;
        const rows = arr
          .filter((p) => p.seriesName !== "Total")
          .map(
            (p) =>
              `<div style="display:flex;justify-content:space-between;gap:12px">` +
              `<span style="color:${p.color}">&#9632;</span>` +
              `<span style="color:${CHART_COLORS.greyMuted};flex:1;margin-left:4px">${p.seriesName}:</span>` +
              `<b>${(p.value ?? 0).toLocaleString("en-GB")}</b></div>`,
          );
        const totalEntry = arr.find((p) => p.seriesName === "Total");
        const total = totalEntry ? totalEntry.value : 0;
        return `<div style="font-size:11px;min-width:160px">
          <div style="color:${CHART_COLORS.greyMuted};margin-bottom:4px">${date}</div>
          ${rows.join("")}
          <div style="border-top:1px solid ${CHART_COLORS.greyBorder};margin-top:4px;padding-top:4px;display:flex;justify-content:space-between">
            <span style="color:${CHART_COLORS.greyMuted}">Total:</span>
            <b>${total.toLocaleString("en-GB")}</b>
          </div>
        </div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// ECharts: net daily flow chart (new above / fixed below)
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
    xAxis: {
      ...standardXAxis(dates, axisLabels),
    },
    yAxis: {
      type: "value" as const,
      minInterval: 1,
      axisLabel: {
        fontSize: 9,
        color: CHART_COLORS.greyMuted,
        formatter: (v: number) => (v === 0 ? "0" : v > 0 ? `+${v}` : String(v)),
      },
      splitLine: {
        lineStyle: { color: CHART_COLORS.greyBorder, type: "dashed" as const },
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      {
        type: "bar",
        name: "New",
        data: newCounts,
        itemStyle: { color: CHART_COLORS.darkRed },
        barMaxWidth: 16,
      },
      {
        type: "bar",
        name: "Closed",
        data: fixedCounts.map((v) => -v),
        itemStyle: { color: CHART_COLORS.fixedGreen },
        barMaxWidth: 16,
      },
    ],
    tooltip: {
      trigger: "axis",
      ...STANDARD_TOOLTIP_STYLE,
      formatter: (params: unknown[]) => {
        const arr = params as Array<{
          axisValue: string;
          seriesName: string;
          value: number;
        }>;
        if (!arr.length) return "";
        const date = arr[0].axisValue;
        const lines = arr.map((p) => {
          const v = Math.abs(p.value ?? 0);
          const col =
            p.seriesName === "New"
              ? CHART_COLORS.severityCritical
              : CHART_COLORS.fixedGreen;
          return `<div style="display:flex;justify-content:space-between;gap:12px">
            <span style="color:${col}">${p.seriesName}:</span>
            <b>${v.toLocaleString("en-GB")}</b></div>`;
        });
        return `<div style="font-size:11px;min-width:120px">
          <div style="color:${CHART_COLORS.greyMuted};margin-bottom:4px">${date}</div>
          ${lines.join("")}
        </div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Reason-code decomposition bar
// ---------------------------------------------------------------------------
interface ReasonBarProps {
  patched: number;
  retired: number;
  accepted: number;
  other: number;
}

function ReasonCodeBar({ patched, retired, accepted, other }: ReasonBarProps) {
  const total = patched + retired + accepted + other;

  if (total === 0) {
    return (
      <div className="text-xs italic" style={{ color: "var(--fg-muted)" }}>
        No findings closed in this window for this repository.
      </div>
    );
  }

  const segments = [
    { label: "PATCHED",  count: patched,  color: CHART_COLORS.fixedGreen },
    { label: "RETIRED",  count: retired,  color: CHART_COLORS.greyMuted },
    { label: "ACCEPTED", count: accepted, color: CHART_COLORS.severityMedium },
    { label: "OTHER",    count: other,    color: CHART_COLORS.greyBorder },
  ].filter((s) => s.count > 0);

  return (
    <div>
      <div
        className="text-[10px] font-medium tracking-widest uppercase mb-1.5"
        style={{ color: "var(--fg-muted)" }}
      >
        Why {total.toLocaleString("en-GB")} closed?
      </div>

      <div
        className="flex w-full h-7 overflow-hidden"
        style={{ borderRadius: "var(--radius)" }}
      >
        {segments.map((s) => (
          <div
            key={s.label}
            style={{
              width: `${(s.count / total) * 100}%`,
              backgroundColor: s.color,
              minWidth: 24,
            }}
            className="flex items-center justify-center text-[10px] font-semibold text-white"
            title={`${s.label}: ${s.count}`}
          >
            {((s.count / total) * 100) >= 8 && `${s.count}`}
          </div>
        ))}
      </div>

      <div
        className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]"
        style={{ color: "var(--fg-muted)" }}
      >
        {segments.map((s) => (
          <div key={s.label} className="flex items-center gap-1">
            <span
              className="inline-block w-2 h-2 rounded-sm"
              style={{ backgroundColor: s.color }}
            />
            <span>{s.label} {s.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auto-narrative — repository-centric
// ---------------------------------------------------------------------------
function buildNarrative(
  repoName: string,
  metrics: HeadlineMetrics,
  tagCount: number,
  latestTag: string | null,
  reason?: ReasonTotals,
): string {
  const { totalNew, totalFixed, totalRegressed } = metrics;

  if (totalNew === 0 && totalFixed === 0 && totalRegressed === 0) {
    return `The \`${repoName}\` repository spans ${tagCount} tag${tagCount !== 1 ? "s" : ""} — no new findings, closures, or regressions recorded in the last 90 days.`;
  }

  if (totalRegressed > totalFixed && totalFixed < 2) {
    return `The \`${repoName}\` repository (${tagCount} tag${tagCount !== 1 ? "s" : ""}) regressed ${totalRegressed.toLocaleString("en-GB")} finding${totalRegressed !== 1 ? "s" : ""} whilst only ${totalFixed.toLocaleString("en-GB")} ${totalFixed !== 1 ? "were" : "was"} closed across all tags. Worth a look.`;
  }

  if (reason && totalFixed > 0) {
    const { patched, retired, accepted } = reason;
    const totalClosed = patched + retired + accepted + reason.other;

    if (totalClosed > 0) {
      if (patched > retired + accepted + reason.other) {
        return `The \`${repoName}\` repository spans ${tagCount} tag${tagCount !== 1 ? "s" : ""} — in the last 90 days ${patched.toLocaleString("en-GB")} findings were patched (real fixes)${retired > 0 ? `, ${retired.toLocaleString("en-GB")} retired` : ""}${latestTag ? ` with \`${latestTag}\` as the most recent tag` : ""}.`;
      }

      if (retired > patched) {
        return `The \`${repoName}\` repository spans ${tagCount} tag${tagCount !== 1 ? "s" : ""} — ${retired.toLocaleString("en-GB")} findings disappeared via image retirement vs only ${patched.toLocaleString("en-GB")} actually patched. Worth investigating.`;
      }

      if (accepted > 0 && accepted / totalClosed > 0.1) {
        return `The \`${repoName}\` repository spans ${tagCount} tag${tagCount !== 1 ? "s" : ""} — ${accepted.toLocaleString("en-GB")} findings were risk-accepted whilst only ${patched.toLocaleString("en-GB")} were patched across all tags.`;
      }
    }
  }

  const netImprovement = totalFixed - totalNew;

  if (netImprovement > 0 && totalFixed > 2) {
    return `The \`${repoName}\` repository (${tagCount} tag${tagCount !== 1 ? "s" : ""}) has been improving — ${totalFixed.toLocaleString("en-GB")} findings closed${totalRegressed > 0 ? `, ${totalRegressed.toLocaleString("en-GB")} regressed` : ", no regressions"} in the last 90 days.`;
  }

  if (totalNew > totalFixed && totalNew > 5) {
    return `The \`${repoName}\` repository spans ${tagCount} tag${tagCount !== 1 ? "s" : ""} — ${totalNew.toLocaleString("en-GB")} new findings appeared whilst only ${totalFixed.toLocaleString("en-GB")} ${totalFixed !== 1 ? "were" : "was"} closed in the last 90 days. The backlog is growing.`;
  }

  return `The \`${repoName}\` repository spans ${tagCount} tag${tagCount !== 1 ? "s" : ""} — ${totalNew.toLocaleString("en-GB")} new, ${totalFixed.toLocaleString("en-GB")} fixed, and ${totalRegressed.toLocaleString("en-GB")} regressed across all tags in the last 90 days.`;
}

// ---------------------------------------------------------------------------
// Tag Genealogy Panel
// ---------------------------------------------------------------------------
interface TagGenealogyPanelProps {
  rows: TagGenealogyRow[] | null;
}

function TagGenealogyPanel({ rows }: TagGenealogyPanelProps) {
  if (rows === null) {
    return (
      <div className="flex flex-col gap-2">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-[68px] animate-pulse rounded"
            style={{ backgroundColor: "var(--bg-surface)" }}
          />
        ))}
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <p className="text-[11px] italic" style={{ color: "var(--fg-muted)" }}>
        No tags found for this repository.
      </p>
    );
  }

  // Single shared denominator across both severity tiers and all rows so bars
  // are comparable at a glance — Critical=2 vs High=17 will look proportionally correct.
  const maxAny = Math.max(
    1,
    ...rows.flatMap((r) => [r.critical ?? 0, r.high ?? 0])
  );

  return (
    <div
      className="relative flex flex-col overflow-y-auto"
      style={{ maxHeight: "460px", paddingLeft: "18px" }}
    >
      {/* Vertical connector line */}
      <div
        style={{
          position: "absolute",
          left: 6,
          top: 16,
          bottom: 16,
          width: 2,
          backgroundColor: "var(--border-subtle)",
          borderRadius: 1,
        }}
      />

      {rows.map((row, idx) => {
        const critPct =
          row.critical !== null && row.critical > 0
            ? Math.max(4, Math.round((row.critical / maxAny) * 100))
            : 0;
        const highPct =
          row.high !== null && row.high > 0
            ? Math.max(4, Math.round((row.high / maxAny) * 100))
            : 0;
        const bothZero = (row.critical ?? 0) === 0 && (row.high ?? 0) === 0;

        return (
          <div
            key={row.imageId}
            className="relative flex flex-col gap-1 rounded p-2 mb-2"
            style={{
              backgroundColor: row.isCurrent ? "var(--bg-surface)" : "transparent",
              border: row.isCurrent
                ? `1px solid ${CHART_COLORS.severityCritical}`
                : "1px solid transparent",
            }}
          >
            {/* Dot on the connector line */}
            <div
              style={{
                position: "absolute",
                left: -15,
                top: 14,
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: row.isCurrent
                  ? CHART_COLORS.severityCritical
                  : "var(--border-strong)",
                border: "2px solid var(--bg-base)",
                zIndex: 1,
              }}
            />

            {/* Row header: date + tag pill + CURRENT pill */}
            <div className="flex items-center justify-between gap-1">
              <div className="flex flex-col gap-0.5 min-w-0">
                <span className="text-[9px]" style={{ color: "var(--fg-muted)" }}>
                  {row.firstSeen ? row.firstSeen.slice(0, 10) : "—"}
                </span>
                {/* Tag pill — coloured rounded rectangle */}
                <span
                  className="text-[10px] font-semibold px-2 py-0.5 rounded-full truncate"
                  style={{
                    backgroundColor: row.isCurrent
                      ? CHART_COLORS.deepSee
                      : "var(--bg-surface)",
                    color: row.isCurrent ? "#ffffff" : "var(--fg-primary)",
                    border: `1px solid ${row.isCurrent ? CHART_COLORS.deepSee : "var(--border-subtle)"}`,
                    display: "inline-block",
                    maxWidth: "140px",
                  }}
                  title={row.tag}
                >
                  {row.tag || "(untagged)"}
                </span>
              </div>
              {row.isCurrent && (
                <span
                  className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded"
                  style={{
                    backgroundColor: `${CHART_COLORS.lumin}33`,
                    color: CHART_COLORS.deepSee,
                    border: `1px solid ${CHART_COLORS.lumin}`,
                    flexShrink: 0,
                  }}
                >
                  current
                </span>
              )}
            </div>

            {/* Digest prefix */}
            <span className="text-[9px] font-mono" style={{ color: "var(--fg-muted)" }}>
              {row.digestPrefix}
            </span>

            {/* Severity bars — fixed 100px track so bars never overflow */}
            {bothZero ? (
              <span className="text-[9px] italic" style={{ color: "var(--fg-muted)" }}>
                0 Critical, 0 High
              </span>
            ) : (
              <div className="flex flex-col gap-0.5">
                {/* Critical bar */}
                {row.critical !== null && (
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] shrink-0" style={{ color: "var(--fg-muted)", width: 28 }}>
                      Crit
                    </span>
                    {/* Fixed-width track — bar fills proportionally within 100px */}
                    <div style={{ width: 100, height: 6, backgroundColor: "var(--bg-surface)", borderRadius: 3, flexShrink: 0, overflow: "hidden" }}>
                      <div style={{ width: `${critPct}px`, height: "100%", backgroundColor: CHART_COLORS.severityCritical, borderRadius: 3 }} />
                    </div>
                    <span className="text-[9px] tabular-nums shrink-0" style={{ color: CHART_COLORS.severityCritical }}>
                      {(row.critical ?? 0).toLocaleString("en-GB")}
                    </span>
                  </div>
                )}

                {/* High bar */}
                {row.high !== null && (
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] shrink-0" style={{ color: "var(--fg-muted)", width: 28 }}>
                      High
                    </span>
                    <div style={{ width: 100, height: 6, backgroundColor: "var(--bg-surface)", borderRadius: 3, flexShrink: 0, overflow: "hidden" }}>
                      <div style={{ width: `${highPct}px`, height: "100%", backgroundColor: CHART_COLORS.severityHigh, borderRadius: 3 }} />
                    </div>
                    <span className="text-[9px] tabular-nums shrink-0" style={{ color: CHART_COLORS.severityHigh }}>
                      {(row.high ?? 0).toLocaleString("en-GB")}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Separator except after last row */}
            {idx < rows.length - 1 && (
              <div
                style={{
                  position: "absolute",
                  bottom: -1,
                  left: 8,
                  right: 8,
                  height: 1,
                  backgroundColor: "var(--border-subtle)",
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Repository picker
// ---------------------------------------------------------------------------
interface RepositoryPickerProps {
  repositories: RepositoryEntity[];
  selectedRepo: string;
  onSelect: (repo: string) => void;
}

function RepositoryPicker({ repositories, selectedRepo, onSelect }: RepositoryPickerProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const filtered = repositories
    .filter((r) =>
      r.repository.toLowerCase().includes(query.toLowerCase()),
    )
    .slice(0, 10);

  const handleSelect = useCallback(
    (repo: string) => {
      onSelect(repo);
      setQuery(repo);
      setOpen(false);
    },
    [onSelect],
  );

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={containerRef} className="relative w-full" style={{ maxWidth: 480 }}>
      <Input
        value={open ? query : (selectedRepo || query)}
        placeholder="Search repositories…"
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => {
          setQuery("");
          setOpen(true);
        }}
        className="h-[36px] text-[12px]"
      />
      {open && filtered.length > 0 && (
        <div
          className="absolute left-0 right-0 z-50 rounded shadow-lg overflow-y-auto"
          style={{
            top: "calc(100% + 4px)",
            maxHeight: "260px",
            backgroundColor: "var(--bg-base)",
            border: "1px solid var(--border-strong)",
          }}
        >
          {filtered.map((r) => (
            <button
              key={r.repository}
              className="w-full text-left px-3 py-2 text-[12px] hover:bg-muted transition-colors"
              style={{ color: "var(--fg-primary)" }}
              onMouseDown={(e) => {
                e.preventDefault();
                handleSelect(r.repository);
              }}
            >
              <span className="font-medium">{r.repository}</span>
              <span
                className="ml-2 text-[10px]"
                style={{ color: "var(--fg-muted)" }}
              >
                {r.imageCount} tag{r.imageCount !== 1 ? "s" : ""}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function ImageRemediationStory({ repository: externalRepo }: ImageRemediationStoryProps) {
  const [allImages, setAllImages] = useState<ImageEntity[] | null>(null);
  const [selectedRepo, setSelectedRepo] = useState<string | null>(externalRepo ?? null);
  const [data, setData] = useState<RemediationData | null>(null);
  const [metrics, setMetrics] = useState<HeadlineMetrics | null>(null);
  const [reasonTotals, setReasonTotals] = useState<ReasonTotals | null>(null);
  const [genealogyRows, setGenealogyRows] = useState<TagGenealogyRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [axisLabels, setAxisLabels] = useState(false);

  // Load entity list
  useEffect(() => {
    getEntities("Image")
      .then((raw) => {
        const imgs = raw as ImageEntity[];
        setAllImages(imgs);
        // Auto-select first repository alphabetically when no external repo provided
        if (!externalRepo && imgs.length > 0) {
          const repos = Array.from(new Set(imgs.map((i) => i.repository || i.id))).sort();
          if (repos.length > 0) setSelectedRepo(repos[0]);
        }
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load images.");
      });
  }, [externalRepo]);

  // Build unique repository list from allImages
  const repositories = useMemo<RepositoryEntity[]>(() => {
    if (!allImages) return [];
    const seen = new Set<string>();
    const out: RepositoryEntity[] = [];
    for (const img of allImages) {
      const repo = img.repository || img.id;
      if (!seen.has(repo)) {
        seen.add(repo);
        out.push({ repository: repo, imageCount: 1 });
      } else {
        const existing = out.find((r) => r.repository === repo);
        if (existing) existing.imageCount++;
      }
    }
    out.sort((a, b) => a.repository.localeCompare(b.repository));
    return out;
  }, [allImages]);

  // Images belonging to the selected repository
  const imagesInRepo = useMemo<ImageEntity[]>(() => {
    if (!allImages || !selectedRepo) return [];
    return allImages.filter((img) => (img.repository || img.id) === selectedRepo);
  }, [allImages, selectedRepo]);

  const imageIds = useMemo<string[]>(() => imagesInRepo.map((img) => img.id), [imagesInRepo]);

  // Load chart data whenever selected repository changes
  useEffect(() => {
    if (!selectedRepo || imageIds.length === 0) return;

    setData(null);
    setMetrics(null);
    setReasonTotals(null);
    setGenealogyRows(null);
    setError(null);

    let cancelled = false;

    Promise.all([
      runQuery(repoQuery("count_open_critical", imageIds)),
      runQuery(repoQuery("count_open_high", imageIds)),
      runQuery(repoQuery("count_open_medium", imageIds)),
      runQuery(repoQuery("count_open_low", imageIds)),
      runQuery(repoQuery("count_new", imageIds)),
      runQuery(repoQuery("count_fixed", imageIds)),
      runQuery(repoQuery("count_regressed", imageIds)),
      runQuery(repoQuery("count_fixed_patched", imageIds)),
      runQuery(repoQuery("count_fixed_retired", imageIds)),
      runQuery(repoQuery("count_fixed_accepted", imageIds)),
      runQuery(repoQuery("count_fixed_other", imageIds)),
    ])
      .then(
        ([
          critResult, highResult, medResult, lowResult,
          newResult, fixResult, regResult,
          patchedResult, retiredResult, acceptedResult, otherResult,
        ]) => {
          if (cancelled) return;

          const critSeries = extractSeries(critResult);
          const highSeries = extractSeries(highResult);
          const medSeries = extractSeries(medResult);
          const lowSeries = extractSeries(lowResult);
          const newSeries = extractSeries(newResult);
          const fixSeries = extractSeries(fixResult);
          const regSeries = extractSeries(regResult);

          const { dates, aligned } = alignToSharedDates([
            critSeries,
            highSeries,
            medSeries,
            lowSeries,
            newSeries,
            fixSeries,
            regSeries,
          ]);

          const [critical, high, medium, low, newCounts, fixedCounts, regressedCounts] = aligned;
          const totals = dates.map((_, i) => critical[i] + high[i] + medium[i] + low[i]);

          // Skip the first data point — first snapshot counts everything as "new"
          const sliceFrom = newCounts.findIndex((v) => v === 0) === 0 ? 1 : 0;
          const slicedDates = dates.slice(sliceFrom);
          const slicedCritical = critical.slice(sliceFrom);
          const slicedHigh = high.slice(sliceFrom);
          const slicedMedium = medium.slice(sliceFrom);
          const slicedLow = low.slice(sliceFrom);
          const slicedTotals = totals.slice(sliceFrom);
          const slicedNew = newCounts.slice(sliceFrom);
          const slicedFixed = fixedCounts.slice(sliceFrom);
          const slicedRegressed = regressedCounts.slice(sliceFrom);

          setData({
            dates: slicedDates,
            critical: slicedCritical,
            high: slicedHigh,
            medium: slicedMedium,
            low: slicedLow,
            totals: slicedTotals,
            newCounts: slicedNew,
            fixedCounts: slicedFixed,
            regressedCounts: slicedRegressed,
          });

          const currentOpen = slicedTotals[slicedTotals.length - 1] ?? 0;
          const totalNew = slicedNew.reduce((a, b) => a + b, 0);
          const totalFixed = slicedFixed.reduce((a, b) => a + b, 0);
          const totalRegressed = slicedRegressed.reduce((a, b) => a + b, 0);

          let deltaVs7Days: number | null = null;
          if (slicedTotals.length >= 8) {
            const sevenDaysAgo = slicedTotals[slicedTotals.length - 8];
            deltaVs7Days = currentOpen - sevenDaysAgo;
          }

          setMetrics({ currentOpen, totalNew, totalFixed, totalRegressed, deltaVs7Days });

          const sumSeries = (r: QueryResult) => {
            const s = r.series[0];
            if (!s) return 0;
            return (s.y as number[]).reduce((a, b) => a + (b ?? 0), 0);
          };
          setReasonTotals({
            patched:  sumSeries(patchedResult),
            retired:  sumSeries(retiredResult),
            accepted: sumSeries(acceptedResult),
            other:    sumSeries(otherResult),
          });
        },
      )
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load chart data.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRepo, imageIds]);

  // Load tag genealogy data using Option B (single batch query with group_by)
  useEffect(() => {
    if (!selectedRepo || imagesInRepo.length === 0) return;

    let cancelled = false;
    setGenealogyRows(null);

    const ids = imagesInRepo.map((img) => img.id);

    // Determine which image is "current" (latest first_seen / last_seen)
    const sorted = [...imagesInRepo].sort((a, b) =>
      (b.first_seen ?? b.last_seen ?? "").localeCompare(a.first_seen ?? a.last_seen ?? ""),
    );
    const currentImageId = sorted[0]?.id ?? null;

    // Two batch queries: critical and high, both grouped by image_id
    Promise.all([
      runQuery({
        lens: "Image",
        traversal: [],
        time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
        measure: "count_open_critical",
        filters: [{ field: "image_id", operator: "in", value: ids }],
        group_by: ["image_id"],
        order_by: null,
        limit: null,
      }),
      runQuery({
        lens: "Image",
        traversal: [],
        time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
        measure: "count_open_high",
        filters: [{ field: "image_id", operator: "in", value: ids }],
        group_by: ["image_id"],
        order_by: null,
        limit: null,
      }),
    ])
      .then(([critResult, highResult]) => {
        if (cancelled) return;

        // Build lookup maps from image_id -> latest value
        const critMap = new Map<string, number>();
        for (const s of critResult.series) {
          const imgId = (s.key as Record<string, string>)?.image_id;
          if (imgId) {
            const arr = s.y as number[];
            critMap.set(imgId, arr[arr.length - 1] ?? 0);
          }
        }

        const highMap = new Map<string, number>();
        for (const s of highResult.series) {
          const imgId = (s.key as Record<string, string>)?.image_id;
          if (imgId) {
            const arr = s.y as number[];
            highMap.set(imgId, arr[arr.length - 1] ?? 0);
          }
        }

        // Build rows sorted oldest → newest (conventional genealogy order)
        const rows: TagGenealogyRow[] = imagesInRepo
          .slice()
          .sort((a, b) =>
            (a.first_seen ?? a.last_seen ?? "").localeCompare(b.first_seen ?? b.last_seen ?? ""),
          )
          .map((img) => ({
            imageId: img.id,
            tag: img.tag || img.label || "(untagged)",
            digestPrefix: img.id.length > 19 ? img.id.slice(0, 19) : img.id,
            firstSeen: img.first_seen ?? img.last_seen ?? "",
            critical: critMap.get(img.id) ?? null,
            high: highMap.get(img.id) ?? null,
            isCurrent: img.id === currentImageId,
          }));

        setGenealogyRows(rows);
      })
      .catch(() => {
        if (!cancelled) {
          // Fall back to showing rows without counts rather than breaking the whole widget
          const rows: TagGenealogyRow[] = imagesInRepo
            .slice()
            .sort((a, b) =>
              (a.first_seen ?? a.last_seen ?? "").localeCompare(b.first_seen ?? b.last_seen ?? ""),
            )
            .map((img) => ({
              imageId: img.id,
              tag: img.tag || img.label || "(untagged)",
              digestPrefix: img.id.length > 19 ? img.id.slice(0, 19) : img.id,
              firstSeen: img.first_seen ?? img.last_seen ?? "",
              critical: null,
              high: null,
              isCurrent: img.id === currentImageId,
            }));
          setGenealogyRows(rows);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRepo, imagesInRepo]);

  // Derived values
  const tagCount = imagesInRepo.length;
  const latestTag = genealogyRows
    ? (genealogyRows[genealogyRows.length - 1]?.tag ?? null)
    : null;

  const widgetTitle = selectedRepo
    ? `Image Remediation — ${selectedRepo}`
    : "Image Remediation Story";

  const narrative =
    metrics && selectedRepo
      ? buildNarrative(selectedRepo, metrics, tagCount, latestTag, reasonTotals ?? undefined)
      : undefined;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <WidgetCard
      label="Remediation"
      title={widgetTitle}
      footer={narrative}
      axisLabels={axisLabels}
      onAxisLabelsChange={setAxisLabels}
    >
      <div className="flex flex-col gap-3">
        {/* Repository picker (only when no external repo prop) */}
        {!externalRepo && (
          <div style={{ height: 36 }}>
            {allImages === null ? (
              <div
                className="h-[36px] w-full animate-pulse rounded"
                style={{ backgroundColor: "var(--bg-surface)" }}
              />
            ) : (
              <RepositoryPicker
                repositories={repositories}
                selectedRepo={selectedRepo ?? ""}
                onSelect={setSelectedRepo}
              />
            )}
          </div>
        )}

        {/* Error state */}
        {error && (
          <div
            className="flex items-center justify-center py-6 text-sm"
            style={{ color: "var(--severity-critical)" }}
            role="alert"
          >
            Unable to load data: {error}
          </div>
        )}

        {!error && (
          <>
            {/* Headline metrics strip */}
            <div className="flex gap-2">
              <StatCard
                label="Current Open"
                value={metrics?.currentOpen ?? 0}
                delta={
                  metrics?.deltaVs7Days !== null && metrics?.deltaVs7Days !== undefined
                    ? {
                        value: metrics.deltaVs7Days,
                        label: "vs 7 days ago",
                        positive: metrics.deltaVs7Days <= 0,
                      }
                    : null
                }
              />
              <StatCard
                label="90-Day New"
                value={metrics?.totalNew ?? 0}
                delta={null}
              />
              <StatCard
                label="90-Day Fixed"
                value={metrics?.totalFixed ?? 0}
                delta={null}
              />
              <StatCard
                label="90-Day Regressed"
                value={metrics?.totalRegressed ?? 0}
                delta={null}
              />
            </div>

            {/* Main visualisation grid */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 280px",
                gap: "0 16px",
              }}
            >
              {/* LEFT: charts stacked vertically */}
              <div className="flex flex-col" style={{ gap: 6 }}>
                {/* Main severity-stack chart */}
                <div style={{ height: 280 }}>
                  {data === null ? (
                    <ChartSkeleton height={280} />
                  ) : (
                    <ReactECharts
                      option={buildMainChartOption(data, axisLabels, genealogyRows)}
                      style={{ height: "280px", width: "100%" }}
                      notMerge
                      lazyUpdate={false}
                    />
                  )}
                </div>

                {/* Net daily flow chart */}
                <div style={{ height: 100 }}>
                  {data === null ? (
                    <ChartSkeleton height={100} />
                  ) : (
                    <ReactECharts
                      option={buildFlowChartOption(
                        data.dates,
                        data.newCounts,
                        data.fixedCounts,
                        false,
                      )}
                      style={{ height: "100px", width: "100%" }}
                      notMerge
                      lazyUpdate={false}
                    />
                  )}
                </div>

                {/* Reason-code decomposition bar */}
                <div className="flex items-center" style={{ height: 72 }}>
                  {data === null || reasonTotals === null ? (
                    <ChartSkeleton height={72} />
                  ) : (
                    <div className="w-full">
                      <ReasonCodeBar
                        patched={reasonTotals.patched}
                        retired={reasonTotals.retired}
                        accepted={reasonTotals.accepted}
                        other={reasonTotals.other}
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* RIGHT: tag genealogy panel — spans full height */}
              <div className="flex flex-col" style={{ paddingTop: 0 }}>
                <span
                  className="text-[10px] font-medium tracking-widest uppercase mb-2"
                  style={{ color: "var(--fg-muted)" }}
                >
                  Tag Lineage
                </span>

                <TagGenealogyPanel rows={genealogyRows} />
              </div>
            </div>
          </>
        )}
      </div>
    </WidgetCard>
  );
}
