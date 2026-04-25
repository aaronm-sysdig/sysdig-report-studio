"use client";

import { useEffect, useState, useRef, useCallback } from "react";
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
  /** If provided, skips the image picker and renders the story for this image directly. */
  imageId?: string;
}

interface ImageEntity {
  id: string;
  label: string;
  repository: string;
  tag: string;
  last_seen: string;
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

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------
const COMMON_TIME: QueryIn["time"] = {
  mode: "last_n_snapshots",
  n: 90,
  granularity: "day",
};

function imageQuery(measure: string, imageId: string): QueryIn {
  return {
    lens: "Image",
    traversal: [],
    time: COMMON_TIME,
    measure,
    filters: [{ field: "image_id", operator: "eq", value: imageId }],
    group_by: [],
    order_by: null,
    limit: null,
  };
}

/** Extract a single series (the image we filtered to) into {dates, values} */
function extractSeries(result: QueryResult): { dates: string[]; values: number[] } {
  if (!result.series.length) return { dates: [], values: [] };
  // With an image_id filter there should be exactly 1 series — take it
  const s = result.series[0];
  return { dates: s.x as string[], values: s.y as number[] };
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
): object {
  const { dates, critical, high, medium, low, totals } = data;

  return {
    backgroundColor: "transparent",
    grid: standardGrid(axisLabels ? 32 : 4),
    xAxis: standardXAxis(dates, axisLabels),
    yAxis: STANDARD_Y_AXIS,
    series: [
      // Stacked bars — order: Negligible/Low at bottom, Critical at top
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
      // Mirror the main chart x-axis exactly so columns align
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
        itemStyle: { color: CHART_COLORS.severityCritical },
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
// Reason-code decomposition bar — segmented by patched / retired / accepted / other
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
        No findings closed in this window for this image.
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
      {/* Header label */}
      <div
        className="text-[10px] font-medium tracking-widest uppercase mb-1.5"
        style={{ color: "var(--fg-muted)" }}
      >
        Why {total.toLocaleString("en-GB")} closed?
      </div>

      {/* Segmented bar */}
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

      {/* Legend */}
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
// Auto-narrative
// ---------------------------------------------------------------------------
interface ReasonTotals {
  patched: number;
  retired: number;
  accepted: number;
  other: number;
}

function buildNarrative(metrics: HeadlineMetrics, reason?: ReasonTotals): string {
  const { totalNew, totalFixed, totalRegressed } = metrics;
  const netImprovement = totalFixed - totalNew;

  if (totalNew === 0 && totalFixed === 0 && totalRegressed === 0) {
    return "In the last 90 days, this image's posture has been unchanged — no new findings, no closures, no regressions.";
  }

  if (totalRegressed > totalFixed && totalFixed < 2) {
    return `In the last 90 days, ${totalRegressed.toLocaleString("en-GB")} finding${totalRegressed !== 1 ? "s" : ""} regressed for this image whilst only ${totalFixed.toLocaleString("en-GB")} ${totalFixed !== 1 ? "were" : "was"} closed. Worth a look.`;
  }

  // Per-reason narrative when breakdown data is available
  if (reason && totalFixed > 0) {
    const { patched, retired, accepted } = reason;
    const totalClosed = patched + retired + accepted + reason.other;

    if (totalClosed > 0) {
      // Patched dominates — real engineering work
      if (patched > retired + accepted + reason.other) {
        return `In the last 90 days, this image has been getting better — ${patched.toLocaleString("en-GB")} findings patched (real fixes)${retired > 0 ? `, ${retired.toLocaleString("en-GB")} retired` : ""}.`;
      }

      // Retired dominates — image churn rather than genuine fixing
      if (retired > patched) {
        return `In the last 90 days, ${retired.toLocaleString("en-GB")} findings disappeared because the image was retired, vs only ${patched.toLocaleString("en-GB")} actually patched. Worth investigating.`;
      }

      // Accepted is meaningful (>10% of closed)
      if (accepted > 0 && accepted / totalClosed > 0.1) {
        return `In the last 90 days, ${accepted.toLocaleString("en-GB")} findings were risk-accepted whilst only ${patched.toLocaleString("en-GB")} were patched.`;
      }
    }
  }

  if (netImprovement > 0 && totalFixed > 2) {
    return `In the last 90 days, this image has been getting better — ${totalFixed.toLocaleString("en-GB")} findings closed${totalRegressed > 0 ? `, ${totalRegressed.toLocaleString("en-GB")} regressed` : ", no regressions"}.`;
  }

  if (totalNew > totalFixed && totalNew > 5) {
    return `In the last 90 days, ${totalNew.toLocaleString("en-GB")} new findings appeared on this image whilst only ${totalFixed.toLocaleString("en-GB")} ${totalFixed !== 1 ? "were" : "was"} closed — the backlog is growing.`;
  }

  return `In the last 90 days, ${totalNew.toLocaleString("en-GB")} new, ${totalFixed.toLocaleString("en-GB")} fixed, and ${totalRegressed.toLocaleString("en-GB")} regressed for this image.`;
}

// ---------------------------------------------------------------------------
// Tag lineage panel
// ---------------------------------------------------------------------------
interface TagEntry {
  id: string;
  label: string;
  repository: string;
  tag: string;
  last_seen: string;
  criticalCount: number | null;
}

interface TagLineagePanelProps {
  selectedId: string;
  allImages: ImageEntity[];
  repository: string;
}

function TagLineagePanel({ selectedId, allImages, repository }: TagLineagePanelProps) {
  const [tags, setTags] = useState<TagEntry[] | null>(null);

  useEffect(() => {
    if (!repository) return;

    // Filter all images to the same repo
    const repoImages = allImages
      .filter((img) => img.repository === repository)
      .sort((a, b) => a.last_seen.localeCompare(b.last_seen));

    if (repoImages.length === 0) {
      setTags([]);
      return;
    }

    // Query count_open_critical for each image in the repo concurrently
    const queries = repoImages.map((img) =>
      runQuery({
        lens: "Image",
        traversal: [],
        time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
        measure: "count_open_critical",
        filters: [{ field: "image_id", operator: "eq", value: img.id }],
        group_by: [],
        order_by: null,
        limit: null,
      })
        .then((result) => {
          const s = result.series[0];
          const val = s ? (s.y[s.y.length - 1] as number | null) : null;
          return { ...img, criticalCount: typeof val === "number" ? val : null };
        })
        .catch(() => ({ ...img, criticalCount: null })),
    );

    Promise.all(queries).then(setTags);
  }, [repository, allImages, selectedId]);

  const maxCritical = tags
    ? Math.max(1, ...tags.map((t) => t.criticalCount ?? 0))
    : 1;

  if (!tags) {
    return (
      <div className="flex flex-col gap-2">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-[52px] animate-pulse rounded"
            style={{ backgroundColor: "var(--bg-surface)" }}
          />
        ))}
      </div>
    );
  }

  if (tags.length <= 1) {
    return (
      <p
        className="text-[11px] italic"
        style={{ color: "var(--fg-muted)" }}
      >
        This image is the only tag in its repository — no tag lineage to compare.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2 overflow-y-auto" style={{ maxHeight: "420px" }}>
      {tags.map((t) => {
        const isCurrent = t.id === selectedId;
        const barPct =
          t.criticalCount !== null
            ? Math.max(4, Math.round((t.criticalCount / maxCritical) * 100))
            : 0;

        return (
          <div
            key={t.id}
            className="flex flex-col gap-1 rounded p-2"
            style={{
              backgroundColor: isCurrent
                ? "var(--bg-surface)"
                : "transparent",
              border: isCurrent
                ? `1px solid ${CHART_COLORS.severityCritical}`
                : "1px solid transparent",
              cursor: "default",
            }}
          >
            <div className="flex items-center justify-between gap-1">
              <span
                className="text-[11px] font-medium truncate"
                style={{ color: isCurrent ? "var(--fg-primary)" : "var(--fg-muted)" }}
                title={t.tag}
              >
                {t.tag || t.label}
              </span>
              {isCurrent && (
                <span
                  className="text-[9px] font-bold uppercase tracking-widest px-1 rounded"
                  style={{
                    backgroundColor: CHART_COLORS.severityCritical,
                    color: CHART_COLORS.white,
                    flexShrink: 0,
                  }}
                >
                  current
                </span>
              )}
            </div>
            {/* Critical bar */}
            <div
              className="w-full rounded-sm overflow-hidden"
              style={{ height: 4, backgroundColor: "var(--border-subtle)" }}
            >
              <div
                style={{
                  width: `${barPct}%`,
                  height: "100%",
                  backgroundColor: CHART_COLORS.severityCritical,
                  borderRadius: 2,
                }}
              />
            </div>
            <span
              className="text-[9px]"
              style={{ color: "var(--fg-muted)" }}
            >
              {t.criticalCount !== null
                ? `${t.criticalCount.toLocaleString("en-GB")} critical`
                : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Image picker
// ---------------------------------------------------------------------------
interface ImagePickerProps {
  images: ImageEntity[];
  selectedId: string;
  onSelect: (id: string) => void;
}

function ImagePicker({ images, selectedId, onSelect }: ImagePickerProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const filtered = images
    .filter((img) =>
      img.label.toLowerCase().includes(query.toLowerCase()),
    )
    .slice(0, 10);

  const selectedImage = images.find((img) => img.id === selectedId);

  const handleSelect = useCallback(
    (id: string) => {
      onSelect(id);
      const img = images.find((i) => i.id === id);
      setQuery(img?.label ?? "");
      setOpen(false);
    },
    [images, onSelect],
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
        value={open ? query : (selectedImage?.label ?? query)}
        placeholder="Search images…"
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
          {filtered.map((img) => (
            <button
              key={img.id}
              className="w-full text-left px-3 py-2 text-[12px] hover:bg-muted transition-colors"
              style={{ color: "var(--fg-primary)" }}
              onMouseDown={(e) => {
                e.preventDefault(); // prevent onBlur
                handleSelect(img.id);
              }}
            >
              <span className="font-medium">{img.label}</span>
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
export function ImageRemediationStory({ imageId: externalImageId }: ImageRemediationStoryProps) {
  const [allImages, setAllImages] = useState<ImageEntity[] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(externalImageId ?? null);
  const [data, setData] = useState<RemediationData | null>(null);
  const [metrics, setMetrics] = useState<HeadlineMetrics | null>(null);
  const [reasonTotals, setReasonTotals] = useState<ReasonTotals | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [axisLabels, setAxisLabels] = useState(false);

  // Load entity list (always needed — for picker and for tag lineage)
  useEffect(() => {
    getEntities("Image")
      .then((raw) => {
        const imgs = raw as ImageEntity[];
        const sorted = [...imgs].sort((a, b) => a.label.localeCompare(b.label));
        setAllImages(sorted);
        // Auto-select first image when no external imageId provided
        if (!externalImageId && sorted.length > 0) {
          setSelectedId(sorted[0].id);
        }
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load images.");
      });
  }, [externalImageId]);

  // Load chart data whenever selected image changes
  useEffect(() => {
    if (!selectedId) return;

    setData(null);
    setMetrics(null);
    setReasonTotals(null);
    setError(null);

    let cancelled = false;

    Promise.all([
      runQuery(imageQuery("count_open_critical", selectedId)),
      runQuery(imageQuery("count_open_high", selectedId)),
      runQuery(imageQuery("count_open_medium", selectedId)),
      runQuery(imageQuery("count_open_low", selectedId)),
      runQuery(imageQuery("count_new", selectedId)),
      runQuery(imageQuery("count_fixed", selectedId)),
      runQuery(imageQuery("count_regressed", selectedId)),
      runQuery(imageQuery("count_fixed_patched", selectedId)),
      runQuery(imageQuery("count_fixed_retired", selectedId)),
      runQuery(imageQuery("count_fixed_accepted", selectedId)),
      runQuery(imageQuery("count_fixed_other", selectedId)),
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

          // Align everything to a shared date axis
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

          // Skip the first data point on new/fixed/regressed — first snapshot counts every finding as "new"
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

          // Headline metrics
          const currentOpen = slicedTotals[slicedTotals.length - 1] ?? 0;
          const totalNew = slicedNew.reduce((a, b) => a + b, 0);
          const totalFixed = slicedFixed.reduce((a, b) => a + b, 0);
          const totalRegressed = slicedRegressed.reduce((a, b) => a + b, 0);

          // Delta vs 7 days ago
          let deltaVs7Days: number | null = null;
          if (slicedTotals.length >= 8) {
            const sevenDaysAgo = slicedTotals[slicedTotals.length - 8];
            deltaVs7Days = currentOpen - sevenDaysAgo;
          }

          setMetrics({ currentOpen, totalNew, totalFixed, totalRegressed, deltaVs7Days });

          // Reason-code totals: sum the full y-arrays
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
  }, [selectedId]);

  // Derived values
  const selectedImage = allImages?.find((img) => img.id === selectedId) ?? null;
  const widgetTitle = selectedImage
    ? `Image Remediation Story — ${selectedImage.label}`
    : "Image Remediation Story";
  const narrative = metrics ? buildNarrative(metrics, reasonTotals ?? undefined) : undefined;

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
        {/* ── Image picker (only when no external imageId) ── */}
        {!externalImageId && (
          <div style={{ height: 36 }}>
            {allImages === null ? (
              <div
                className="h-[36px] w-full animate-pulse rounded"
                style={{ backgroundColor: "var(--bg-surface)" }}
              />
            ) : (
              <ImagePicker
                images={allImages}
                selectedId={selectedId ?? ""}
                onSelect={setSelectedId}
              />
            )}
          </div>
        )}

        {/* ── Error state ── */}
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
            {/* ── Headline metrics strip ── */}
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

            {/* ── Main visualisation grid ── */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 200px",
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
                      option={buildMainChartOption(data, axisLabels)}
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
                        false, // no axis labels on flow chart — shares same x
                      )}
                      style={{ height: "100px", width: "100%" }}
                      notMerge
                      lazyUpdate={false}
                    />
                  )}
                </div>

                {/* Reason-code decomposition bar */}
                <div
                  className="flex items-center"
                  style={{ height: 72 }}
                >
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

              {/* RIGHT: tag lineage panel — spans full height */}
              <div
                className="flex flex-col"
                style={{ paddingTop: 0 }}
              >
                <span
                  className="text-[10px] font-medium tracking-widest uppercase mb-2"
                  style={{ color: "var(--fg-muted)" }}
                >
                  Tag Lineage
                </span>

                {allImages === null || selectedId === null ? (
                  <div className="flex flex-col gap-2">
                    {[1, 2, 3].map((i) => (
                      <div
                        key={i}
                        className="h-[52px] animate-pulse rounded"
                        style={{ backgroundColor: "var(--bg-surface)" }}
                      />
                    ))}
                  </div>
                ) : (
                  <TagLineagePanel
                    selectedId={selectedId}
                    allImages={allImages}
                    repository={selectedImage?.repository ?? ""}
                  />
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </WidgetCard>
  );
}
