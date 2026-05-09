"use client";

import React, { useEffect, useState, useMemo, useCallback, useRef } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { WidgetCard } from "./WidgetCard";
import { Input } from "@/components/ui/input";
import { getFindings, getWorkloadCounts, getWorkloadsForCve } from "@/lib/api/client";
import type { FindingsResponse, WeightedCve } from "@/lib/api/client";
import { CHART_COLORS } from "@/lib/charts/defaults";
import { TABLE_DEFAULTS } from "@/lib/table.defaults";
import { loadWeights, saveWeights, DEFAULT_WEIGHTS, type WeightConfig } from "@/lib/weighted-weights";
import { useDrillFilter, DRILL_COLUMNS } from "@/lib/drill";
import { FilterChips } from "@/components/ui/FilterChips";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type FindingRow = FindingsResponse["rows"][number];
type GroupBy = "none" | "cve" | "image" | "package" | "weighted";

// Severity ordering for max-severity aggregation
const SEVERITY_ORDER: Record<string, number> = {
  Critical: 5,
  High: 4,
  Medium: 3,
  Low: 2,
  Negligible: 1,
};

function maxSeverity(severities: string[]): string {
  return severities.reduce((best, cur) => {
    return (SEVERITY_ORDER[cur] ?? 0) > (SEVERITY_ORDER[best] ?? 0) ? cur : best;
  }, severities[0] ?? "");
}

interface CveRow {
  cve_id: string;
  severity: string;
  affected_images: number;
  affected_packages: number;
  first_seen: string;
  last_seen: string;
}

interface ImageRow {
  image_name: string;
  total_findings: number;
  critical_count: number;
  high_count: number;
  distinct_cves: number;
  distinct_packages: number;
  last_seen: string;
}

interface PackageRow {
  package_name: string;
  total_findings: number;
  critical_count: number;
  high_count: number;
  distinct_cves: number;
  distinct_images: number;
}

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

interface WorkloadDetailRow {
  cluster_name: string;
  namespace_name: string;
  workload_type: string;
  workload_name: string;
  container_name: string;
  team_id: string | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SEVERITIES = ["All", "Critical", "High", "Medium", "Low", "Negligible"] as const;
const STATES = ["All", "OPEN", "CLOSED"] as const;
const GROUP_BY_OPTIONS: { value: GroupBy; label: string }[] = [
  { value: "none", label: "None" },
  { value: "cve", label: "CVE" },
  { value: "image", label: "Image" },
  { value: "package", label: "Package" },
  { value: "weighted", label: "Weighted" },
];
const LIMIT_OPTIONS = [25, 50, 100, 250, 500] as const;
type LimitOption = (typeof LIMIT_OPTIONS)[number];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function relativeTime(iso: string): string {
  const now = new Date();
  const then = new Date(iso);
  const hours = Math.floor((now.getTime() - then.getTime()) / 36e5);
  if (hours < 1) return "just now";
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

function SortIcon({ direction }: { direction: "asc" | "desc" | false }) {
  if (!direction) {
    return <span style={{ opacity: 0.3, fontSize: "10px", marginLeft: "4px" }}>⇅</span>;
  }
  return (
    <span style={{ fontSize: "10px", marginLeft: "4px" }}>
      {direction === "asc" ? "↑" : "↓"}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Severity pill
// ---------------------------------------------------------------------------
const SEVERITY_STYLES: Record<string, { background: string; color: string }> = {
  Critical:   { background: CHART_COLORS.severityCritical, color: "#fff" },
  High:       { background: CHART_COLORS.severityHigh,     color: "#000" },
  Medium:     { background: CHART_COLORS.severityMedium,   color: "#000" },
  Low:        { background: CHART_COLORS.severityLow,      color: "#000" },
  Negligible: { background: CHART_COLORS.greyMuted,        color: "#fff" },
};

function SeverityPill({ value }: { value: string }) {
  const style = SEVERITY_STYLES[value] ?? { background: CHART_COLORS.greyMuted, color: "#fff" };
  return (
    <span
      style={{
        background: style.background,
        color: style.color,
        paddingLeft: "4px",
        paddingRight: "4px",
        paddingTop: "2px",
        paddingBottom: "2px",
        fontSize: "9px",
        fontWeight: 600,
        textTransform: "uppercase",
        borderRadius: "3px",
        display: "inline-block",
        whiteSpace: "nowrap",
      }}
    >
      {value}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Column definitions — flat list
// ---------------------------------------------------------------------------
const FLAT_COLUMNS: ColumnDef<FindingRow>[] = [
  {
    accessorKey: "cve_id",
    header: "CVE",
    size: 140,
    minSize: 80,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
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
    accessorKey: "image_name",
    header: "Image",
    size: 220,
    minSize: 100,
    cell: (info) => {
      const val = info.getValue() as string | null;
      return (
        <span
          className="font-mono text-[11px] truncate block"
          title={val ?? ""}
          style={{ color: "var(--fg-primary)" }}
        >
          {val ?? "—"}
        </span>
      );
    },
  },
  {
    accessorKey: "package_name",
    header: "Package",
    size: 160,
    minSize: 80,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-muted)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "first_seen",
    header: "First seen",
    size: 100,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {relativeTime(String(info.getValue()))}
      </span>
    ),
  },
  {
    accessorKey: "last_seen",
    header: "Last seen",
    size: 100,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {relativeTime(String(info.getValue()))}
      </span>
    ),
  },
  {
    accessorKey: "state",
    header: "State",
    size: 80,
    minSize: 50,
    cell: (info) => {
      const val = String(info.getValue());
      const isOpen = val === "OPEN";
      return (
        <span
          className="text-[11px] font-medium"
          style={{ color: isOpen ? CHART_COLORS.severityCritical : "var(--fg-muted)" }}
        >
          {val}
        </span>
      );
    },
  },
  {
    accessorKey: "reason_code",
    header: "Reason",
    size: 110,
    minSize: 60,
    cell: (info) => {
      const val = info.getValue() as string | null;
      if (!val) return null;
      return (
        <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
          {val}
        </span>
      );
    },
  },
];

// ---------------------------------------------------------------------------
// Column definitions — grouped by CVE
// ---------------------------------------------------------------------------
const CVE_COLUMNS: ColumnDef<CveRow>[] = [
  {
    accessorKey: "cve_id",
    header: "CVE ID",
    size: 160,
    minSize: 100,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
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
    accessorKey: "affected_images",
    header: "Affected images",
    size: 130,
    minSize: 80,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-primary)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "affected_packages",
    header: "Affected packages",
    size: 140,
    minSize: 90,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "first_seen",
    header: "First seen",
    size: 100,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {relativeTime(String(info.getValue()))}
      </span>
    ),
  },
  {
    accessorKey: "last_seen",
    header: "Last seen",
    size: 100,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {relativeTime(String(info.getValue()))}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Column definitions — grouped by Image
// ---------------------------------------------------------------------------
const IMAGE_COLUMNS: ColumnDef<ImageRow>[] = [
  {
    accessorKey: "image_name",
    header: "Image",
    size: 260,
    minSize: 120,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-primary)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "total_findings",
    header: "Total findings",
    size: 120,
    minSize: 80,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-primary)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "critical_count",
    header: "Critical",
    size: 90,
    minSize: 60,
    cell: (info) => {
      const val = info.getValue() as number;
      return (
        <span
          className="text-[11px] font-semibold"
          style={{ color: val > 0 ? CHART_COLORS.severityCritical : "var(--fg-muted)" }}
        >
          {val.toLocaleString("en-GB")}
        </span>
      );
    },
  },
  {
    accessorKey: "high_count",
    header: "High",
    size: 80,
    minSize: 60,
    cell: (info) => {
      const val = info.getValue() as number;
      return (
        <span
          className="text-[11px] font-semibold"
          style={{ color: val > 0 ? CHART_COLORS.severityHigh : "var(--fg-muted)" }}
        >
          {val.toLocaleString("en-GB")}
        </span>
      );
    },
  },
  {
    accessorKey: "distinct_cves",
    header: "Distinct CVEs",
    size: 110,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "distinct_packages",
    header: "Distinct packages",
    size: 140,
    minSize: 90,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "last_seen",
    header: "Last seen",
    size: 100,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {relativeTime(String(info.getValue()))}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Column definitions — grouped by Package
// ---------------------------------------------------------------------------
const PACKAGE_COLUMNS: ColumnDef<PackageRow>[] = [
  {
    accessorKey: "package_name",
    header: "Package",
    size: 240,
    minSize: 100,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-primary)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "total_findings",
    header: "Total findings",
    size: 120,
    minSize: 80,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-primary)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "critical_count",
    header: "Critical",
    size: 90,
    minSize: 60,
    cell: (info) => {
      const val = info.getValue() as number;
      return (
        <span
          className="text-[11px] font-semibold"
          style={{ color: val > 0 ? CHART_COLORS.severityCritical : "var(--fg-muted)" }}
        >
          {val.toLocaleString("en-GB")}
        </span>
      );
    },
  },
  {
    accessorKey: "high_count",
    header: "High",
    size: 80,
    minSize: 60,
    cell: (info) => {
      const val = info.getValue() as number;
      return (
        <span
          className="text-[11px] font-semibold"
          style={{ color: val > 0 ? CHART_COLORS.severityHigh : "var(--fg-muted)" }}
        >
          {val.toLocaleString("en-GB")}
        </span>
      );
    },
  },
  {
    accessorKey: "distinct_cves",
    header: "Distinct CVEs",
    size: 110,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "distinct_images",
    header: "Distinct images",
    size: 120,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
];

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

// ---------------------------------------------------------------------------
// Column definitions — Workload detail (drill-in)
// ---------------------------------------------------------------------------
const WORKLOAD_DETAIL_COLUMNS: ColumnDef<WorkloadDetailRow>[] = [
  {
    accessorKey: "cluster_name",
    header: "Cluster",
    size: 160,
    minSize: 100,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-primary)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "namespace_name",
    header: "Namespace",
    size: 130,
    minSize: 80,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-muted)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "workload_name",
    header: "Workload",
    size: 160,
    minSize: 100,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-primary)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "workload_type",
    header: "Type",
    size: 110,
    minSize: 70,
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "container_name",
    header: "Container",
    size: 130,
    minSize: 80,
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block"
        title={String(info.getValue())}
        style={{ color: "var(--fg-muted)" }}
      >
        {String(info.getValue())}
      </span>
    ),
  },
  {
    accessorKey: "team_id",
    header: "Team",
    size: 110,
    minSize: 70,
    cell: (info) => {
      const val = info.getValue() as string | null;
      return (
        <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
          {val ?? "—"}
        </span>
      );
    },
  },
];

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------
function TableSkeleton() {
  return (
    <div
      className="w-full animate-pulse"
      style={{
        backgroundColor: "var(--bg-surface)",
        borderRadius: "var(--radius)",
        height: "360px",
      }}
      aria-label="Loading findings…"
      role="status"
    />
  );
}

// ---------------------------------------------------------------------------
// Dropdown helper
// ---------------------------------------------------------------------------
function FilterSelect({
  value,
  options,
  onChange,
  label,
}: {
  value: string;
  options: readonly string[];
  onChange: (v: string) => void;
  label: string;
}) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>{label}:</span>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-[11px] rounded px-2 py-1"
        style={{
          border: "1px solid var(--border-subtle)",
          background: "var(--bg-surface)",
          color: "var(--fg-primary)",
          cursor: "pointer",
        }}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Boolean toggle checkbox
// ---------------------------------------------------------------------------
function BooleanCheckbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label
      className="flex items-center gap-1 text-[11px] cursor-pointer select-none"
      style={{ color: "var(--fg-primary)" }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="rounded"
        style={{ accentColor: "var(--severity-high)" }}
      />
      {label}
    </label>
  );
}


// ---------------------------------------------------------------------------
// Column autosizing hook — measures content via DOM after data loads
// ---------------------------------------------------------------------------
function useColumnAutoSizing<T extends object>(
  table: {
    getHeaderGroups: () => unknown[];
    getRowModel: () => { rows: unknown[] };
    setColumnSizing: (s: Record<string, number>) => void;
  },
  data: T[],
  columns: unknown[]
) {
  const prevLenRef = useRef(0);

  useEffect(() => {
    if (!data.length) return;
    if (data.length === prevLenRef.current) return;
    prevLenRef.current = data.length;

    requestAnimationFrame(() => {
      const headerGroups = table.getHeaderGroups();
      if (!headerGroups.length) return;
      const headers = (headerGroups[0] as any).headers;
      const rows = table.getRowModel().rows;

      const sizing: Record<string, number> = {};
      const canvas = typeof document !== "undefined" ? document.createElement("canvas") : null;
      const ctx = canvas ? canvas.getContext("2d") : null;

      for (const header of headers) {
        const colId = header.id;
        const colDef = header.column.columnDef;
        const minSize = colDef.minSize ?? 60;
        const maxSize = colDef.maxSize ?? 400;

        let maxWidth = 0;

        // Measure header
        const headerText = typeof colDef.header === "string" ? colDef.header : "";
        if (ctx && headerText) {
          ctx.font = "10px system-ui, -apple-system, sans-serif";
          maxWidth = Math.max(maxWidth, ctx.measureText(headerText).width + 36);
        } else {
          maxWidth = Math.max(maxWidth, 60);
        }

        // Measure first 20 cells
        const sampleRows = rows.slice(0, 20);
        for (const row of sampleRows) {
          const cell = (row as any).getVisibleCells().find((c: any) => c.column.id === colId);
          if (!cell) continue;
          const val = String(cell.getValue());
          if (ctx && val) {
            const isMono = colDef.cell && String(colDef.cell).includes("font-mono");
            ctx.font = isMono ? "11px menlo, monospace" : "11px system-ui, -apple-system, sans-serif";
            maxWidth = Math.max(maxWidth, ctx.measureText(val).width + 36);
          }
        }

        sizing[colId] = Math.max(minSize, Math.min(maxSize, Math.ceil(maxWidth)));
      }

      console.log("[FindingsTable] autosized columns:", sizing);
      table.setColumnSizing(sizing);
    });
  }, [data, columns, table]);
}

// ---------------------------------------------------------------------------
// Generic table renderer with column resize support
// ---------------------------------------------------------------------------
function ResizableTable<T extends object>({
  data,
  columns,
  emptyMessage,
  colSpan,
}: {
  data: T[];
  columns: ColumnDef<T>[];
  emptyMessage: string;
  colSpan: number;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnSizing, setColumnSizing] = useState<Record<string, number>>(() => {
    const s: Record<string, number> = {};
    for (const c of columns) {
      const id = (c as { id?: string; accessorKey?: string }).id || (c as { accessorKey?: string }).accessorKey || "";
      if (id) s[id] = (c as { size?: number }).size ?? 120;
    }
    return s;
  });

  const table = useReactTable({
    ...TABLE_DEFAULTS,
    data,
    columns,
    state: { sorting, columnSizing },
    onSortingChange: setSorting,
    onColumnSizingChange: setColumnSizing,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    manualPagination: true,
  });

  useColumnAutoSizing(table, data, columns);

  return (
    <table className="w-full text-left border-collapse" style={{ tableLayout: "fixed" }}>
      <thead>
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th
                key={header.id}
                className="text-[10px] font-medium tracking-widest uppercase pb-2 pt-1 select-none"
                style={{
                  position: "relative",
                  width: header.getSize(),
                  color: "var(--fg-muted)",
                  cursor: header.column.getCanSort() ? "pointer" : "default",
                  borderBottom: "1px solid var(--border-subtle)",
                  paddingRight: "8px",
                  whiteSpace: "nowrap",
                }}
                onClick={header.column.getToggleSortingHandler()}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
                <SortIcon direction={header.column.getIsSorted()} />
                {header.column.getCanResize() && (
                  <div
                    onMouseDown={header.getResizeHandler()}
                    onTouchStart={header.getResizeHandler()}
                    style={{
                      position: "absolute",
                      right: 0,
                      top: 0,
                      height: "100%",
                      width: "5px",
                      cursor: "col-resize",
                      userSelect: "none",
                      touchAction: "none",
                    }}
                  />
                )}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.length === 0 ? (
          <tr>
            <td
              colSpan={colSpan}
              className="text-center text-[12px] py-6"
              style={{ color: "var(--fg-muted)" }}
            >
              {emptyMessage}
            </td>
          </tr>
        ) : (
          table.getRowModel().rows.map((row) => (
            <tr
              key={row.id}
              style={{ height: "36px" }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.backgroundColor = "var(--bg-surface)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
              }}
            >
              {row.getVisibleCells().map((cell) => (
                <td
                  key={cell.id}
                  style={{
                    paddingRight: "8px",
                    borderBottom: "1px solid var(--border-subtle)",
                    verticalAlign: "middle",
                    overflow: "hidden",
                  }}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))
        )}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// Drillable cell wrapper
// ---------------------------------------------------------------------------
function DrillableCell({
  accessorKey,
  value,
  children,
  onDrill,
}: {
  accessorKey: string;
  value: string | number;
  children: React.ReactNode;
  onDrill: (accessorKey: string, value: string) => void;
}) {
  const config = DRILL_COLUMNS[accessorKey];
  if (!config) {
    return <>{children}</>;
  }

  return (
    <span
      className="cursor-pointer underline decoration-dotted underline-offset-2"
      style={{ color: "var(--fg-primary)" }}
      onClick={(e) => {
        e.stopPropagation();
        onDrill(accessorKey, String(value));
      }}
      title={`Filter by ${config.field}: ${value}`}
    >
      {children}
    </span>
  );
}

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

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function FindingsTable() {
  // Server-side pagination / filter state
  const [severityFilter, setSeverityFilter] = useState<string>("All");
  const [stateFilter, setStateFilter] = useState<string>("OPEN");
  const [fixFilter, setFixFilter] = useState<boolean>(false);
  const [inUseFilter, setInUseFilter] = useState<boolean>(false);
  const [exploitFilter, setExploitFilter] = useState<boolean>(false);
  const [serverPage, setServerPage] = useState(0);
  const [limit, setLimit] = useState<LimitOption>(25);

  // Group-by state
  const [groupBy, setGroupBy] = useState<GroupBy>("none");

  // Client-side text search
  const [globalFilter, setGlobalFilter] = useState("");

  // Data state
  const [rows, setRows] = useState<FindingRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Flat-list sorting (only used in "none" mode)
  const [sorting, setSorting] = useState<SortingState>([
    { id: "last_seen", desc: true },
  ]);

  // Weighted scoring state
  const [weights, setWeights] = useState<WeightConfig>(DEFAULT_WEIGHTS);
  const [weightedCves, setWeightedCves] = useState<WeightedCve[]>([]);
  const [snapshotDate, setSnapshotDate] = useState<string>("");
  const [weightsLoading, setWeightsLoading] = useState(false);

  // Drill filter state (URL-driven)
  const { filter, applyFilter, setMode, clearFilter, isFiltered } = useDrillFilter();

  // Workload drill data
  const [workloadRows, setWorkloadRows] = useState<WorkloadDetailRow[]>([]);
  const [workloadLoading, setWorkloadLoading] = useState(false);
  const [workloadError, setWorkloadError] = useState<string | null>(null);

  // Load weights from localStorage on mount
  useEffect(() => {
    setWeights(loadWeights());
  }, []);

  // Save weights to localStorage when changed
  useEffect(() => {
    saveWeights(weights);
  }, [weights]);

  // Fetch all CVE scoring data when weighted mode is active
  useEffect(() => {
    if (groupBy !== "weighted") return;
    setWeightsLoading(true);
    getWorkloadCounts()
      .then((data) => {
        setWeightedCves(data.counts);
        setSnapshotDate(data.snapshot_date);
      })
      .catch((e) => console.error("Failed to load workload counts:", e))
      .finally(() => setWeightsLoading(false));
  }, [groupBy]);

  // Fetch workload details when in workload_drill mode
  useEffect(() => {
    if (filter.mode !== "workload_drill" || !filter.value) {
      setWorkloadRows([]);
      return;
    }
    setWorkloadLoading(true);
    setWorkloadError(null);
    getWorkloadsForCve(filter.value)
      .then((data) => {
        setWorkloadRows(data.workloads);
      })
      .catch((e: unknown) => {
        setWorkloadError(e instanceof Error ? e.message : "Failed to load workloads");
      })
      .finally(() => setWorkloadLoading(false));
  }, [filter.mode, filter.value]);

  // Reset page when drill filter changes
  useEffect(() => {
    if (isFiltered) {
      setServerPage(0);
    }
  }, [isFiltered, filter.field, filter.value]);

  // ---------------------------------------------------------------------------
  // Fetch
  // ---------------------------------------------------------------------------
  const fetchPage = useCallback(
    (page: number, sev: string, st: string, fix: boolean, inUse: boolean, exploit: boolean, pageLimit: number) => {
      setLoading(true);
      setError(null);
      getFindings({
        limit: pageLimit,
        offset: page * pageLimit,
        severity: sev === "All" ? undefined : sev,
        state: st === "All" ? undefined : st,
        fix_available: fix ? true : undefined,
        in_use: inUse ? true : undefined,
        public_exploit: exploit ? true : undefined,
      })
        .then((data) => {
          setRows(data.rows);
          setTotal(data.total);
          setLoading(false);
        })
        .catch((e: unknown) => {
          setError(e instanceof Error ? e.message : "Failed to load findings.");
          setLoading(false);
        });
    },
    []
  );

  useEffect(() => {
    fetchPage(serverPage, severityFilter, stateFilter, fixFilter, inUseFilter, exploitFilter, limit);
  }, [fetchPage, serverPage, severityFilter, stateFilter, fixFilter, inUseFilter, exploitFilter, limit]);

  // Reset to page 0 when filters/limit change
  const handleSeverityChange = (v: string) => {
    setSeverityFilter(v);
    setServerPage(0);
    setGlobalFilter("");
  };
  const handleStateChange = (v: string) => {
    setStateFilter(v);
    setServerPage(0);
    setGlobalFilter("");
  };
  const handleFixChange = (v: boolean) => {
    setFixFilter(v);
    setServerPage(0);
    setGlobalFilter("");
  };
  const handleInUseChange = (v: boolean) => {
    setInUseFilter(v);
    setServerPage(0);
    setGlobalFilter("");
  };
  const handleExploitChange = (v: boolean) => {
    setExploitFilter(v);
    setServerPage(0);
    setGlobalFilter("");
  };
  const handleLimitChange = (v: string) => {
    setLimit(Number(v) as LimitOption);
    setServerPage(0);
  };

  // Drill-in cell click handler
  const handleCellDrill = useCallback(
    (accessorKey: string, value: string) => {
      const config = DRILL_COLUMNS[accessorKey];
      if (!config) return;

      if (config.mode === "workload_drill") {
        // Handled specially in weighted columns
        return;
      }

      setGlobalFilter("");
      applyFilter(config.field, value);
    },
    [applyFilter]
  );

  // ---------------------------------------------------------------------------
  // Client-side text search filter on top of server page
  // ---------------------------------------------------------------------------
  const filteredRows = useMemo(() => {
    // Use drill filter value if active, otherwise use manual search
    const searchValue = (isFiltered && filter.mode !== "workload_drill") ? filter.value : globalFilter;
    if (!searchValue?.trim()) return rows;
    const q = searchValue.toLowerCase();
    return rows.filter(
      (r) =>
        r.cve_id.toLowerCase().includes(q) ||
        (r.image_name ?? "").toLowerCase().includes(q) ||
        r.package_name.toLowerCase().includes(q)
    );
  }, [rows, globalFilter, isFiltered, filter.value, filter.mode]);

  // ---------------------------------------------------------------------------
  // Aggregated rows — derived from filteredRows based on groupBy
  // ---------------------------------------------------------------------------
  const cveRows = useMemo<CveRow[]>(() => {
    if (groupBy !== "cve") return [];
    const map = new Map<string, { severities: string[]; images: Set<string>; packages: Set<string>; firstSeen: string; lastSeen: string }>();
    for (const row of filteredRows) {
      const key = row.cve_id;
      const existing = map.get(key);
      if (!existing) {
        map.set(key, {
          severities: [row.severity],
          images: new Set(row.image_name ? [row.image_name] : []),
          packages: new Set([row.package_name]),
          firstSeen: row.first_seen,
          lastSeen: row.last_seen,
        });
      } else {
        existing.severities.push(row.severity);
        if (row.image_name) existing.images.add(row.image_name);
        existing.packages.add(row.package_name);
        if (row.first_seen < existing.firstSeen) existing.firstSeen = row.first_seen;
        if (row.last_seen > existing.lastSeen) existing.lastSeen = row.last_seen;
      }
    }
    return Array.from(map.entries()).map(([cve_id, v]) => ({
      cve_id,
      severity: maxSeverity(v.severities),
      affected_images: v.images.size,
      affected_packages: v.packages.size,
      first_seen: v.firstSeen,
      last_seen: v.lastSeen,
    }));
  }, [filteredRows, groupBy]);

  const imageRows = useMemo<ImageRow[]>(() => {
    if (groupBy !== "image") return [];
    const map = new Map<string, { total: number; crit: number; high: number; cves: Set<string>; pkgs: Set<string>; lastSeen: string }>();
    for (const row of filteredRows) {
      const key = row.image_name ?? "(unknown)";
      const existing = map.get(key);
      if (!existing) {
        map.set(key, {
          total: 1,
          crit: row.severity === "Critical" ? 1 : 0,
          high: row.severity === "High" ? 1 : 0,
          cves: new Set([row.cve_id]),
          pkgs: new Set([row.package_name]),
          lastSeen: row.last_seen,
        });
      } else {
        existing.total += 1;
        if (row.severity === "Critical") existing.crit += 1;
        if (row.severity === "High") existing.high += 1;
        existing.cves.add(row.cve_id);
        existing.pkgs.add(row.package_name);
        if (row.last_seen > existing.lastSeen) existing.lastSeen = row.last_seen;
      }
    }
    return Array.from(map.entries()).map(([image_name, v]) => ({
      image_name,
      total_findings: v.total,
      critical_count: v.crit,
      high_count: v.high,
      distinct_cves: v.cves.size,
      distinct_packages: v.pkgs.size,
      last_seen: v.lastSeen,
    }));
  }, [filteredRows, groupBy]);

  const packageRows = useMemo<PackageRow[]>(() => {
    if (groupBy !== "package") return [];
    const map = new Map<string, { total: number; crit: number; high: number; cves: Set<string>; images: Set<string> }>();
    for (const row of filteredRows) {
      const key = row.package_name;
      const existing = map.get(key);
      if (!existing) {
        map.set(key, {
          total: 1,
          crit: row.severity === "Critical" ? 1 : 0,
          high: row.severity === "High" ? 1 : 0,
          cves: new Set([row.cve_id]),
          images: new Set(row.image_name ? [row.image_name] : []),
        });
      } else {
        existing.total += 1;
        if (row.severity === "Critical") existing.crit += 1;
        if (row.severity === "High") existing.high += 1;
        existing.cves.add(row.cve_id);
        if (row.image_name) existing.images.add(row.image_name);
      }
    }
    return Array.from(map.entries()).map(([package_name, v]) => ({
      package_name,
      total_findings: v.total,
      critical_count: v.crit,
      high_count: v.high,
      distinct_cves: v.cves.size,
      distinct_images: v.images.size,
    }));
  }, [filteredRows, groupBy]);

  const weightedRows = useMemo<WeightedRow[]>(() => {
    if (groupBy !== "weighted") return [];

    // Score each CVE from the backend-provided data
    const scored: WeightedRow[] = [];
    for (const cve of weightedCves) {
      const sevWeight = weights.weights[cve.severity as keyof typeof weights.weights] || 0;
      const flags = sevWeight
        + (cve.in_use ? weights.weights.in_use : 0)
        + (cve.fix_available ? weights.weights.fix_available : 0)
        + (cve.public_exploit ? weights.weights.public_exploit : 0);

      const score = flags * cve.workload_count;

      // Build breakdown string
      const parts: number[] = [];
      if (sevWeight > 0) parts.push(sevWeight);
      if (cve.in_use && weights.weights.in_use > 0) parts.push(weights.weights.in_use);
      if (cve.fix_available && weights.weights.fix_available > 0) parts.push(weights.weights.fix_available);
      if (cve.public_exploit && weights.weights.public_exploit > 0) parts.push(weights.weights.public_exploit);
      const breakdown = `(${parts.join(" + ")}) × ${cve.workload_count}`;

      scored.push({
        cve_id: cve.cve_id,
        severity: cve.severity,
        workload_count: cve.workload_count,
        in_use: cve.in_use,
        fix_available: cve.fix_available,
        public_exploit: cve.public_exploit,
        score,
        breakdown,
      });
    }

    // Filter by severity gate, sort by score descending
    const filtered = scored
      .filter(r => weights.severityGate.includes(r.severity))

    // Apply drill filter if active (e.g. clicked a CVE to narrow down)
    const drilled = isFiltered && filter.mode !== "workload_drill" && filter.field === "cve"
      ? filtered.filter(r => r.cve_id === filter.value)
      : filtered;

    const sorted = drilled.toSorted((a, b) => b.score - a.score);

    // Client-side pagination
    const start = serverPage * limit;
    return sorted.slice(start, start + limit);
  }, [weightedCves, groupBy, weights, serverPage, limit, isFiltered, filter.value, filter.field, filter.mode]);

  // Total count after severity gate and drill filter (for pagination footer)
  const weightedTotal = useMemo(() => {
    if (groupBy !== "weighted") return 0;
    let total = weightedCves.filter(c => weights.severityGate.includes(c.severity)).length;
    if (isFiltered && filter.mode !== "workload_drill" && filter.field === "cve") {
      total = weightedCves.filter(c => c.cve_id === filter.value).length;
    }
    return total;
  }, [weightedCves, groupBy, weights, isFiltered, filter.value, filter.field, filter.mode]);

  // ---------------------------------------------------------------------------
  // Dynamic column overrides — wrap drillable cells
  // ---------------------------------------------------------------------------
  const flatColumns = useMemo((): ColumnDef<FindingRow>[] => {
    return FLAT_COLUMNS.map((col) => {
      const accessorKey = (col as { accessorKey?: string }).accessorKey;

      // image_name is nullable, handle specially
      if (accessorKey === "image_name") {
        return {
          ...col,
          cell: (info: { getValue: () => unknown }) => {
            const val = info.getValue() as string | null;
            if (!val) {
              return (
                <span className="font-mono text-[11px] truncate block" style={{ color: "var(--fg-muted)" }}>
                  —
                </span>
              );
            }
            return (
              <DrillableCell accessorKey={accessorKey} value={val} onDrill={handleCellDrill}>
                <span
                  className="font-mono text-[11px] truncate block"
                  title={val}
                  style={{ color: "var(--fg-primary)" }}
                >
                  {val}
                </span>
              </DrillableCell>
            );
          },
        };
      }

      // For cve_id and package_name
      if (accessorKey === "cve_id" || accessorKey === "package_name") {
        const originalCell = (col as { cell?: (info: { getValue: () => unknown }) => React.ReactNode }).cell;
        return {
          ...col,
          cell: (info: { getValue: () => unknown }) => (
            <DrillableCell accessorKey={accessorKey} value={String(info.getValue())} onDrill={handleCellDrill}>
              {originalCell ? originalCell(info) : String(info.getValue())}
            </DrillableCell>
          ),
        };
      }

      return col;
    });
  }, [handleCellDrill]);

  const weightedColumns = useMemo((): ColumnDef<WeightedRow>[] => {
    return WEIGHTED_COLUMNS.map((col) => {
      const accessorKey = (col as { accessorKey?: string }).accessorKey;

      if (accessorKey === "cve_id") {
        return {
          ...col,
          cell: (info: { getValue: () => unknown }) => (
            <span
              className="font-mono text-[12px] truncate block cursor-pointer underline decoration-dotted underline-offset-2"
              title="Click to filter by this CVE"
              onClick={(e) => {
                e.stopPropagation();
                setGlobalFilter("");
                applyFilter("cve", String(info.getValue()));
              }}
              style={{ color: "var(--fg-primary)" }}
            >
              {String(info.getValue())}
            </span>
          ),
        };
      }

      if (accessorKey === "workload_count") {
        return {
          ...col,
          cell: (info: { getValue: () => unknown; row: { original: WeightedRow } }) => (
            <span
              className="text-[11px] font-semibold cursor-pointer underline decoration-dotted underline-offset-2"
              title="Click to see workloads running this CVE"
              onClick={(e) => {
                e.stopPropagation();
                const cveId = info.row.original.cve_id;
                applyFilter("cve", cveId, "workload_drill");
              }}
              style={{ color: "var(--fg-primary)" }}
            >
              {(info.getValue() as number).toLocaleString("en-GB")} ▸
            </span>
          ),
        };
      }

      return col;
    });
  }, [applyFilter, setMode]);

  // ---------------------------------------------------------------------------
  // Flat-list TanStack table (used only when groupBy === "none")
  // ---------------------------------------------------------------------------
  const [flatColumnSizing, setFlatColumnSizing] = useState<Record<string, number>>(() => {
    const s: Record<string, number> = {};
    for (const c of FLAT_COLUMNS) {
      const id = (c as { id?: string; accessorKey?: string }).id || (c as { accessorKey?: string }).accessorKey || "";
      if (id) s[id] = (c as { size?: number }).size ?? 120;
    }
    return s;
  });

  const flatTable = useReactTable({
    ...TABLE_DEFAULTS,
    data: filteredRows,
    columns: flatColumns,
    state: { sorting, columnSizing: flatColumnSizing },
    onSortingChange: setSorting,
    onColumnSizingChange: setFlatColumnSizing,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    manualPagination: true,
  });

  useColumnAutoSizing(flatTable, filteredRows, flatColumns);

  // ---------------------------------------------------------------------------
  // Pagination labels
  // ---------------------------------------------------------------------------
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const firstRow = serverPage * limit + 1;
  const lastRow = Math.min((serverPage + 1) * limit, total);

  // Empty-state message per groupBy mode
  const emptyMessages: Record<GroupBy, string> = {
    none: "No findings match your filters.",
    cve: "No CVEs match your filters.",
    image: "No images match your filters.",
    package: "No packages match your filters.",
    weighted: "No findings match your severity gate, or no workload data available.",
  };

  // ---------------------------------------------------------------------------
  // Render body
  // ---------------------------------------------------------------------------
  let body: React.ReactNode;

  if (error) {
    body = (
      <div
        className="flex items-center justify-center text-sm"
        style={{ color: "var(--severity-critical)", minHeight: "320px" }}
        role="alert"
      >
        Unable to load data: {error}
      </div>
    );
  } else if (loading) {
    body = <TableSkeleton />;
  } else {
    // Toolbar
    const toolbar = (
      <div className="flex flex-col gap-2">
        {/* Filter chips row */}
        <FilterChips
          filter={filter}
          onClear={clearFilter}
          onModeReset={filter.mode === "workload_drill" ? () => setMode("findings") : undefined}
        />
        <div className="flex flex-wrap gap-2 items-center">
        <Input
          placeholder="Search CVE, image, package…"
          value={isFiltered && filter.mode !== "workload_drill" ? filter.value ?? globalFilter : globalFilter}
          onChange={(e) => {
            // Typing clears any active drill filter
            if (isFiltered) {
              clearFilter();
            }
            setGlobalFilter(e.target.value);
          }}
          className="text-[12px] max-w-[240px]"
        />
        <FilterSelect
          label="Severity"
          value={severityFilter}
          options={SEVERITIES}
          onChange={handleSeverityChange}
        />
        <FilterSelect
          label="State"
          value={stateFilter}
          options={STATES}
          onChange={handleStateChange}
        />
        <BooleanCheckbox label="Has Fix" checked={fixFilter} onChange={handleFixChange} />
        <BooleanCheckbox label="In-use" checked={inUseFilter} onChange={handleInUseChange} />
        <BooleanCheckbox label="Has Exploit" checked={exploitFilter} onChange={handleExploitChange} />
        {/* Group-by */}
        <div className="flex items-center gap-1">
          <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>Group by:</span>
          <select
            aria-label="Group by"
            value={groupBy}
            onChange={(e) => setGroupBy(e.target.value as GroupBy)}
            className="text-[11px] rounded px-2 py-1"
            style={{
              border: "1px solid var(--border-subtle)",
              background: "var(--bg-surface)",
              color: "var(--fg-primary)",
              cursor: "pointer",
            }}
          >
            {GROUP_BY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        {/* Limit selector */}
        <div className="flex items-center gap-1">
          <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>Show:</span>
          <select
            aria-label="Row limit"
            value={String(limit)}
            onChange={(e) => handleLimitChange(e.target.value)}
            className="text-[11px] rounded px-2 py-1"
            style={{
              border: "1px solid var(--border-subtle)",
              background: "var(--bg-surface)",
              color: "var(--fg-primary)",
              cursor: "pointer",
            }}
          >
            {LIMIT_OPTIONS.map((n) => (
              <option key={n} value={String(n)}>
                {n} rows
              </option>
            ))}
          </select>
        </div>
        </div>
        {/* Weight config panel — shows when weighted mode is active AND not in drill mode */}
        {groupBy === "weighted" && filter.mode !== "workload_drill" && (
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
      </div>
    );

    // Table area — switches based on drill mode and groupBy
    let tableArea: React.ReactNode;

    if (filter.mode === "workload_drill") {
      // Workload detail view
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          {workloadLoading ? (
            <div className="text-center text-[12px] py-6 animate-pulse" style={{ color: "var(--fg-muted)" }}>
              Loading workloads…
            </div>
          ) : workloadError ? (
            <div className="text-center text-[12px] py-6" style={{ color: "var(--severity-critical)" }}>
              {workloadError}
            </div>
          ) : (
            <ResizableTable
              data={workloadRows}
              columns={WORKLOAD_DETAIL_COLUMNS}
              emptyMessage={`No workloads found running images affected by ${filter.value}`}
              colSpan={WORKLOAD_DETAIL_COLUMNS.length}
            />
          )}
        </div>
      );
    } else if (groupBy === "weighted") {
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          {weightsLoading ? (
            <div className="text-center text-[12px] py-6 animate-pulse" style={{ color: "var(--fg-muted)" }}>
              Loading workload data…
            </div>
          ) : (
            <>
              <ResizableTable
                data={weightedRows}
                columns={weightedColumns}
                emptyMessage="No findings match your severity gate, or no workload data available."
                colSpan={weightedColumns.length}
              />
              {snapshotDate && (
                <p className="text-[10px] mt-1" style={{ color: "var(--fg-muted)" }}>
                  Workload data from snapshot {snapshotDate}
                </p>
              )}
            </>
          )}
        </div>
      );
    } else if (groupBy === "none") {
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          <table className="w-full text-left border-collapse" style={{ tableLayout: "fixed" }}>
            <thead>
              {flatTable.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="text-[10px] font-medium tracking-widest uppercase pb-2 pt-1 select-none"
                      style={{
                        position: "relative",
                        width: header.getSize(),
                        color: "var(--fg-muted)",
                        cursor: header.column.getCanSort() ? "pointer" : "default",
                        borderBottom: "1px solid var(--border-subtle)",
                        paddingRight: "8px",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      <SortIcon direction={header.column.getIsSorted()} />
                      {header.column.getCanResize() && (
                        <div
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          style={{
                            position: "absolute",
                            right: 0,
                            top: 0,
                            height: "100%",
                            width: "5px",
                            cursor: "col-resize",
                            userSelect: "none",
                            touchAction: "none",
                          }}
                        />
                      )}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {flatTable.getRowModel().rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={FLAT_COLUMNS.length}
                    className="text-center text-[12px] py-6"
                    style={{ color: "var(--fg-muted)" }}
                  >
                    {emptyMessages.none}
                  </td>
                </tr>
              ) : (
                flatTable.getRowModel().rows.map((row) => (
                  <tr
                    key={row.id}
                    style={{ height: "36px" }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.backgroundColor = "var(--bg-surface)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
                    }}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <td
                        key={cell.id}
                        style={{
                          paddingRight: "8px",
                          borderBottom: "1px solid var(--border-subtle)",
                          verticalAlign: "middle",
                          overflow: "hidden",
                        }}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      );
    } else if (groupBy === "cve") {
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          <ResizableTable
            data={cveRows}
            columns={CVE_COLUMNS}
            emptyMessage={emptyMessages.cve}
            colSpan={CVE_COLUMNS.length}
          />
        </div>
      );
    } else if (groupBy === "image") {
      tableArea = (
        <div style={{ overflowX: "auto" }}>
          <ResizableTable
            data={imageRows}
            columns={IMAGE_COLUMNS}
            emptyMessage={emptyMessages.image}
            colSpan={IMAGE_COLUMNS.length}
          />
        </div>
      );
    } else if (groupBy === "package") {
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

    // Pagination footer
    const weightedTotalPages = Math.max(1, Math.ceil(weightedTotal / limit));
    const weightedFirstRow = serverPage * limit + 1;
    const weightedLastRow = Math.min((serverPage + 1) * limit, weightedTotal);

    const paginationFooter = filter.mode === "workload_drill" ? (
      <p className="text-[10px] pt-1" style={{ color: "var(--fg-muted)" }}>
        {workloadRows.length.toLocaleString("en-GB")} workloads running images affected by {filter.value}
      </p>
    ) : groupBy === "none" ? (
      <div className="flex items-center justify-between pt-1">
        <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
          {total === 0
            ? "No findings"
            : `Showing ${firstRow}–${lastRow} of ${total.toLocaleString("en-GB")} findings`}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => setServerPage((p) => Math.max(0, p - 1))}
            disabled={serverPage === 0}
            className="text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
            style={{
              border: "1px solid var(--border-subtle)",
              color: "var(--fg-primary)",
              cursor: serverPage === 0 ? "not-allowed" : "pointer",
            }}
          >
            Prev
          </button>
          <span className="text-[11px] self-center" style={{ color: "var(--fg-muted)" }}>
            {serverPage + 1} / {totalPages}
          </span>
          <button
            onClick={() => setServerPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={serverPage >= totalPages - 1}
            className="text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
            style={{
              border: "1px solid var(--border-subtle)",
              color: "var(--fg-primary)",
              cursor: serverPage >= totalPages - 1 ? "not-allowed" : "pointer",
            }}
          >
            Next
          </button>
        </div>
      </div>
    ) : groupBy === "weighted" ? (
      <div className="flex items-center justify-between pt-1">
        <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
          {weightedTotal === 0
            ? "No CVEs match your severity gate"
            : `Showing ${weightedFirstRow}–${weightedLastRow} of ${weightedTotal.toLocaleString("en-GB")} CVEs`}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => setServerPage((p) => Math.max(0, p - 1))}
            disabled={serverPage === 0}
            className="text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
            style={{
              border: "1px solid var(--border-subtle)",
              color: "var(--fg-primary)",
              cursor: serverPage === 0 ? "not-allowed" : "pointer",
            }}
          >
            Prev
          </button>
          <span className="text-[11px] self-center" style={{ color: "var(--fg-muted)" }}>
            {serverPage + 1} / {weightedTotalPages}
          </span>
          <button
            onClick={() => setServerPage((p) => Math.min(weightedTotalPages - 1, p + 1))}
            disabled={serverPage >= weightedTotalPages - 1}
            className="text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
            style={{
              border: "1px solid var(--border-subtle)",
              color: "var(--fg-primary)",
              cursor: serverPage >= weightedTotalPages - 1 ? "not-allowed" : "pointer",
            }}
          >
            Next
          </button>
        </div>
      </div>
    ) : (
      <p className="text-[10px] pt-1" style={{ color: "var(--fg-muted)" }}>
        Group-by aggregates the currently loaded {rows.length.toLocaleString("en-GB")} rows. To see all data grouped, use a higher limit.
      </p>
    );

    body = (
      <div className="flex flex-col gap-2">
        {toolbar}
        {tableArea}
        {paginationFooter}
      </div>
    );
  }

  // Reset button — always visible, outside loading/error conditional
  const resetButton = (
    <button
      onClick={() => {
        clearFilter();
        setGlobalFilter("");
        setSeverityFilter("All");
        setStateFilter("OPEN");
        setFixFilter(false);
        setInUseFilter(false);
        setExploitFilter(false);
        setGroupBy("none");
        setServerPage(0);
      }}
      className="text-[11px] px-2 py-0.5 rounded"
      style={{
        border: "1px solid var(--border-subtle)",
        background: "var(--bg-surface)",
        color: "var(--fg-muted)",
        cursor: "pointer",
      }}
      title="Reset all filters and group-by to defaults"
    >
      Reset
    </button>
  );

  return (
    <WidgetCard label="All Findings" title="Findings list">
      <div className="flex items-center justify-between mb-2">
        <div /> {/* spacer for alignment */}
        {resetButton}
      </div>
      {body}
    </WidgetCard>
  );
}
