"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type ColumnResizeMode,
} from "@tanstack/react-table";
import { WidgetCard } from "./WidgetCard";
import { Input } from "@/components/ui/input";
import { getFindings } from "@/lib/api/client";
import type { FindingsResponse } from "@/lib/api/client";
import { CHART_COLORS } from "@/lib/charts/defaults";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type FindingRow = FindingsResponse["rows"][number];
type GroupBy = "none" | "cve" | "image" | "package";

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
];
const LIMIT_OPTIONS = [50, 100, 250, 500] as const;
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
  );
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
  const columnResizeMode: ColumnResizeMode = "onChange";

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    enableColumnResizing: true,
    columnResizeMode,
    manualPagination: true,
  });

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
// Component
// ---------------------------------------------------------------------------
export function FindingsTable() {
  // Server-side pagination / filter state
  const [severityFilter, setSeverityFilter] = useState<string>("All");
  const [stateFilter, setStateFilter] = useState<string>("All");
  const [serverPage, setServerPage] = useState(0);
  const [limit, setLimit] = useState<LimitOption>(100);

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

  const columnResizeMode: ColumnResizeMode = "onChange";

  // ---------------------------------------------------------------------------
  // Fetch
  // ---------------------------------------------------------------------------
  const fetchPage = useCallback(
    (page: number, sev: string, st: string, pageLimit: number) => {
      setLoading(true);
      setError(null);
      getFindings({
        limit: pageLimit,
        offset: page * pageLimit,
        severity: sev === "All" ? undefined : sev,
        state: st === "All" ? undefined : st,
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
    fetchPage(serverPage, severityFilter, stateFilter, limit);
  }, [fetchPage, serverPage, severityFilter, stateFilter, limit]);

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
  const handleLimitChange = (v: string) => {
    setLimit(Number(v) as LimitOption);
    setServerPage(0);
  };

  // ---------------------------------------------------------------------------
  // Client-side text search filter on top of server page
  // ---------------------------------------------------------------------------
  const filteredRows = useMemo(() => {
    if (!globalFilter.trim()) return rows;
    const q = globalFilter.toLowerCase();
    return rows.filter(
      (r) =>
        r.cve_id.toLowerCase().includes(q) ||
        (r.image_name ?? "").toLowerCase().includes(q) ||
        r.package_name.toLowerCase().includes(q)
    );
  }, [rows, globalFilter]);

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

  // ---------------------------------------------------------------------------
  // Flat-list TanStack table (used only when groupBy === "none")
  // ---------------------------------------------------------------------------
  const flatTable = useReactTable({
    data: filteredRows,
    columns: FLAT_COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    enableColumnResizing: true,
    columnResizeMode,
    manualPagination: true,
  });

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
      <div className="flex flex-wrap gap-2 items-center">
        <Input
          placeholder="Search CVE, image, package…"
          value={globalFilter}
          onChange={(e) => setGlobalFilter(e.target.value)}
          className="text-[12px] max-w-[240px]"
        />
        <FilterSelect
          label="Severity filter"
          value={severityFilter}
          options={SEVERITIES}
          onChange={handleSeverityChange}
        />
        <FilterSelect
          label="State filter"
          value={stateFilter}
          options={STATES}
          onChange={handleStateChange}
        />
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
    );

    // Table area — switches based on groupBy
    let tableArea: React.ReactNode;
    if (groupBy === "none") {
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
    } else {
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

    // Pagination footer (only meaningful in flat mode; grouped rows show count caption)
    const paginationFooter = groupBy === "none" ? (
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

  return (
    <WidgetCard label="All Findings" title="Findings list">
      {body}
    </WidgetCard>
  );
}
