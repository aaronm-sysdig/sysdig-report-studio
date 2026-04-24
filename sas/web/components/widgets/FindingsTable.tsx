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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const PAGE_SIZE = 25;

const SEVERITIES = ["All", "Critical", "High", "Medium", "Low", "Negligible"] as const;
const STATES = ["All", "OPEN", "CLOSED"] as const;

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
// Column definitions
// ---------------------------------------------------------------------------
const COLUMNS: ColumnDef<FindingRow>[] = [
  {
    accessorKey: "cve_id",
    header: "CVE",
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block max-w-[140px]"
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
    cell: (info) => <SeverityPill value={String(info.getValue())} />,
  },
  {
    accessorKey: "image_name",
    header: "Image",
    cell: (info) => {
      const val = info.getValue() as string | null;
      return (
        <span
          className="font-mono text-[11px] truncate block max-w-[220px]"
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
    cell: (info) => (
      <span
        className="font-mono text-[11px] truncate block max-w-[160px]"
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
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {relativeTime(String(info.getValue()))}
      </span>
    ),
  },
  {
    accessorKey: "last_seen",
    header: "Last seen",
    cell: (info) => (
      <span className="text-[11px]" style={{ color: "var(--fg-muted)" }}>
        {relativeTime(String(info.getValue()))}
      </span>
    ),
  },
  {
    accessorKey: "state",
    header: "State",
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
// Component
// ---------------------------------------------------------------------------
export function FindingsTable() {
  // Server-side pagination / filter state
  const [severityFilter, setSeverityFilter] = useState<string>("All");
  const [stateFilter, setStateFilter] = useState<string>("All");
  const [serverPage, setServerPage] = useState(0);

  // Client-side text search
  const [globalFilter, setGlobalFilter] = useState("");

  // Data state
  const [rows, setRows] = useState<FindingRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Sorting
  const [sorting, setSorting] = useState<SortingState>([
    { id: "last_seen", desc: true },
  ]);

  // ---------------------------------------------------------------------------
  // Fetch
  // ---------------------------------------------------------------------------
  const fetchPage = useCallback(
    (page: number, sev: string, st: string) => {
      setLoading(true);
      setError(null);
      getFindings({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
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
    fetchPage(serverPage, severityFilter, stateFilter);
  }, [fetchPage, serverPage, severityFilter, stateFilter]);

  // Reset to page 0 when filters change
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

  // ---------------------------------------------------------------------------
  // Client-side search filter applied on top of server page
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
  // TanStack table (client-side sort only; pagination is server-side)
  // ---------------------------------------------------------------------------
  const table = useReactTable({
    data: filteredRows,
    columns: COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    manualPagination: true,
  });

  // ---------------------------------------------------------------------------
  // Pagination labels
  // ---------------------------------------------------------------------------
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const firstRow = serverPage * PAGE_SIZE + 1;
  const lastRow = Math.min((serverPage + 1) * PAGE_SIZE, total);

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
    body = (
      <div className="flex flex-col gap-2">
        {/* Toolbar */}
        <div className="flex flex-wrap gap-2 items-center">
          <Input
            placeholder="Search CVE, image, package…"
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="text-[12px] max-w-[260px]"
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
        </div>

        {/* Table */}
        <div style={{ overflowX: "auto" }}>
          <table className="w-full text-left border-collapse" style={{ tableLayout: "fixed" }}>
            <colgroup>
              <col style={{ width: "14%" }} />
              <col style={{ width: "9%" }} />
              <col style={{ width: "22%" }} />
              <col style={{ width: "16%" }} />
              <col style={{ width: "10%" }} />
              <col style={{ width: "10%" }} />
              <col style={{ width: "8%" }} />
              <col style={{ width: "11%" }} />
            </colgroup>
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="text-[10px] font-medium tracking-widest uppercase pb-2 pt-1 select-none"
                      style={{
                        color: "var(--fg-muted)",
                        cursor: header.column.getCanSort() ? "pointer" : "default",
                        borderBottom: "1px solid var(--border-subtle)",
                        paddingRight: "8px",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      <SortIcon direction={header.column.getIsSorted()} />
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={8}
                    className="text-center text-[12px] py-6"
                    style={{ color: "var(--fg-muted)" }}
                  >
                    No findings match your filters.
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

        {/* Pagination footer */}
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
      </div>
    );
  }

  return (
    <WidgetCard label="All Findings" title="Findings list">
      {body}
    </WidgetCard>
  );
}
