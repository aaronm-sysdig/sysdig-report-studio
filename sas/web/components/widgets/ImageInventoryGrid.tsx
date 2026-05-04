"use client";

import { useEffect, useState, useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { WidgetCard } from "./WidgetCard";
import { Input } from "@/components/ui/input";
import { TABLE_DEFAULTS } from "@/lib/table.defaults";
import { runQuery, getEntities } from "@/lib/api/client";
import type { QueryIn } from "@/lib/api/client";
import { CHART_COLORS } from "@/lib/charts/defaults";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface ImageEntity {
  id: string;
  label: string;
  repository: string;
  tag: string;
  last_seen?: string;
}

interface ImageRow {
  imageId: string;
  image: string;
  critical: number;
  high: number;
  openTotal: number;
  lastSeen: string | null;
}

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
    return (
      <span style={{ opacity: 0.3, fontSize: "10px", marginLeft: "4px" }}>⇅</span>
    );
  }
  return (
    <span style={{ fontSize: "10px", marginLeft: "4px" }}>
      {direction === "asc" ? "↑" : "↓"}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------
const PAGE_SIZE = 10;

const COLUMNS: ColumnDef<ImageRow>[] = [
  {
    accessorKey: "image",
    header: "Image",
    // No fixed size — IMAGE column flexes to fill remaining space via colgroup
    minSize: 120,
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
    accessorKey: "critical",
    header: "Critical",
    size: 100,
    minSize: 60,
    cell: (info) => {
      const val = info.getValue() as number;
      return (
        <span
          className="font-semibold text-[12px]"
          style={{ color: val > 0 ? CHART_COLORS.severityCritical : "var(--fg-muted)" }}
        >
          {val.toLocaleString("en-GB")}
        </span>
      );
    },
  },
  {
    accessorKey: "high",
    header: "High",
    size: 100,
    minSize: 60,
    cell: (info) => {
      const val = info.getValue() as number;
      return (
        <span
          className="font-semibold text-[12px]"
          style={{ color: val > 0 ? CHART_COLORS.severityHigh : "var(--fg-muted)" }}
        >
          {val.toLocaleString("en-GB")}
        </span>
      );
    },
  },
  {
    accessorKey: "openTotal",
    header: "Open total",
    size: 120,
    minSize: 80,
    cell: (info) => (
      <span
        className="text-[12px]"
        style={{ color: "var(--fg-primary)" }}
      >
        {(info.getValue() as number).toLocaleString("en-GB")}
      </span>
    ),
  },
  {
    accessorKey: "lastSeen",
    header: "Last seen",
    size: 120,
    minSize: 80,
    cell: (info) => {
      const val = info.getValue() as string | null;
      return (
        <span className="text-[12px]" style={{ color: "var(--fg-muted)" }}>
          {val ? relativeTime(val) : "—"}
        </span>
      );
    },
  },
];

// ---------------------------------------------------------------------------
// Query builders
// ---------------------------------------------------------------------------
function makeQuery(measure: string): QueryIn {
  return {
    lens: "Image",
    traversal: [],
    time: { mode: "last_n_snapshots", n: 1, granularity: "day" },
    measure,
    filters: [],
    group_by: [],
    order_by: null,
    limit: null,
  };
}

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
        height: "320px",
      }}
      aria-label="Loading image inventory…"
      role="status"
    />
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ImageInventoryGrid() {
  const [data, setData] = useState<ImageRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [globalFilter, setGlobalFilter] = useState("");
  const [sorting, setSorting] = useState<SortingState>([
    { id: "critical", desc: true },
  ]);

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      getEntities("Image") as Promise<ImageEntity[]>,
      runQuery(makeQuery("count_open_critical")),
      runQuery(makeQuery("count_open_high")),
      runQuery(makeQuery("count_open")),
    ])
      .then(([entities, criticalResult, highResult, totalResult]) => {
        if (cancelled) return;

        // Build lookup maps: image_id → value at latest snapshot
        function buildMap(result: { series: Array<{ key: { image_id?: string }; x: string[]; y: (number | null)[] }> }): Map<string, number> {
          const map = new Map<string, number>();
          for (const s of result.series) {
            const id = s.key.image_id;
            if (!id) continue;
            // Take the last (latest) snapshot value
            const lastY = s.y[s.y.length - 1];
            if (typeof lastY === "number") {
              map.set(id, lastY);
            }
          }
          return map;
        }

        const criticalMap = buildMap(criticalResult);
        const highMap = buildMap(highResult);
        const totalMap = buildMap(totalResult);

        const rows: ImageRow[] = entities.map((entity) => ({
          imageId: entity.id,
          image: entity.label || `${entity.repository}:${entity.tag}`,
          critical: criticalMap.get(entity.id) ?? 0,
          high: highMap.get(entity.id) ?? 0,
          openTotal: totalMap.get(entity.id) ?? 0,
          lastSeen: entity.last_seen ?? null,
        }));

        setData(rows);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load image inventory.");
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, []);

  const table = useReactTable({
    ...TABLE_DEFAULTS,
    data,
    columns: COLUMNS,
    state: {
      sorting,
      globalFilter,
      pagination: { pageIndex: 0, pageSize: PAGE_SIZE },
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    // Allow manual pagination control
    manualPagination: false,
  });

  // Keep pagination index reset when filter changes
  const filteredRowCount = table.getFilteredRowModel().rows.length;
  const pageIndex = table.getState().pagination.pageIndex;
  const pageCount = table.getPageCount();

  // Clamp page index if filter shrinks results
  const safePageIndex = useMemo(() => {
    if (pageCount === 0) return 0;
    return Math.min(pageIndex, pageCount - 1);
  }, [pageIndex, pageCount]);

  useEffect(() => {
    if (safePageIndex !== pageIndex) {
      table.setPageIndex(safePageIndex);
    }
  }, [safePageIndex, pageIndex, table]);

  const firstRow = safePageIndex * PAGE_SIZE + 1;
  const lastRow = Math.min((safePageIndex + 1) * PAGE_SIZE, filteredRowCount);

  // Render body
  let body: React.ReactNode;

  if (error) {
    body = (
      <div
        className="flex items-center justify-center text-sm"
        style={{ color: "var(--severity-critical)", minHeight: "280px" }}
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
        {/* Search input */}
        <Input
          placeholder="Search images…"
          value={globalFilter}
          onChange={(e) => {
            setGlobalFilter(e.target.value);
            table.setPageIndex(0);
          }}
          className="text-[12px]"
        />

        {/* Table */}
        <div style={{ overflowX: "auto" }}>
          <table className="w-full text-left border-collapse" style={{ tableLayout: "fixed" }}>
            <colgroup>
              {/* IMAGE column: auto — flexes to fill remaining space */}
              <col style={{ width: "auto" }} />
              <col style={{ width: "100px" }} />
              <col style={{ width: "100px" }} />
              <col style={{ width: "120px" }} />
              <col style={{ width: "120px" }} />
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
                        position: "relative",
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
              {table.getRowModel().rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={5}
                    className="text-center text-[12px] py-6"
                    style={{ color: "var(--fg-muted)" }}
                  >
                    No images match your search.
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
            {filteredRowCount === 0
              ? "No images"
              : `Showing ${firstRow}–${lastRow} of ${filteredRowCount} images`}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
              className="text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
              style={{
                border: "1px solid var(--border-subtle)",
                color: "var(--fg-primary)",
                cursor: table.getCanPreviousPage() ? "pointer" : "not-allowed",
              }}
            >
              Prev
            </button>
            <button
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
              className="text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
              style={{
                border: "1px solid var(--border-subtle)",
                color: "var(--fg-primary)",
                cursor: table.getCanNextPage() ? "pointer" : "not-allowed",
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
    <WidgetCard
      label="Image Inventory"
      title="All images in the fleet"
    >
      {body}
    </WidgetCard>
  );
}
