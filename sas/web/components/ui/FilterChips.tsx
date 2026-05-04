"use client";

import type { DrillFilter } from "@/lib/drill/drill-types";

interface FilterChipsProps {
  filter: DrillFilter;
  onClear: () => void;
  onModeReset?: () => void;
}

const FIELD_LABELS: Record<string, string> = {
  cve: "CVE",
  package: "Package",
  image: "Image",
};

export function FilterChips({ filter, onClear, onModeReset }: FilterChipsProps) {
  if (!filter.field || !filter.value) {
    return null;
  }

  const label = FIELD_LABELS[filter.field] ?? filter.field;

  return (
    <div className="flex flex-wrap items-center gap-2 pt-1 pb-1">
      {/* Filter chip */}
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium cursor-pointer select-none"
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-subtle)",
          color: "var(--fg-primary)",
        }}
        onClick={onClear}
        title="Click to clear filter"
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") onClear();
        }}
      >
        {label}: {filter.value}
        <span style={{ fontWeight: "bold", marginLeft: "2px" }}>✕</span>
      </span>

      {/* Mode chip — shown when in workload_drill mode */}
      {filter.mode === "workload_drill" && onModeReset && (
        <span
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium cursor-pointer select-none"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--severity-high)",
            color: "var(--severity-high)",
          }}
          onClick={onModeReset}
          title="Click to return to findings view"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") onModeReset();
          }}
        >
          Workload details
          <span style={{ fontWeight: "bold", marginLeft: "2px" }}>✕</span>
        </span>
      )}
    </div>
  );
}
