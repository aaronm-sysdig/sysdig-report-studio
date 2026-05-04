/** Field names that can be filtered via drill-in. */
export type DrillField = "cve" | "package" | "image";

/** Display mode for the FindingsTable. */
export type DrillMode = "findings" | "workload_drill";

/** The active drill filter parsed from URL params. */
export interface DrillFilter {
  field?: DrillField;
  value?: string;
  mode?: DrillMode;
}

/** Return type of useDrillFilter hook. */
export interface UseDrillFilterReturn {
  /** Current filter state derived from URL params. */
  filter: DrillFilter;
  /** Apply a filter by field and value (updates URL, pushes history). */
  applyFilter: (field: DrillField, value: string) => void;
  /** Switch display mode (e.g. workload_drill). */
  setMode: (mode: DrillMode) => void;
  /** Clear all drill params (restores default view). */
  clearFilter: () => void;
  /** True when any drill filter is active. */
  isFiltered: boolean;
}

/** Configuration for a drillable column. */
export interface DrillConfig {
  /** URL param name (matches DrillField). */
  field: DrillField;
  /** Display mode to activate. */
  mode: DrillMode;
  /** Whether to populate the search box with the filter value. */
  searchBox: boolean;
}

/** Map of column accessor keys to their drill config. */
export type DrillColumnMap = Record<string, DrillConfig>;
