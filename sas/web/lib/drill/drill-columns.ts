import type { DrillColumnMap } from "./drill-types";

/**
 * Maps column accessor keys to drill behaviour.
 *
 * - "findings" mode: narrows the table to matching rows, populates search box
 * - "workload_drill" mode: replaces table content with workload detail rows
 *
 * Adding a new drillable column requires only a config entry.
 */
export const DRILL_COLUMNS: DrillColumnMap = {
  cve_id: {
    field: "cve",
    mode: "findings",
    searchBox: true,
  },
  package_name: {
    field: "package",
    mode: "findings",
    searchBox: true,
  },
  image_name: {
    field: "image",
    mode: "findings",
    searchBox: true,
  },
  // Weighted mode: clicking workload count drills into workload details
  workload_count: {
    field: "cve",
    mode: "workload_drill",
    searchBox: false,
  },
};
