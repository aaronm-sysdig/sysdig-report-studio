import type { ColumnResizeMode } from "@tanstack/react-table";

/**
 * Default options applied to every TanStack Table instance.
 * Add new defaults here so widgets inherit them automatically.
 */
export const TABLE_DEFAULTS = {
  /** Sorting toggles asc ↔ desc only — never removes sort on third click */
  enableSortingRemoval: false,
  /** Column resizing enabled by default */
  enableColumnResizing: true,
  /** Resize on drag (not on mouseup) for smoother UX */
  columnResizeMode: "onChange" as ColumnResizeMode,
};
