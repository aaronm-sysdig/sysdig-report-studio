# Cross-Widget Drill-In — Design Spec

**Date:** 2026-05-04
**Phase:** 3.3
**Status:** Draft — awaiting user review

## Problem

The dashboard shows aggregate views across all findings. Users need to drill into specific items (CVEs, packages, images) to understand root cause and scope. Currently there is no mechanism to filter widgets or navigate between related data.

## Scope (Phase 3.3 — MVP)

- **FindingsTable self-filtering:** Click a cell (CVE, package, image) to filter the table to that value
- **Workload drill:** Click workload count in weighted mode to see which workloads run the affected image
- **Browser navigation:** Back/forward buttons reverse drill actions
- **Visible feedback:** Search box populates with filter value; filter chips show active state
- **Extensible foundation:** Design supports cross-widget filtering in future phases

**Out of scope (future):**
- Cross-widget filtering (FleetCriticalTrend, ImageRemediationStory, etc.)
- Multi-filter composition (AND/OR logic)
- Group-by mode preservation during drill
- CSV export of filtered results

## Approach: Column Drill Config

A `DRILL_COLUMNS` configuration object maps each column to its drill behaviour. Clicking a cell emits a filter, the URL updates, the table refetches. All widgets subscribe to the same filter hook, making cross-widget a future addition rather than a refactor.

## Architecture

```
URL (?cve=CVE-2025-68121)
    ↕
useDrillFilter() hook
    - Reads URL params on mount + onChange
    - Returns { filter, applyFilter, clearFilter, mode, setMode }
    ↕ (React state)
FindingsTable
    - Passes filter to API query
    - DRILL_COLUMNS config makes cells clickable
    - FilterChips bar shows active filter
    ↕ (future)
Other widgets (subscribe to useDrillFilter())
```

### Filter Lifecycle

1. User clicks CVE cell → `applyFilter("cve", "CVE-2025-68121")`
2. Hook updates URL params → `?cve=CVE-2025-68121` (via `router.push`)
3. URL change re-triggers hook → FindingsTable refetches with filter
4. Search box populates with filter value; FilterChips renders "CVE-2025-68121 [✕]"
5. Browser back → URL reverts → hook fires → table clears filter

## URL Scheme

### Parameters

| Param | Example | Meaning |
|---|---|---|
| `cve` | `CVE-2025-68121` | Filter by CVE ID |
| `package` | `openssl` | Filter by package name |
| `image` | `nginx:1.25` | Filter by image name |
| `mode` | `workload_drill` | Show workload rows for filtered CVE |

Only one of `cve`, `package`, `image` is active at a time. Setting a new filter replaces the old one.

### Examples

| URL | State |
|---|---|
| `/dashboard` | Default — all findings |
| `?cve=CVE-2025-68121` | Filtered findings for that CVE |
| `?cve=CVE-2025-68121&mode=workload_drill` | Workload details for that CVE |
| `?package=openssl` | Filtered by package |

### Back Button Behaviour

Each drill action pushes a new history entry:
- Click CVE → `?cve=X` (push)
- Click workload count → `?cve=X&mode=workload_drill` (push)
- Back → removes `mode`, returns to CVE view
- Back again → removes `cve`, returns to all findings
- Clear filter chip (✕) → equivalent to back, clears params

Implementation: Next.js `useRouter().push(path, { scroll: false })` to avoid page jump.

## Component Design

### `useDrillFilter()` Hook

**File:** `sas/web/lib/drill/use-drill-filter.ts`

```typescript
interface DrillFilter {
  field?: "cve" | "package" | "image";
  value?: string;
  mode?: "findings" | "workload_drill";
}

interface UseDrillFilterReturn {
  filter: DrillFilter;
  applyFilter: (field: string, value: string) => void;
  setMode: (mode: "findings" | "workload_drill") => void;
  clearFilter: () => void;
  isFiltered: boolean;
}

export function useDrillFilter(): UseDrillFilterReturn;
```

Reads URL params via `useSearchParams`. Each update uses `useRouter().push()` to maintain history. Returns a stable `DrillFilter` object that widgets can consume.

### `DRILL_COLUMNS` Config

**File:** `sas/web/lib/drill/drill-columns.ts`

```typescript
interface DrillConfig {
  field: "cve" | "package" | "image";  // URL param name
  mode: "filter" | "workload_drill";   // display mode
  searchBox: boolean;                   // populate search box?
}

export const DRILL_COLUMNS: Record<string, DrillConfig> = {
  cve_id:    { field: "cve",     mode: "filter",          searchBox: true  },
  package:   { field: "package", mode: "filter",          searchBox: true  },
  image_id:  { field: "image",   mode: "filter",          searchBox: true  },
  workloads: { field: "cve",     mode: "workload_drill",  searchBox: false },
};
```

Column accessor keys map to drill behaviour. Adding a new drillable column requires only a config entry — no component changes.

### FilterChips Component

**File:** `sas/web/components/ui/FilterChips.tsx`

Renders active filters as dismissible chips above the table. Each chip shows the field and value with a ✕ button. Clicking ✕ calls `clearFilter()`.

Visual distinction between filter mode and workload drill mode (different chip colour).

### FindingsTable Integration

- Accepts no new props — consumes `useDrillFilter()` internally
- Search box value is **controlled** by `filter.value` when a drill filter is active; free text when empty
- Typing in search box while filter is active replaces the drill filter (URL param clears)
- Each column's cell renderer checks `DRILL_COLUMNS` to decide if cell is clickable
- Click handler calls `applyFilter(config.field, cellValue)`
- Workload drill mode switches column definitions and fetches workload rows via new API endpoint

### API Query Adaptation

**Filter mode:** Add equality filter to existing `QueryIn`:
```typescript
filters: [
  { column: filter.field, operator: "eq", value: filter.value },
]
```

**Workload drill mode:** New backend endpoint `GET /api/workloads-for-cve?cve_id=CVE-2025-68121` returns:
```sql
SELECT DISTINCT
  wr.cluster_name, wr.namespace_name, wr.workload_type, wr.workload_name,
  wr.container_name, wo.team_id
FROM finding_state fs
JOIN workload_runs_image_daily wr ON wr.image_id = fs.image_id
  AND wr.date = (SELECT MAX(date) FROM workload_runs_image_daily)
LEFT JOIN workload_owned_by wo
  ON wo.cluster_name = wr.cluster_name
  AND wo.namespace_name = wr.namespace_name
  AND wo.workload_type = wr.workload_type
  AND wo.workload_name = wr.workload_name
WHERE fs.cve_id = ? AND fs.state = 'OPEN'
```

## Cross-Widget Extension Path

When ready for cross-widget filtering, each widget calls `useDrillFilter()` and applies the filter to its own query:

| Widget | Response to `?cve=X` |
|---|---|
| FindingsTable | Filter rows to that CVE |
| FleetCriticalTrend | Show trend for that CVE only |
| ImageRemediationStory | Show images containing that CVE |
| KevRansomwareExposure | No change (already scoped to KEV) |

No URL or hook changes required — widgets ignore params they don't understand.

## Error Handling & Edge Cases

| Scenario | Behaviour |
|---|---|
| Invalid filter value (`?cve=nonexistent`) | Empty state: "No findings match this filter" + clear button |
| Workload drill on CVE with 0 workloads | "No workloads running this image" + link back to CVE view |
| Type in search box while drill filter active | Search replaces drill filter, URL param clears |
| Switch group-by mode with active filter | Filter persists (URL unchanged), applies to new grouping |
| Deep history navigation | Each step is a clean URL, no state loss |

## Testing

- **Unit:** `useDrillFilter()` correctly parses and sets URL params for all field types
- **Unit:** `DRILL_COLUMNS` config covers all expected columns, no typos in accessor keys
- **Integration:** Click CVE cell → URL updates → table refetches with correct filter → count updates
- **Integration:** Click workload count → switches to workload rows → shows correct data
- **Integration:** Browser back → filter clears → table resets to default
- **Integration:** Filter chip ✕ → clears filter → URL cleans up

## Files to Create/Modify

### New files
- `sas/web/lib/drill/use-drill-filter.ts` — hook
- `sas/web/lib/drill/drill-columns.ts` — config
- `sas/web/lib/drill/index.ts` — barrel export
- `sas/web/components/ui/FilterChips.tsx` — chip component
- `sas/api/routes/workloads_for_cve.py` — backend endpoint

### Modified files
- `sas/web/components/widgets/FindingsTable.tsx` — consume hook, clickable cells, workload mode
- `sas/api/routes/__init__.py` — register new route
- `sas/web/app/dashboard/page.tsx` — no changes (FindingsTable is self-contained)

## Open Questions

- **Group-by modes:** Should drill filter persist when switching group-by? (Assumed yes, can revisit)
- **Which columns are drillable in non-weighted modes?** Currently only weighted mode has workload counts. Other modes may have different drillable columns. Config is extensible.
- **BreadcrumbStrip integration:** Currently empty. Could wire to drill filter for visual navigation aid. Deferred to cross-widget phase.
