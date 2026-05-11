# Image Tag Filter for Remediation Widgets

> **Date:** 2026-05-11
> **Status:** Approved
> **Approach:** Query-level filter on Image lens (Approach 1)

## Problem

The Image Remediation Story widget aggregates vulnerability counts by `image_id` (sha256 digest). For repositories with multiple tags and frequent rebuilds (e.g., `pdev` with 25 distinct image_ids across 4 tags), each new build appears as a new series. Old builds drop to zero, new ones spike in — creating volatile charts that don't reflect the actual remediation trend for a logical image like `codeserver`.

The trend graph needs to be filterable by tag so users can see the vulnerability trend for a specific tag across builds, rather than the aggregate of all sha256 digests.

## Architecture

### Backend: Tag Filter on the Image Lens

The existing query compiler supports filters on the Image lens. We extend it to recognise `current_tag` as a valid filter field.

**Rollup path** (primary): When `current_tag` is filtered:
- Compiler joins `daily_metrics_by_image` with `image` table to resolve `current_tag`
- Applies `WHERE image.current_tag = ?` filter
- Aggregates (SUM) across all matching `image_id` rows per date
- Result: one time series per date, summing open findings for all image_ids with that tag

**Direct path** (fallback): Adds `JOIN image ON image.image_id = finding_state.image_id` with tag predicate.

**Rollup router**: Add `current_tag` to `_ROLLUP_FILTER_COLUMNS["daily_metrics_by_image"]` so queries with tag filters use the fast rollup path.

**New endpoint**: `GET /api/entities/tags?repository=<repo>` returns `[{tag, image_count}]` — distinct tags for a repository, populated from the `image` table filtered by `current_repository`.

### Frontend: Tag Combobox

In `ImageRemediationStory.tsx`, a `TagSelect` component sits to the right of the existing `RepositoryPicker`:

```
[ RepositoryPicker (searchable) ] [ TagSelect (dropdown) ]
```

- When a repository is selected, `TagSelect` populates by calling `GET /api/entities/tags?repository=<repo>`
- Options: "All tags" (default, no filter) plus each distinct tag
- Selecting a tag adds `{field: "current_tag", operator: "eq", value: tag}` to the widget's query filters
- "All tags" removes the filter (current behaviour — aggregate all image_ids)
- Tag selection is local widget state (not URL-driven), as it refines the repository selection rather than acting as a cross-widget drill filter
- The chart X-axis (date range) is unchanged. Only the data points change as the tag filter narrows the image_ids included

## Data Flow

1. User selects repository → existing flow fires, fetches data
2. `TagSelect` populates via `/api/entities/tags?repository=<repo>`
3. User selects tag → widget adds `current_tag` filter to query
4. Query compiles: rollup path joins + filters + aggregates across matching image_ids
5. Chart renders: one time series, full date window, data only for dates where tag had open findings
6. User switches tag → new query, same X-axis, new data
7. User selects "All tags" → filter removed, back to aggregate

## Edge Cases

| Case | Behaviour |
|---|---|
| Tag has no data on some dates | Chart shows `null` for missing days — ECharts gaps the line. X-axis width unchanged. |
| No tags for repository | `TagSelect` shows "(no tags)" disabled. Should not occur if repo has data. |
| Tag changes on rebuild | `current_tag` on the `image` table reflects the *current* tag for each sha256 image_id. Historical snapshot counts are preserved per image_id, but filtered by the image's present tag. If an image was rebuilt and retagged, its historical data moves with it. This is an existing limitation — full tag genealogy (which tag was active on which date) is future work. |
| Multiple image_ids share a tag | Counts are summed across all matching image_ids per date (aggregation). |

## What Does Not Change

- Ingest pipeline — no schema or rollup table changes
- Existing widget behaviour without tag filter — identical to current
- Drill-down URL params — tag filter is local state, not cross-widget
- Other widgets — tag filter is scoped to ImageRemediationStory

## Testing

- **Backend**: Unit test that rollup query with `current_tag` filter returns correct aggregated counts vs. direct path
- **Backend**: Unit test for `/api/entities/tags` endpoint with known repository data
- **Frontend**: Verify TagSelect populates correctly for repos with multiple tags
- **Frontend**: Verify chart data changes when switching tags, X-axis unchanged
- **Frontend**: Verify "All tags" removes filter and restores aggregate view
