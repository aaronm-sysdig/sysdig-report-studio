# Fleet Severity Snapshot — Design Spec

**Date:** 2026-05-03
**Status:** Approved by user (Aaron Miles)
**Scope:** Single widget — static traffic-light display of latest severity counts

## Problem

The dashboard has no immediate "where are we at right now" view. The user must scroll past the Image Remediation Story widget to see any current-state numbers. We need a top-level summary showing the latest snapshot's open findings broken down by severity.

## Design

**Widget name:** Fleet Severity Snapshot
**Label:** "Fleet Metrics"
**Title:** "Fleet Severity Snapshot"

Five coloured blocks displayed side-by-side, each showing:
- Large count number (28px, bold)
- Severity name (10px, uppercase)

Colours from the existing severity palette (`CHART_COLORS`):
| Severity | Background | Text |
|---|---|---|
| Critical | `#cb87da` | White |
| High | `#ff7875` | Black |
| Medium | `#ffaa40` | Black |
| Low | `#fdd836` | Black |
| Negligible | `#b5c4cc` | White |

Footer narrative: "As of {latest_date} · {total} total open findings across {image_count} images"

**Interaction:** None (static display). Drill-in is out of scope for this iteration.

## Data

Five parallel queries to `POST /api/query`:

| Measure | Lens | Time Window | Filters | Group By |
|---|---|---|---|---|
| `count_open_critical` | `Image` | `last_n_snapshots: 1` | none | none |
| `count_open_high` | `Image` | `last_n_snapshots: 1` | none | none |
| `count_open_medium` | `Image` | `last_n_snapshots: 1` | none | none |
| `count_open_low` | `Image` | `last_n_snapshots: 1` | none | none |
| `count_open_negligible` | `Image` | `last_n_snapshots: 1` | none | none |

Each query returns a `QueryResult` with series. The total for each severity is the sum of `y[]` across all series for the single returned date.

**Note:** `count_open_negligible` uses the direct query path (not rollup) — this is acceptable for a single-date query.

## Component Architecture

**File:** `sas/web/components/widgets/FleetSeveritySnapshot.tsx`

- `"use client"` directive
- Import `WidgetCard`, `runQuery`, `CHART_COLORS`
- State: `counts` (record of severity → number | null), `error`, `loading`, `latestDate`, `totalImages`
- `useEffect` fires 5 parallel `runQuery()` calls on mount, with cancelled guard
- Renders 5 `<div>` blocks in a flex row, each with severity-specific background/text colours
- Three states: skeleton (5 grey blocks) → error (red message) → data (coloured blocks)

**No ECharts dependency** — pure CSS flexbox layout.

## Placement

Added to `sas/web/app/dashboard/page.tsx` as Row 1, `span 12`, above all existing widgets.

## Edge Cases

- **Loading:** 5 skeleton blocks with `animate-pulse` and `--bg-surface` colour
- **Error:** Red error message with `role="alert"` and `--severity-critical` colour
- **Null/missing data for a severity:** Show "—" with muted text on that block
- **All queries fail:** Show error state
- **Partial failure:** Show counts for successful queries, "—" for failed ones

## Out of Scope

- Click-to-filter (Phase 3.3 drill-in)
- Day-over-day delta arrows
- Time period selector
- Reason code breakdown
