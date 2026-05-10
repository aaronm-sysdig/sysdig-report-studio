# Findings Table — Priority Columns (in_use, fix_available, public_exploit)

**Date:** 2026-05-03
**Status:** Draft — awaiting approval
**Scope:** Add 3 boolean columns to FindingsTable widget with sort support

## Problem

A security professional triaging findings needs to answer: "of these 9 Criticals, which ones have fixes, are in active use, or have known exploits?" The current FindingsTable shows severity, image, package, and timestamps — but not the priority signals that determine what to fix first.

## Design

Three new columns added to the flat-list view (groupBy = "none"):

| Column | DB Field | Render | Sort |
|---|---|---|---|
| **Fix** | `fix_available` | Green check (✓) / grey dash (—) | Boolean (true first) |
| **In-use** | `in_use` | Orange dot (●) / grey dash (—) | Boolean (true first) |
| **Exploit** | `public_exploit` | Red warning (⚠) / grey dash (—) | Boolean (true first) |

Columns placed after "Severity", before "Image" — the priority triage cluster sits between severity and context.

**Grouped views (CVE/Image/Package):** No change — these columns are finding-level, not aggregate-level.

## Data Model

Fields already exist in `finding_state` (confirmed in `ingest/schema.py`):

```sql
in_use BOOLEAN,
fix_available BOOLEAN,
public_exploit BOOLEAN,
```

Fields are populated during CSV ingest (`ingest/finding_diff.py` lines 50, 56).

## Tasks

### Task 1: Expose boolean fields in API

**File:** `sas/api/routes/findings.py`

- [ ] Add `in_use: bool`, `fix_available: bool`, `public_exploit: bool` to `FindingRow` Pydantic model
- [ ] Add columns to SQL SELECT (after `reason_code`)
- [ ] Add values to `FindingRow` constructor in response builder

### Task 2: Regenerate frontend types

**File:** `sas/web/lib/api/types.ts` (auto-generated)

- [ ] Run `npm run generate-api` (requires backend running on :8000)
- [ ] Verify `FindingRow` in `types.ts` includes the 3 new boolean fields

### Task 3: Add columns to FindingsTable

**File:** `sas/web/components/widgets/FindingsTable.tsx`

- [ ] Add 3 column definitions to `FLAT_COLUMNS` (after severity, before image_name)
- [ ] Each column renders icon/dash based on boolean value
- [ ] Each column is sortable (boolean sort: true > false)
- [ ] Column sizes: Fix (60), In-use (70), Exploit (70)

---

## Notes

- No spec doc needed — this is a straightforward data-exposure + UI addition
- No change to ImageInventoryGrid — already defaults to `critical desc`
- Default sort for FindingsTable remains `last_seen desc` (unchanged)
