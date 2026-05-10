# Findings Table — Boolean Filter Checkboxes

**Date:** 2026-05-03
**Status:** In progress
**Scope:** Add 3 boolean filter checkboxes to FindingsTable toolbar with server-side filtering

## Problem

The FindingsTable has dropdown filters for Severity and State, but the 3 new boolean fields (fix_available, in_use, public_exploit) lack filter controls. Users cannot quickly narrow to "only findings with public exploits" or "only in-use packages".

## Design

Three toggle checkboxes added to the toolbar after the State filter:

| Filter | Label | API Param | SQL |
|---|---|---|---|
| Fix | "Has Fix" | `fix_available=1` | `fs.fix_available = 1` |
| In-use | "In-use" | `in_use=1` | `fs.in_use = 1` |
| Exploit | "Has Exploit" | `public_exploit=1` | `fs.public_exploit = 1` |

Each checkbox is a toggle (checked = filter on, unchecked = off). No "All/True/False" dropdown — the semantics are "show me findings where this is true".

**Toolbar layout:**
```
[Search box] [Severity ▾] [State ▾] [☑ Has Fix] [☑ In-use] [☑ Has Exploit]   Group by: [none ▾]   Show: [25 rows ▾]
```

## Tasks

### Task 1: Backend — Add boolean query parameters

**File:** `sas/api/routes/findings.py`

- [ ] Add `fix_available: bool | None = None`, `in_use: bool | None = None`, `public_exploit: bool | None = None` to `list_findings`
- [ ] Add WHERE clauses for each boolean param when provided
- [ ] No validation needed (booleans are self-validating)

### Task 2: Frontend API client — Add boolean params

**File:** `sas/web/lib/api/client.ts`

- [ ] Add `fix_available?: boolean`, `in_use?: boolean`, `public_exploit?: boolean` to `getFindings` opts
- [ ] Append params as `1`/`0` when provided (e.g., `params.set("fix_available", "1")`)

### Task 3: Frontend widget — Add checkbox filters

**File:** `sas/web/components/widgets/FindingsTable.tsx`

- [ ] Add 3 boolean state variables: `fixFilter`, `inUseFilter`, `exploitFilter` (default `false`)
- [ ] Create `BooleanCheckbox` component (styled checkbox with label)
- [ ] Add 3 checkboxes to toolbar after State filter
- [ ] Wire handlers: toggle boolean, reset to page 0, clear global filter
- [ ] Pass boolean filters through to `fetchPage` and `getFindings`

---

## Notes

- Checkboxes are toggles, not tri-state (All/True/False). Checked = "show only true", Unchecked = "no filter"
- No change to grouped views — filters apply server-side before any grouping
- Default sort remains `last_seen desc` (unchanged)
