# Grace Period for Disappeared Findings

**Date:** 2026-05-16
**Branch:** sas-phase1-data-foundation

## Problem

On 2026-05-13, the system marked 30,533 findings as REMEDIED in a single day. Investigation revealed that 99% were false positives: 86 images present in the May 12 Sysdig scan were absent from the May 13 scan (intermittent scan coverage, not real fixes). When those images' findings disappeared from the snapshot, the diff logic immediately closed them.

The root cause: the current logic treats "not seen in latest snapshot" as "fixed". This is incorrect for any system with intermittent scan coverage, report failures, or ephemeral workloads.

## Design

### Approach

Add a configurable grace period (default: 3 days) between a finding disappearing and it being marked CLOSED. During the grace period, the finding is marked STALE but remains in OPEN state. If it reappears within the grace period, it is cleared back to normal OPEN. Only after N consecutive days absent is it truly closed.

### Schema Change

Add one column to `finding_state`:

```sql
ALTER TABLE finding_state ADD COLUMN grace_period_since TIMESTAMPTZ DEFAULT NULL;
```

Applied via the existing migration path in `schema.py` (same pattern as `cisa_kev_known_ransomware`).

### State Machine Changes

A finding is considered STALE when: `state = 'OPEN' AND reason_code = 'STALE' AND grace_period_since IS NOT NULL`

No new state values are introduced. STALE is a reason_code on an OPEN finding.

#### Transitions

| Scenario | Before | After |
|----------|--------|-------|
| Finding disappears (first time) | CLOSED / REMEDIED | OPEN / STALE, grace_period_since = now |
| STALE finding reappears | (N/A - was closed) | OPEN / NULL (normal RESEEN) |
| STALE for >= 3 days | (already closed day 1) | CLOSED / REMEDIED, closed_at = now |
| STALE for < 3 days, still absent | (already closed day 1) | Still STALE, no change |

### Diff Logic Changes

**fast_finding_diff.py** — current step 5 (DISAPPEARED) is replaced with two steps:

**Step 5a — Mark newly disappeared as STALE:**
- Query: OPEN findings not in today's CSV, excluding already-STALE findings
- Update: `reason_code = 'STALE'`, `grace_period_since = snapshot_at`

**Step 5b — Expire old STALE findings:**
- Query: `reason_code = 'STALE' AND grace_period_since <= today - grace_period_days`
- Update: `state = 'CLOSED'`, `reason_code = 'REMEDIED'`, `closed_at = snapshot_at`
- These expired findings are also written to `daily_closed_snapshot` (same table as step 5a), so the closed count per image is accurate

**Step 3 (RESEEN) — extended:**
- Also clear `reason_code = NULL`, `grace_period_since = NULL` for any STALE finding that reappears in today's CSV

**finding_diff.py (old pipeline)** — same pattern applied to its row-by-row DISAPPEARED path for consistency.

### Configuration

- `GRACE_PERIOD_DAYS = 3` constant in `reason_code.py`
- `'STALE'` added to the `ReasonCode` literal type
- Both pipelines reference the same constant
- Future CLI flag or env var override can be added later

### Metrics / Dashboards

No changes required. `daily_open_snapshot` queries `WHERE state = 'OPEN'`, which naturally includes STALE findings. They continue to count as open.

The frontend's FindingsTable already renders `reason_code` in a Reason column — STALE will appear as a value. A future cosmetic enhancement could highlight STALE rows visually.

### Rollup Impact

No changes. All rollup measures filter on `state = 'OPEN'` or `state = 'CLOSED'`. STALE findings are OPEN, so they are included in open counts. Expired STALE → CLOSED findings will appear in fixed counts with reason_code REMEDIED.

## Testing

1. Disappeared finding → STALE on first absence (unit test)
2. STALE finding reappears → cleared to OPEN / RESEEN (unit test)
3. STALE finding expires after 3 days → CLOSED / REMEDIED (unit test)
4. STALE finding at day 2 → still STALE, not closed yet (unit test)
5. Multi-day ingest with image churn: appear → disappear → reappear (scenario test)
6. Verify daily_open_snapshot counts include STALE findings (integration test)
