# Weighted Runtime Scoring — Design Spec

**Date:** 2026-05-03  
**Status:** Draft  
**Branch:** `sas-phase1-data-foundation`

## Problem

The FindingsTable supports group-by modes (CVE, Package, Repository) but lacks a prioritisation mode that answers: *"which findings should I fix first?"* Severity alone doesn't account for blast radius (how many workloads are affected) or actionability (is there a fix, is the package in-use).

## Goal

Add a **"Weighted"** group-by mode that ranks findings by a configurable score combining severity, actionability flags, and workload blast radius. Users can tune the weights to match their triage philosophy.

## Formula

```
score = (severity_weight + in_use_weight + fix_weight + exploit_weight) × workload_count
```

- **severity_weight** — Critical: 2, High: 1, Medium: 0, Low: 0, Negligible: 0 (configurable)
- **in_use_weight** — +1 if `in_use = true` (configurable)
- **fix_weight** — +1 if `fix_available = true` (configurable)
- **exploit_weight** — +1 if `public_exploit = true` (configurable)
- **workload_count** — `COUNT(*)` from `workload_runs_image_daily` for the latest snapshot date, joined via `image_id`. Represents distinct (cluster, namespace, workload, container) tuples running an image with this finding.

### Why multiplication?

Additive scoring (`flags + workloads`) lets workload count dominate severity. Multiplication keeps severity as the primary differentiator while amplifying by blast radius. A Critical in 50 workloads (150) is 2× a High in 50 (100), and both are meaningfully above a Critical in 1 workload (6).

### Severity gate

Medium/Low default to severity_weight = 0, meaning they only score if they have actionability flags (in_use, fix, exploit). Without flags, a Medium scores 0 and is excluded from the weighted view.

The severity gate (`☑ Critical ☑ High ☐ Medium ☐ Low ☐ Negligible`) controls which severities appear regardless of weight. Default: Critical + High only.

## Data Source

Workload counts are derived from `workload_runs_image_daily`:
- One row per (date, cluster, namespace, workload_type, workload_name, container_name, image_id)
- Query joins `finding_state.image_id` → `workload_runs_image_daily.image_id` filtered to the latest snapshot date
- Sums workload counts across all images affected by a given CVE (for CVE group-by)

**Known limitation:** `replica_count` is stored as 1 per workload. If a Deployment has 3 replicas, it counts as 1. This is acceptable for PoC; a kubectl feed can upgrade this later.

## UI Design

### Toolbar

When the user selects "Weighted" from the "Group by" dropdown, a configuration panel expands below the toolbar:

```
Group by: [Weighted ▾]

┌─────────────────────────────────────────────────────────────────┐
│ Severity:  ☑ Critical  ☑ High  ☐ Medium  ☐ Low  ☐ Negligible   │
│ Weights:   Critical [2]  High [1]  In-use [1]  Has Fix [1]     │
│                        Exploit [1]                              │
└─────────────────────────────────────────────────────────────────┘
```

- Severity checkboxes control inclusion (independent of weight)
- Weight inputs are `<input type="number">` spin buttons with up/down arrows, `min={0}`, `max={10}`, `step={1}`. This prevents invalid input (no text, no negative values) and gives tactile control.
- Panel collapses when group-by switches away from "Weighted"

### Table Columns

| Column | Description | Example |
|---|---|---|
| **Score** | Weighted score, sortable (desc default) | `150` |
| **CVE** | CVE ID, links to drill-in (future) | `CVE-2025-68121` |
| **Severity** | Badge with severity colour | `Critical` |
| **Workloads** | Count of affected workloads | `50` |
| **Fix** | ✓ (green) or — (muted) | `✓` |
| **In-use** | ✓ (green) or ✕ (orange) | `✓` |
| **Exploit** | ⚠ (dark red) or — (muted) | `—` |
| **Breakdown** | The math: `(sev + flags) × workloads` | `(3+1+1) × 50` |

The Breakdown column shows the contributing weights (non-zero only) multiplied by workload count. This helps users understand why a finding ranks where it does.

### Group-by behaviour

The Weighted mode groups by CVE (same as the existing "CVE" group-by). Each row represents a distinct CVE with aggregated data:
- `workload_count` = sum of workloads across all images affected by this CVE
- Boolean flags (Fix, In-use, Exploit) = OR across all packages/images (true if any instance has it)
- Severity = highest severity across all instances

## Configuration Persistence

Weights and severity gate preferences are stored in `localStorage` under key `sas:weighted-weights`. Schema:

```json
{
  "severityGate": ["Critical", "High"],
  "weights": {
    "Critical": 2,
    "High": 1,
    "Medium": 0,
    "Low": 0,
    "Negligible": 0,
    "in_use": 1,
    "fix_available": 1,
    "public_exploit": 1
  }
}
```

**Future work:** Migrate to user profile in database when multi-tenant auth is implemented. The localStorage key should be treatable as a migration target.

## Scoring Computation

Scoring is computed **client-side** in the frontend. The existing `/api/findings` endpoint returns `FindingRow` data including `image_id`, `in_use`, `fix_available`, `public_exploit`. To get workload counts, a new API endpoint `/api/workload-counts` is needed returning `{ cve_id, workload_count }` aggregated from `workload_runs_image_daily` for the latest snapshot.

The frontend:
1. Fetches findings grouped by CVE (existing group-by CVE mode)
2. Fetches workload counts per CVE from `/api/workload-counts`
3. For each CVE row, computes: `(severity_weight + flag_weights) × workload_count`
4. Filters by severity gate, sorts by score descending

**Future:** If performance requires it, scoring can move server-side with weights as query params.

## Validation Example

For workload `sysdig-notifier:0.12` (image `sha256:aa2376...`, cluster `sysdn02`, namespace `sysdig-agent`):
- Sysdig Runtime reports: Critical=2, High=18, Medium=38, Low=6, Negligible=50
- Our DB (after loading through 2026-05-02): **exact match**
- Example CVE score: CVE-2026-4438 (Critical, in_use=true, fix=false, exploit=false, 1 workload) = `(2 + 1 + 0 + 0) × 1 = 3`

## Out of Scope

- Per-image or per-package weighted views (CVE-only for now)
- Server-side scoring endpoint
- Multi-tenant weight profiles
- Replica-aware workload counting (uses `workload_runs_image_daily` as-is)
- Historical trend of weighted scores over time
