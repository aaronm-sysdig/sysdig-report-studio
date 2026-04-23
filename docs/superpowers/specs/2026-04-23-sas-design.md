# Sysdig Analytics Studio (SAS) — Design Spec

**Status:** Draft, pending user approval.
**Date:** 2026-04-23.
**Owner:** Aaron Miles (aaron.miles@sysdig.com).
**Scope:** v1 MVP of a historical vulnerability analytics product, built as a sub-project of sysdig-report-studio.

---

## 1. Context

Sysdig customer Phoenix HSL has flagged Sysdig's vulnerability trend graph as a renewal blocker for October 2026. The graph counts container instances, not unique vulnerabilities in images — so scaling replicas up/down moves the line without any remediation having occurred. Competitors (Wiz, Upwind, CrowdStrike) are being evaluated.

This product is a **single-customer historical analytics tool** that ingests daily Sysdig CSV exports, stores them efficiently, and exposes a flexible widget-based UI that answers questions Sysdig-native reporting cannot. It is a stopgap + influence play — it demonstrates to Sysdig product that flexibility matters.

Customer context: [`problem.md`](../../../problem.md).
Adversarial stress-test of the data model against 60 competitor questions: [`2026-04-23-adversarial-brainstorm.md`](../research/2026-04-23-adversarial-brainstorm.md).

## 2. Tenets

These drive every decision. If a design choice violates one, the choice is wrong.

1. **Flexibility > opinion.** LEGO blocks on a LEGO mat, not a fixed solution.
2. **Cater for unknown unknowns.** The data model must answer questions nobody has asked yet.
3. **Apple-grade polish.** Every pixel, transition, default, empty state is deliberate.
4. **Image (not container instance) is the anchor** for remediation tracking.
5. **Graph + time.** Entities are nodes; relationships are edges; time is a property on edges.
6. **Free/open-source only.**
7. **Density: Gmail compact.** Every pixel fights for its place. Whitespace only where it earns scannability.
8. **Honesty over cleverness.** Stale data is labelled stale. Missing days are shown as gaps, not interpolated.

## 3. Architecture

Sub-project lives at `/sas/` inside the existing repo. The existing Streamlit app is untouched. Three components, each with one clear job.

```
sas/
├── ingest/       Python CLI — reads CSV, updates DuckDB
├── api/          FastAPI — HTTP layer over DuckDB, serves Query primitive
└── web/          Next.js + TypeScript — UI
```

**Stack:**
- **Analytics store:** DuckDB (single file).
- **Metadata store:** SQLite (existing, unchanged).
- **Backend:** FastAPI, Python. OpenAPI spec auto-generated.
- **Frontend:** Next.js + TypeScript, shadcn/ui + Tailwind, Apache ECharts, TanStack Table, Zustand.
- **Deployment:** Docker Compose for dev/demo. Kubernetes-ready (PVC for DuckDB, stateless pods, CronJob ingest, env-var config). No external services required.

**Language boundary:** Python for anything touching the CSV; TypeScript for frontend; FastAPI is the typed bridge between them.

**Invariant:** Nothing above the ingest layer writes raw SQL. DuckDB sits behind the Query primitive. DuckDB swappable (for Postgres, etc.) without touching UI.

## 4. Data model

Four kinds of tables in DuckDB.

### 4.1 Entity tables (graph nodes, deduplicated, small)

| Table | Primary key | Notes |
|---|---|---|
| `image` | `image_id` (sha256 digest) | OS, first_seen, last_seen, current repository+tag snapshot |
| `repository` | `repository` | First seen, last tag seen |
| `cve` | `cve_id` | Disclosure date, fix available date, CVSS version, initial severity, CISA KEV publish/due dates, KEV ransomware flag |
| `package` | `(package_name, package_type)` | |
| `cluster` | `cluster_name` | Distribution, first/last seen |
| `namespace` | `(cluster_name, namespace_name)` | |
| `workload` | `(cluster, namespace, workload_type, workload_name)` | |
| `team` | `team_id` | Derived via ownership resolver (§4.5) |
| `owner` | `owner_id` | Derived via ownership resolver (§4.5) |

### 4.2 State log — `finding_state`

The heart of the model. Natural key: `(image_id, cve_id, package_name, package_version, package_path)`. Multiple rows per natural key over time, one per state transition.

Columns:
```
finding_id            synthetic PK
image_id, cve_id, package_name, package_version, package_path
severity, cvss_score, in_use, fix_available, fix_version, risk_accepted, public_exploit
first_seen            timestamp — first snapshot this finding appeared
last_seen             timestamp — most recent snapshot it was observed
state                 OPEN | CLOSED
reason_code           NULL while OPEN. On CLOSED: PATCHED | RETIRED | SCALED_TO_ZERO | ACCEPTED | FEED_WITHDRAWN | UNKNOWN
closed_at             timestamp — NULL while OPEN
reopened_at           timestamp — NULL if never
reopen_count          integer
days_open             computed — ingestion-populated for fast queries
is_regression         boolean — TRUE if reopen_count > 0
```

**Reason code logic** (computed at ingest when a finding transitions OPEN → CLOSED):
```
if risk_accepted flipped TRUE on same image_id      → ACCEPTED
elif image_id has newer sibling digest in same repo
     AND CVE not present in newer digest            → PATCHED
elif no workload currently runs this image_id       → RETIRED
elif CVE missing from today's feed entirely         → FEED_WITHDRAWN
else                                                 → UNKNOWN
```

`SCALED_TO_ZERO` is detected at workload level from `workload_runs_image_daily` snapshots, not from finding disappearance.

**UNKNOWN is first-class** — it's itself a data-quality signal ("findings vanished for no explainable reason").

**Drift columns** — fields allowed to change on a re-seen OPEN row without creating a new row: `severity`, `cvss_score`, `fix_available`, `fix_version`, `risk_accepted`, `public_exploit`, `in_use`. Changes are overwritten in-place (no drift history for v1). Everything else on the natural key is immutable.

### 4.3 Relationship tables (edges, time-scoped)

| Table | Keys | Notes |
|---|---|---|
| `image_in_repository` | `(image_id, repository, tag, first_seen, last_seen)` | |
| `workload_runs_image_daily` | `(date, cluster, namespace, workload_type, workload_name, container_name, image_id, replica_count)` | Daily runtime snapshot. Largest table (~1–2M rows / 6 months). |
| `namespace_in_cluster` | `(cluster, namespace)` | |
| `workload_in_namespace` | `(cluster, namespace, workload_type, workload_name)` | |
| `workload_owned_by` | `(workload_key, team_id, owner_id, resolved_by_strategy, resolved_from)` | Populated by ownership resolver |

### 4.4 Rollup cache (Layer 3 — reconstructible from 4.1–4.3)

Rebuilt on each ingest. Always safe to drop and recompute.

| Table | Grain |
|---|---|
| `daily_metrics_by_image` | `(date, image_id)` — counts by severity, count_open, count_new, count_fixed, count_regressed, mttr_sum, mttr_count |
| `daily_metrics_by_workload` | `(date, cluster, namespace, workload)` |
| `daily_metrics_by_team` | `(date, team_id)` |
| `daily_metrics_by_repository` | `(date, repository)` |
| `daily_metrics_by_cluster_severity` | `(date, cluster, severity)` |

### 4.5 Ownership resolver

Pluggable resolver chain evaluated per workload at ingest:

```
strategies: [
  LabelStrategy(label="team"),
  LabelStrategy(label="cost-center"),
  MappingFileStrategy(path="~/sysdig-vuln-data/ownership.csv"),
  NamespaceFallback()
]
```

First non-None wins. `resolved_by_strategy` and `resolved_from` stored with the attribution so "why was this workload attributed to team X?" is always answerable.

Mapping file is glob-matched CSV: `cluster,namespace,workload_type,workload_name,team,owner`. All columns accept `*` wildcards. `workload_type` may be left blank to match any type. Hot-reloaded each ingest. `sas reattribute` CLI re-runs the resolver over existing rows without touching findings data.

New strategies (GitHub CODEOWNERS, Confluence page, Sysdig ownership API) slot in as additional entries in the list — no model change.

## 5. Time semantics

Two clocks in play. Model them separately.

- **Observation clock** — when we ingested. Precise, from the report's timestamp. Stored as `snapshot_at`.
- **Knowledge clock** — when Sysdig last scanned each image. Lags per-image.

Every ingest creates one `snapshot_id` with one `snapshot_at`. Derived:
- `workload_runs_image_daily.date = DATE(snapshot_at)` — runtime snapshot is point-in-time.
- `finding_state.last_seen = snapshot_at` — finding was visible in this report.

**Widget semantics:**
- "Last 30 days" = last 30 snapshots, not 30 clock-days.
- Time-series rendered as **step functions**, not smoothed lines. Between snapshots we don't pretend to know.
- Missing snapshots render as **gaps** — blank X-axis positions, no interpolation.
- Every page shows `As of <snapshot_at>` — the real latest-ingest timestamp, never `now()`.
- Every X-axis has a **gap-coverage strip** showing which days had snapshots.

**Reason codes require 2+ snapshots.** First ingest produces no CLOSED rows (no prior state to diff). Expected and flagged.

**Stale-scan heuristic (replaces abandoned Theme 7 coverage assurance):** track `last_scan_seen_per_image` and surface images with stale scans as amber in the UI. Heuristic, not ground truth, labelled as such.

## 6. Ingestion pipeline

One CLI: `python -m sas.ingest <csv-file>`. Idempotent; re-running same CSV is a no-op.

```
1.  Validate CSV schema (reject on mismatch)
2.  Compute snapshot_id = hash(filename + row_count); respect --snapshot-id override
3.  Abort if snapshot_id already ingested, unless --force
4.  Load CSV to staging (pyarrow-fast)
5.  Upsert entity tables
6.  Run ownership resolver per workload → upsert workload_owned_by
7.  Write workload_runs_image_daily rows
8.  Diff findings against current OPEN state:
      - new natural keys → INSERT OPEN
      - re-seen keys    → UPDATE last_seen, drift columns
      - reopened keys   → new row with reopened_at, reopen_count++
      - disappeared     → UPDATE state=CLOSED, compute reason_code
9.  Rebuild daily_metrics_* rollups for affected dates
10. Append to ingest_log (counts, timing, snapshot_id)
```

**Performance target:** <60s for a 1M-row CSV on a laptop. Raw CSV not retained — deduplicated state log is the only persistence.

**Stage 1:** manual invocation by user. **Stage 2 (future):** same CLI as K8s CronJob entrypoint, zero code change.

## 7. Query primitive

The intellectual core. 5-tuple that compiles to DuckDB SQL.

```python
@dataclass
class Query:
    lens: Lens                    # registry: Image | CVE | Workload | Cluster | Namespace | Package | Repository | Team | Owner
    traversal: list[Edge]         # registry — edges to walk
    time: TimeWindow              # last_n_snapshots | date_range | all_time; granularity: day|week|month|quarter
    measure: Measure              # registry — count_distinct, count_open, count_new, count_fixed, count_regressed, mttr
    filters: list[Filter]         # (field, operator, value) tuples
    group_by: list[Dimension] = []
    order_by: Ordering | None = None
    limit: int | None = None
```

**Lens, Edge, Measure are registries, not enums.** v1 ships nine lenses, the obvious edges between them, and six measures. New measures (e.g., `p95_cvss`, `stddev_days_open`) are additive — a ~30-line class implementing `build_select_sql(context)` + `required_columns`, plus a registration call. Zero core changes, zero breaking changes to existing widgets.

**Result shape — always `QueryResult`:**
```python
@dataclass
class QueryResult:
    series: list[Series]          # one per group_by combination
    dimensions: dict[str, list]
    snapshot_range: tuple[date, date]
    missing_days: list[date]
    exec_time_ms: int

@dataclass
class Series:
    key: dict                     # e.g., {"severity": "Critical"}
    x: list[date]
    y: list[float | int]
```

**Compilation rules:**
- Filter values are parameterised — SQL injection structurally impossible.
- **Rollup routing:** if measure + filters can be satisfied by a `daily_metrics_*` table, hit the rollup (10–100× faster). Else fall through to `finding_state`. UI never knows.
- SQL compiler is a single pure function `compile(Query) -> str`.

**Frontend mirror:** the TypeScript types are generated from the FastAPI OpenAPI spec. Widget Builder UI is a form that constructs a `Query` and POSTs to `/api/query`. The primitive is the UI ↔ backend contract.

**Deliberate exclusion:** no arbitrary JOINs, no correlated subqueries, no user-exposed SQL. Finite vocabulary is the whole point. New capabilities = new registered measure, not expanded DSL.

## 8. Widget catalog (v1)

Ten starter widgets, curated. Every one maps to a specific question from the adversarial brainstorm. The Builder can clone any of these as a template; users then tweak.

| # | Widget | Answers |
|---|---|---|
| 1 | **Image Remediation Story** (flagship) | Matt's core question |
| 2 | **Fleet Critical Trend** | Dan's "are we more secure than 6 months ago" |
| 3 | **New vs Fixed vs Regressed** (composite) | Matt's "fixing stuff or did new ones appear" |
| 4 | **Team Accountability Leaderboard** | Dan's exec view + auditor gaming detection |
| 5 | **Tag Lineage View** | Wiz "CVE across tag lineage" |
| 6 | **KEV-Ransomware Exposure Trend** | CrowdStrike "ransomware residual" |
| 7 | **CVE Blast Radius Timeline** | "Log4j-class drops — where is it?" |
| 8 | **Image Inventory Grid** | Matt's daily working surface |
| 9 | **Repository Tag Hygiene** | Auditor's "tags churning silently" |
| 10 | **Findings Table** (flat, filterable) | Universal escape hatch |

**Tier-1 widgets — Widgets 1, 4, 5, 7.** Four of the ten have custom layouts and are the showpieces. The other six (2, 3, 6, 8, 9, 10) are thin Query-renderer pairs — they render a Query's result via the standard chart/table components with minimal layout code.

**Widget 1 — Image Remediation Story** (the flagship answering Matt):
- Input: pick an image or search by repository.
- Top: step-line of open criticals over N snapshots.
- Overlaid bars: `count_fixed(reason=PATCHED)` green, `count_fixed(reason=RETIRED)` grey, `count_regressed` red.
- Side panel: tag lineage mini-view with per-tag critical counts.
- Annotations: auto-labelled events (digest change, new tag).
- Footer sentence (generated from the data): *"In the last 90 days, 12 criticals were patched via new digests, 3 disappeared because the image was retired, and 1 regressed when v1.3.0 shipped."*

**Widget 7 — CVE Blast Radius Timeline:** pick a CVE → Gantt-style timeline of every image + workload that had it, colored by team, click to drill.

**Widget 4 — Team Accountability Leaderboard:** card per team with MTTR, SLA compliance %, reason-code donut (patched/retired/accepted), integrity-score flag when ratios are suspicious. Click for team detail.

**Widget 5 — Tag Lineage View:** pick a repository → horizontal stacked bar per tag, chronological, hover for per-tag diff vs previous (+3 criticals, −7 fixed, 2 regressed).

## 9. Frontend UX

Three top-level spaces, tabs in the top bar. No deeper hierarchy.

1. **Dashboard** — landing. Curated widgets above the fold: Image Remediation Story, Fleet Critical Trend, Team Leaderboard.
2. **Explore** — Widget Builder, Findings Table, CVE Explorer, Image Inventory. Left-rail list.
3. **Admin** — ownership rules editor, ingestion history, config.

**Layout grammar (every page):**
- Top bar 40px — logo, tabs, search, as-of timestamp, user menu. Dark, permanent.
- Breadcrumb strip 24px — drill-in trail, clickable.
- Content area — widget grid on Dashboard, single-focus on Explore, forms on Admin.
- Optional right rail when a widget is selected — "what this widget is doing" + filters + export/share.

No sidebars competing for attention. No hamburger menus. No FABs. Modals only for genuine dialogs.

**Drill-in mechanism (the feature):**
- Every chart data point is clickable.
- Click opens a drill-overlay sliding from the right, ~60% viewport. Underlying view stays dimmed.
- Drill-overlays contain their own clickable widgets, nesting freely.
- Breadcrumb strip reflects the full stack: `Dashboard > Fleet Trend > 2026-04-15 > frontend-app:v2.1.3 > repository lineage`.
- Click a crumb to unwind. Escape to pop one level.
- **Stack is stored in URL query params** — every drill-in state is a shareable link.

**Honesty strip:**
- Gap-coverage strip under every X-axis (filled = ingested, blank = missing).
- Every page header shows real `As of <timestamp>`, never `now()`.
- Stale images (>14 days no scan) marked amber in inventory views.

**Density — Gmail compact:**
- Table row 32–36px. Card padding 12px. Grid gutters 8–10px. Form field gap 8px.
- Top bar 40px. Breadcrumb strip 24px.
- Chart margins 8–12px, not library defaults.
- Card number 28–30px, label 11px uppercase, delta 12px. 8px vertical rhythm inside cards.

**Motion:**
- All transitions 180ms ease-out.
- Drill-overlays slide, don't pop.
- Charts animate on initial render only.
- Loading: skeletons, not spinners. No layout shift.

**Typography / color:**
- Inter, 400/500/600/700.
- Numbers: tabular lining figures (columnar alignment).
- 12-step neutral gray scale, dark theme first. Brand accent: Sysdig green (formal palette pending).
- Severity: red/orange/amber/slate. No other categorical colors. Multi-series uses monochromatic brand-green ramp.
- Light mode is a toggle; dark is the default.

**Empty & error states:**
- Fresh instance, no data → single centered card, "Ingest your first Sysdig export." Drag-drop supported.
- <7 snapshots → widgets render with a note: "Trends will develop. You have N snapshots; 7+ for meaningful lines."
- Empty filter result → never "no data." Always: "No findings match these filters. Try removing `severity = Critical`."

**Accessibility baseline:**
- Full keyboard navigation (tabs, arrows, `/` search, Escape unwind).
- ARIA labels on all interactive elements.
- "View as table" toggle on any time-series for screen-reader parity.

### 9.1 Design tokens (single source of truth)

**Every visual constant in the UI is a named token in one file.** Components reference tokens by name (`p-card`, `h-row`, `text-metric`), never raw values. Tuning the UI = editing one file, not grep-and-replace across components.

**Token file:** `sas/web/styles/tokens.css` (plus `tailwind.config.ts` that exposes them as utility classes).

**Token groups:**

| Group | Examples | Purpose |
|---|---|---|
| `density` | `--p-card`, `--p-cell`, `--gap-widget`, `--gap-field`, `--h-row`, `--h-topbar`, `--h-breadcrumb`, `--h-card-number` | All spacing/sizing — turn the whole UI denser or roomier by editing these |
| `color` | `--neutral-0` through `--neutral-12`, `--brand-primary`, `--severity-critical/high/medium/low`, `--semantic-success/warn/error/info`, `--bg-base`, `--bg-surface`, `--bg-overlay`, `--fg-primary`, `--fg-muted`, `--border-subtle`, `--border-strong` | All color — swap themes, adjust contrast, rebrand |
| `typography` | `--font-family`, `--fw-regular/medium/semibold/bold`, `--fs-micro/small/base/metric/display`, `--lh-tight/base/loose`, `--letter-spacing-caps` | All type |
| `motion` | `--dur-fast/standard/slow`, `--ease-standard`, `--ease-accel`, `--ease-decel` | All transitions |
| `elevation` | `--shadow-card`, `--shadow-overlay`, `--shadow-drill` | Z-axis |
| `border` | `--radius-sm/md/lg`, `--border-width-thin/base` | Shape |

**Severity tokens pair with intent, not absolute color names.** A component uses `bg-severity-critical`, never `bg-red-600`. If we later decide critical should be a different red (for colorblind accessibility, for brand fit), one edit propagates.

**Numbers from §9 as tokens:**
```
--h-row: 34px              (table row height)
--p-card: 12px             (card padding)
--gap-widget: 10px         (widget grid gutter)
--gap-field: 8px           (form field spacing)
--h-topbar: 40px
--h-breadcrumb: 24px
--fs-metric: 30px          (big numbers in cards)
--fs-metric-label: 11px    (uppercase labels)
--dur-standard: 180ms
--ease-standard: cubic-bezier(0.2, 0, 0, 1)
```

When Aaron sees the live UI and wants it denser/roomier, lighter/heavier, faster/slower — the change is a single-file edit, not a spelunking exercise.

**Extensibility:** a `data-density="compact|cozy|comfortable"` attribute on `<body>` can swap token sets at runtime if a density switcher is ever wanted. Not in v1, but the architecture admits it trivially.

## 10. Phased delivery

Strict sequential. Each phase ends with something concrete.

### Phase 1 — Data foundation
DuckDB schema, `sas ingest` CLI, ownership resolver, state-log diff with reason codes, rollup rebuild, integration tests. Real `phoenix-vuln-findings-2026_04_23.csv` sample ingests cleanly.
**Dispatchable to Sonnet:** fully.
**Demoable:** to Aaron via CLI + SQL inspection. Not customer-facing.

### Phase 2 — Query engine + API
`Query` dataclass + registries, SQL compiler, six measures, rollup routing, FastAPI `/api/query` + `/api/widgets/catalog` + `/api/entities/<lens>`, auto-generated OpenAPI spec, integration tests.
**Dispatchable to Sonnet:** mostly. Registry + compiler architecture stays with Opus; concrete measures/endpoints to Sonnet.
**Demoable:** to Aaron via curl + `/docs`. Not customer-facing.

### Phase 3 — Dashboard (first customer demo)
Next.js scaffold, **design tokens defined first (§9.1)**, typed API client, dashboard layout (top bar, breadcrumb strip, content grid), 10 starter widgets rendering real data, 4 tier-1 widget custom layouts, drill-in overlay + breadcrumb stack + URL sync, as-of header, gap-coverage strip under every X-axis, stale-scan amber markers in image views, empty states, dark-first styling, motion per §9.
**Dispatchable to Sonnet:** layout/routing/state stays with Opus; individual widgets to Sonnet with specs.
**Demoable:** YES. Loris/Matt/Dan demo moment.

### Phase 4 — Widget Builder + polish
Widget Builder UI (5-tuple form), clone-from-template, save to custom dashboard, export/share as URL/JSON, Admin pages (ownership rules, ingest history), comprehensive empty/loading/error states, accessibility audit, performance audit (drill-in p95 <200ms).
**Dispatchable to Sonnet:** mostly routine React/forms.
**Demoable:** CISO-ready.

### Dependencies
Phases strictly sequential. Within Phases 3–4, tasks parallelize across Sonnet workers via `superpowers:subagent-driven-development`.

## 11. Out of scope for v1

Explicit exclusions, acknowledged and deferred:

- **Coverage assurance** (Theme 7 from the brainstorm — 10 questions). We can only chart what the CSV contains. Future feed/API integration.
- **Runtime/network/lateral-movement joins** (Theme 10). Not in CSV. Future integration with Sysdig runtime graph.
- **Automated daily ingestion** — Stage 2; K8s CronJob.
- **PDF export** — reuses existing `pdf_generator.py` in a later phase.
- **Multi-tenancy** — single-customer per deployment by design.
- **Authentication** — v1 runs on trusted network / laptop demo.
- **Alerting / notifications** — wrong product scope.
- **Second-source CVE feed validation** (Auditor-14) — future.

## 12. Collaboration pattern

- **Opus 4.7** as brain: spec, implementation plan, code review, architecture decisions, debugging subtle issues.
- **Sonnet 4.6 workers** for well-scoped implementation tasks (components, modules, subsystems). Dispatched via `superpowers:subagent-driven-development`.
- **Haiku** generally too weak for this project's work.
- Task sizing: delegate *chunks*, not individual functions. Briefing + review has floor cost.
- If a Sonnet worker returns unclean output, Opus takes the task back rather than compromise quality.
- User (Aaron) manages all commits manually — no auto-commits.

## 13. References

- Customer pain context: [`problem.md`](../../../problem.md)
- Adversarial brainstorm (60 questions from 4 personas, 12-theme synthesis): [`2026-04-23-adversarial-brainstorm.md`](../research/2026-04-23-adversarial-brainstorm.md)
- Sample CSV: [`phoenix-vuln-findings-2026_04_23.csv`](../../../phoenix-vuln-findings-2026_04_23.csv)
- Project memory: `~/.claude/projects/-Users-aaron-miles-GitHub-sysdig-report-studio/memory/historical_analytics_project.md`
