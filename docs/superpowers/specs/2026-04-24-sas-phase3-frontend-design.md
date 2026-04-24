# SAS Phase 3 — Frontend Design Spec

**Status:** Approved for implementation.
**Date:** 2026-04-24.
**Owner:** Aaron Miles (aaron.miles@sysdig.com).
**Phase:** 3 of 5. Depends on Phase 2 (FastAPI backend) being complete.
**Budget:** ~$85–110. Leaves ~$390 headroom inside the $500 ceiling.

---

## 1. Context

Phase 3 builds the Next.js frontend that consumes the Phase 2 FastAPI backend. The backend contract is complete: `POST /api/query`, `GET /api/widgets/catalog`, `GET /api/entities/{lens}`, `GET /healthz`. Phase 3 has one job: turn that contract into a demo-ready UI.

Two things must land in Phase 3 to count as success. First, the Image Remediation Story widget must be fully functional and visually compelling — this is the direct answer to Matt's renewal-blocking question. Second, the graph-based data model must be visible to users through the Graph Explorer, demonstrating that SAS thinks in entities and relationships rather than flat rows. A secondary goal is overall polish sufficient to "wow" Loris during a first walkthrough: no rough edges, no empty states that read as half-built, no layout jank.

---

## 2. Tenets

Phase 3 inherits all tenets from the main design doc §2 without modification. The following additions apply specifically to frontend work:

- **Light mode is primary per Sysdig brand.** The main spec defaults dark; the brand (see `sas/branding.md` and `sas/sysdig-brand.css`) is white body with Deep See accents. Light mode ships as the default. Dark mode is a body-level toggle that respects `prefers-color-scheme` on first load; choice persists in `localStorage`.
- **Page-stack drill-in is the aspiration, not the contract.** Visual stacking with perspective transform is the approved target. If implementation cost during Phase 3.3 exceeds plain routing by more than ~$5 of budget, revert to full-page navigation. The drill-in URL structure and breadcrumb logic are the same either way — only the animation differs.
- **Gmail Compact density.** 15–20% tighter than native Sysdig UI. Every density value lives in `tokens.css`. All widgets use the same token set — no widget gets special spacing exemptions.
- **Two-pass iteration is budgeted.** UI always needs 2–3 rounds. Phase 3.1 ships something testable early. Visual tuning rounds are expected, not failures.

---

## 3. Stack

All packages must be free/open-source. Versions are minimums; use the current stable release at implementation time.

| Layer | Technology | Notes |
|---|---|---|
| Framework | Next.js 14+ (App Router, TypeScript) | Server components for data fetching; client components for interactivity |
| UI primitives | shadcn/ui + Tailwind CSS | shadcn components unstyled by default — customize via tokens |
| Charting | Apache ECharts (via echarts-for-react) | All 10 widgets. ECharts chosen for density configurability and SVG output for PDF |
| Graph explorer | react-flow | 2D v1. `dagre` or `elkjs` auto-layout |
| State management | Zustand | Lightweight; one store per domain (drill-in stack, filter state, dashboard layout) |
| Tables | TanStack Table v8 | Findings Table + Image Inventory Grid |
| Animation | framer-motion | Page-stack transitions and drill-in open/close |
| API client | openapi-typescript-codegen (or `openapi-ts`) | Generated at build time from FastAPI's `/openapi.json` |
| Font | Inter (Google Fonts, weights 300–700) | Loaded via `next/font/google` for performance |
| PDF export | Via `POST /api/export/pdf` (backend renders, frontend downloads) | No client-side PDF rendering |
| Dashboard metadata | SQLite via existing `database.py` pattern; new `dashboards` table | Not the DuckDB analytics store |

**Language boundary rule:** Python for anything touching data. TypeScript for everything in `sas/web/`. FastAPI is the typed bridge. No TypeScript code imports from `sas/api/` or `sas/ingest/`.

---

## 4. Information architecture

Three top-level spaces, sidebar navigation. Admin is deferred to Phase 4.

**Dashboard** — landing space. Curated starter layout with the Image Remediation Story prominent above the fold. Second row: Fleet Critical Trend + Team Accountability Leaderboard. Third row: KEV-Ransomware Exposure Trend + New vs Fixed vs Regressed. All widgets load real Phase 2 API data.

**Explore** — sub-items in the left rail (no nested dropdowns; flat list with divider labels):
- Image Inventory
- CVE Explorer
- Findings Table
- Widget Builder (clone & edit — Phase 3.5)
- Graph Explorer

**Admin** — sidebar item present but routes to a "Coming in Phase 4" placeholder. Don't hide the nav item; users need to see the intended scope.

**My Dashboards** — expandable section at bottom of sidebar, above user info. Lists saved dashboards by name. "+" button inline with the section label creates a new dashboard. Clicking a dashboard name navigates to `/dashboard/my/{id}`. Maximum 5 visible before "Show N more" expansion — prevents sidebar overflow.

---

## 5. Routing and URL structure

Next.js App Router. All paths are deep-linkable and shareable. Every drill-in level is one `router.push()` entry — browser back unwinds one level.

```
/signin                                                   cosmetic auth
/dashboard                                                default curated dashboard
/dashboard/my/{dashboard-id}                              saved dashboard

/explore/image/{image-id}                                 image detail
/explore/image/{image-id}/cve/{cve-id}                    CVE within image
/explore/cve/{cve-id}                                     CVE detail
/explore/workload/{cluster}/{namespace}/{type}/{name}      workload detail
/explore/team/{team-id}                                   team detail

/explore/graph                                            Graph Explorer (no focus entity)
/explore/graph/{lens}/{entity-id}                         Graph Explorer centered on entity

/explore/widget/new?template={widget-id}                  Widget Builder, pre-seeded from template
```

Top-level nav transitions (Dashboard → Explore, Explore → Dashboard) are full-screen route changes — no page stack, no animation carry-over.

Drill-ins within a space push history entries. The page-stack visual effect (§10) applies only to these intra-space drills. Each URL in the stack is independently bookmarkable; navigating directly to a drill URL loads that entity's detail page as a full page (no stack behind it).

**URL construction rules:**
- `image-id` = URL-encoded sha256 digest (first 12 chars displayed in UI; full digest in URL)
- `cluster`, `namespace`, `type`, `name` = URL-encoded; slashes in names encoded as `%2F`
- `dashboard-id` = UUID v4, assigned at save time
- `lens` in graph URLs = one of: `image`, `cve`, `workload`, `cluster`, `namespace`, `package`, `repository`, `team`, `owner`

---

## 6. Design tokens

Token file at `sas/web/styles/tokens.css`. This is the single source of truth for every visual constant. Components reference tokens by CSS custom property name — never raw values. `tailwind.config.ts` exposes all tokens as Tailwind utility classes.

Inherits all token groups from main spec §9.1 (density, color, typography, motion, elevation, border). Phase 3 specific values below override or extend:

```css
/* Density — Phase 3 values (tighter than main spec §9.1 defaults) */
--h-row: 32px;              /* table rows — 2px tighter than §9.1 */
--p-card: 14px;             /* card internal padding */
--gap-widget: 10px;         /* grid gutter between widgets */
--h-topbar: 48px;           /* lightweight page header — not the 40px top-bar from §9 */
--h-sidebar-row: 30px;      /* sidebar nav item height */
--h-breadcrumb: 28px;       /* breadcrumb strip */
--radius: 8px;              /* one border-radius value used everywhere */
--shadow-card: 0 1px 2px rgba(0,0,0,0.05);   /* subtly lifted card */
--dur-standard: 180ms;
--ease-standard: cubic-bezier(0.2, 0, 0, 1);

/* Sysdig brand palette (mirrors sas/sysdig-brand.css) */
--deep-see: #01353E;
--lumin: #BDF78B;
--grey-10: #EAEBED;
--grey-20: #D4D6D9;
--grey-70: #4A4D53;
--grey-80: #26282E;
--grey-90: #121217;
--falco-blue: #00CBE2;
--red: #FF7774;
--orange: #FFA940;
--yellow: #FDD835;

/* Semantic surface tokens — light mode */
--bg-base: var(--white);
--bg-surface: var(--grey-10);
--bg-sidebar: var(--deep-see);
--bg-sidebar-active: rgba(189,247,139,0.18);   /* 18% Lumin opacity */
--fg-primary: var(--black);
--fg-muted: var(--grey-60);
--border-subtle: var(--grey-20);
--border-strong: var(--grey-40);
--accent: var(--lumin);

/* Severity tokens — pair with intent, not color names */
--severity-critical: var(--red);
--severity-high: var(--orange);
--severity-medium: var(--yellow);
--severity-low: var(--grey-50);
```

Dark mode via `[data-theme="dark"]` on `<body>`. Swaps `--bg-base`, `--bg-surface`, `--bg-sidebar`, `--fg-primary`, `--fg-muted`, `--border-subtle`, `--border-strong` to their dark equivalents. Severity and brand tokens do not change between modes — they are perceptually adjusted by the semantic surface tokens.

```css
[data-theme="dark"] {
  --bg-base: var(--grey-80);
  --bg-surface: var(--grey-90);
  --bg-sidebar: var(--grey-90);
  --fg-primary: var(--white);
  --fg-muted: var(--grey-50);
  --border-subtle: var(--grey-70);
  --border-strong: var(--grey-60);
}
```

---

## 7. Layout grammar

Based on approved mockup: `layout-b-refined.html`.

**Left sidebar** — 180px wide, fixed. `var(--bg-sidebar)` background (Deep See in light, Grey-90 in dark).
- Top: Sysdig logo (white variant on Deep See) + "ANALYTICS STUDIO" label in 10px uppercase Grey-40 tracking.
- Nav items: icon + label, `var(--h-sidebar-row)` height, `var(--radius)` border-radius on active pill. Active state: `var(--bg-sidebar-active)`. Inactive hover: 8% white overlay.
- Divider labels between nav groups (e.g. "EXPLORE"): 10px uppercase, Grey-50, 20px top margin.
- My Dashboards section: collapsible, below main nav, above user info.
- User info pinned at bottom: avatar (24px circle) + email truncated, sign out icon on right.

**Main content area** — fills remaining viewport. `var(--bg-base)` background.
- Page header: `var(--h-topbar)` height. Left: page title (14px medium). Right: "As of {timestamp}" in 12px muted + dark mode toggle icon + 3-dot overflow.
- Breadcrumb strip: `var(--h-breadcrumb)` height, only present when drilled in. Clickable crumbs separated by `/`. Entire strip slides down on drill-in, up on unwind.
- Content grid: CSS grid, 12-column. Widgets span 4, 6, or 12 columns. Gap: `var(--gap-widget)`.

No floating action buttons. No right rails competing with content. Modals only for genuine dialogs (confirm delete, clone widget). Drawers (half-screen panels) for drill-in detail pages in the page-stack fallback.

---

## 8. Widget card shell

Every widget — all 10 — uses the same card shell. The shell is a presentational component; the chart area inside it varies.

**Card anatomy (top to bottom):**
1. Label row: 10px uppercase muted text left (e.g. "IMAGE METRICS") + 3-dot action menu right. Row height: 24px.
2. Title row: 13px medium text. Row height: 20px.
3. Chart area: variable height, min 180px.
4. Footer (optional): 11px muted narrative text. 1-line max before truncate + expand link. Row height: 20px when present.

**3-dot menu items (standard, every widget):**
- Clone & edit filters (opens Widget Builder "lite")
- Export as PDF (calls `POST /api/export/pdf` for this widget)
- Copy widget link (copies current URL with widget focus)

`var(--shadow-card)` on card. `var(--radius)` corners. `var(--p-card)` internal padding. On hover: border transitions from `var(--border-subtle)` to `var(--border-strong)` in `var(--dur-standard)`.

Loading state: skeleton shimmer fills chart area. No spinner. No layout shift — skeleton matches final chart dimensions exactly.

---

## 9. The 10 starter widgets

All widgets call `POST /api/query` with a `Query` body constructed from their defaults. Filters, time window, and group-by are editable via Widget Builder "lite" (§12).

### Tier-1 widgets (custom layouts)

**Widget 1 — Image Remediation Story** (flagship)

Based on approved mockup `widget-remediation-story-v5.html`.

Layout: two stacked ECharts panels sharing an X-axis, plus a right panel for tag lineage.

*Chart 1 (top, taller):* ECharts stacked bar + line combo. X-axis = snapshots (default: last 90). Stacked bars: `count_fixed(reason=PATCHED)` in Lumin green, `count_fixed(reason=RETIRED)` in Grey-40, `count_fixed(reason=ACCEPTED)` in Grey-50, `count_fixed(reason=UNKNOWN)` in Grey-30. Overlaid line: `count_open(severity=Critical)` in Deep See. Y-axis left = bar scale. Y-axis right = line scale (independent, labeled in muted text).

*Axis-labels toggle:* off by default. When toggled on, weekly cadence for 90-day windows. Labels rotate 45°. Chart reserves 44px at bottom for rotated labels when toggle is on. The line chart's month-anchor labels (Feb 1 / Mar 1 / Apr 1) stay horizontal — only the dense bar X-axis rotates. See §9 below for full cadence rules.

*Chart 2 (bottom, shorter):* ECharts bar chart, same X-axis. Two series: `count_new` in soft red (`#FF7774` at 60% opacity) and `count_fixed` in Lumin green. Shows net flow at each snapshot. Bar height proportional to count.

*Right panel (220px):* Tag Lineage mini-view (Widget 5 logic embedded). Shows chronological tag list with per-tag critical count spark bar. Clicking a tag row drills to `/explore/image/{image-id}` for that tag's digest.

*Footer:* Auto-generated narrative sentence. Template: "In the last {N} days, {patched} criticals were patched via new digests, {retired} closed because the image was retired, and {regressed} regressed." Generated from `QueryResult` data, not AI — pure arithmetic on series values.

*Annotations:* Digest-change events rendered as vertical dashed lines on Chart 1. Auto-detected: snapshot where `image_id` changed on the same repository+tag. Label: "Digest changed".

**Widget 4 — Team Accountability Leaderboard**

Grid of team cards. Each card: team name (13px bold) + MTTR median (28px tabular) + "days avg" label + SLA compliance percentage. Below: 60px donut chart (reason-code breakdown: Patched / Retired / Accepted / Unknown). Click card → drills to `/explore/team/{team-id}`.

Gaming flag: if `count_fixed(reason=RETIRED) / count_fixed(total) > 0.4` for a team, amber badge "Review: high retire ratio". If `count_fixed(reason=ACCEPTED) / count_fixed(total) > 0.25`, amber badge "Review: high accept ratio". Hover shows flag reasoning. Never blocks or hides — it's an advisory signal.

Sort control: MTTR ascending/descending | SLA% | name. Persists in Zustand store scoped to this widget.

**Widget 5 — Tag Lineage View**

Horizontal stacked bar per tag, one bar = one snapshot window. X-axis = snapshots. Bars stacked: Critical / High / Medium / Low (severity token colors). Tags ordered chronologically by `first_seen` descending. Hover tooltip: "+{new} critical, -{fixed} fixed, {regressed} regressed vs previous snapshot". Click bar → drills to `/explore/image/{image-id}` for that tag's digest at that snapshot.

Input: repository picker (typeahead, calls `GET /api/entities/repository`). Defaults to repository with most findings in current window.

**Widget 7 — CVE Blast Radius Timeline**

Gantt-style ECharts chart. Y-axis = image+workload pairs affected by the selected CVE. X-axis = date range. Each bar = duration this CVE was `OPEN` on that image. Bar color = team (up to 8 distinct team colors from a fixed palette; overflow → Grey-40). Click bar → drills to `/explore/image/{image-id}/cve/{cve-id}`.

Input: CVE picker (typeahead, calls `GET /api/entities/cve`). Defaults to the highest-CVSS currently-open CVE with the most affected images.

### Thin widgets (standard ECharts + card shell)

**Widget 2 — Fleet Critical Trend.** Line chart. `count_open(severity=Critical)` over time, fleet-wide. Single series, Deep See color. Step-line rendering (not smooth — honesty tenet). Gap markers where snapshots are missing.

**Widget 3 — New vs Fixed vs Regressed (composite).** Three line series on one chart. New = red, Fixed = Lumin green, Regressed = orange. Shared Y-axis. Legend below chart. Shows whether the fleet is trending toward net-positive.

**Widget 6 — KEV-Ransomware Exposure Trend.** Line chart. `count_open(kev=true, ransomware=true)` over time. Single alarming-red series. Footer: "These are CISA KEV ransomware-associated CVEs currently open in your fleet."

**Widget 8 — Image Inventory Grid.** TanStack Table. Columns: Image (truncated digest + tag), Repository, Critical, High, Medium, Low, Last Seen, Stale (amber indicator if >14 days). Default sort: Critical descending. Sortable + filterable. Row click → drills to `/explore/image/{image-id}`. Stale images: row gets 2px left border in `--yellow`.

**Widget 9 — Repository Tag Hygiene.** Bar chart per repository. Each bar = count of tags seen in last 30 snapshots. Grouped: active tags (currently running in a workload) vs inactive. High inactive count = hygiene risk. Tooltip shows tag names. Click bar → drills to Tag Lineage View (Widget 5) for that repository.

**Widget 10 — Findings Table.** TanStack Table. All OPEN findings, filterable by severity / CVE / image / team / KEV / hasFix. Virtualized for large datasets. Columns: CVE, Image, Package, Severity, Days Open, Fix Available, KEV, Team. Sortable on all columns. Row click → `/explore/cve/{cve-id}` or `/explore/image/{image-id}` depending on which cell is clicked.

---

## 10. Axis-labels toggle principle

Time-series widgets dense with data (Widgets 1, 2, 3, 6) include an axis-labels toggle in the card action row — a calendar icon, not buried in the 3-dot menu. Default: OFF.

When ON:
- 90-day window → weekly cadence (one label per 7 snapshots). Labels rotate 45°.
- 30-day window → every 3 days. Labels rotate 45°.
- 7-day window → daily. Labels stay horizontal.
- Cadence is computed from actual snapshot count in `QueryResult.snapshot_range` — not assumed from calendar days.

When labels are rotated, the chart reserves 44px additional SVG height at the bottom to prevent label-chart overlap. This height delta is applied as a CSS variable (`--echarts-xaxis-rotate-reserve: 44px`) on the chart container — the ECharts `grid.bottom` option reads it. Every chart that supports axis-label rotation must implement this reserve — no exceptions.

Chart 2 in the Image Remediation Story (anchor month labels: Feb 1 / Mar 1 / Apr 1) always stays horizontal regardless of toggle state. Only Chart 1 (dense bar series) rotates.

---

## 11. Drill-in: page-stack pattern

Based on approved mockup `page-stack-drillin.html`. Option B locked.

**Top-level navigation** (Dashboard ↔ Explore ↔ Admin) = full-screen route change, no stacking.

**Intra-space drill-ins** (clicking a data point within a view) = page-stack behavior:
- Prior page scales to ~93% and shifts back in Z via CSS `perspective` + `scale3d` transform.
- New detail page slides in from right, covering the prior page.
- Up to 3 pages visible simultaneously (each scaled ~7% smaller than the one in front). Older pages beyond 3 collapse into a "3 more" chip in the bottom-left.
- Close button (× top-right of detail page), Escape key, or browser back unwinds one level with reverse animation (framer-motion `AnimatePresence`).
- Each drill pushes one `router.push()` history entry. Each close pops one.

**Breadcrumb strip** appears at top of the foreground page when depth > 1. Path reflects the full logical drill trail. Each crumb is clickable (unwinds to that level). Format: `Dashboard > Fleet Trend > 2026-04-15 > frontend-app:v2.1.3`.

**Stack state in Zustand** — `drillStack: DrillFrame[]` where `DrillFrame = { path: string; title: string; entityType: string; entityId: string }`. URL is the canonical source of truth; Zustand mirrors it for animation state. On page load with a drill-in URL, stack is reconstructed from URL segments (no animation on initial load — just render the page at that depth).

**Fallback protocol:** if framer-motion page-stack implementation during Phase 3.3 exceeds $5 budget over plain routing, disable the CSS transform animation and revert to plain `router.push()` with no visual stacking. URL structure, Zustand store, and breadcrumb behavior remain identical. Record the fallback decision in a code comment — it can be upgraded in Phase 3.x without structural changes.

---

## 12. Graph Explorer (2D v1)

Based on approved mockup `graph-explorer.html`. 3D deferred.

**Technology:** react-flow with `@dagrejs/dagre` or `elkjs` for auto-layout. Full-screen canvas within the `/explore/graph` route.

**Initial state:** no focus entity. Shows a "Type to search" centered prompt. Typeahead input (top-left overlay on canvas) accepts entity name or ID. On selection, graph loads with that entity as the center node.

**Graph behavior:**
- Center node = focus entity, styled with Deep See fill + Lumin ring.
- First-degree neighbors rendered around center. Second-degree neighbors rendered dimmer at outer ring.
- Edge labels visible by default: `AFFECTS`, `RUNS IN`, `IN PACKAGE`, `SAME PKG`, `OWNED BY`.
- Click a neighbor node = recenter (new focus, graph re-animates with new center). Each recenter is a `router.push` to `/explore/graph/{lens}/{entity-id}`.
- Pan + zoom via built-in react-flow controls. Double-click canvas = zoom to fit.

**Inspector panel** — 220px right-side panel, slides in when a node is focused:
- Entity type badge + full entity ID
- Key attributes (varies by lens: for Image → digest, repository, first_seen, last_seen, critical count; for CVE → CVSS, KEV, fix available; for Team → MTTR, SLA%)
- Action buttons: "Trend 90d" (opens Widget Builder pre-seeded with this entity as filter), "Show all affected" (opens Findings Table filtered to this entity), "Create Jira stub" (stubs a Jira ticket — text only in Phase 3, no API call)

**Navigation trail** — horizontal chip row at bottom of canvas. Shows walk history: entity chips in order, clickable. Newest chip on right. Older chips fade slightly. Recentering adds a chip; browser back removes the last chip.

**Node types and visual encoding:**
| Entity | Shape | Fill |
|---|---|---|
| Image | Rounded rectangle | White, Deep See border |
| CVE | Diamond | Severity token color |
| Workload | Rectangle | Grey-10, Falco Blue border |
| Repository | Cylinder-ish (stadium) | Grey-20 |
| Team | Hexagon | Lumin fill |
| Package | Circle | Grey-30 |

**Performance constraint:** limit graph render to 150 nodes. If query returns more, show the top 150 by edge count with a notice: "Graph truncated to 150 nodes. Use filters to narrow." This is a hard guard — react-flow degrades at high node counts.

---

## 13. Widget Builder "lite" (clone and edit)

Every starter widget's 3-dot menu includes "Clone & edit filters". Opens a half-screen right drawer (not a modal — it needs vertical space).

Drawer contents:
- Widget name field (pre-filled with "Copy of {original name}")
- Time window selector: Last 7 / 30 / 90 snapshots | Custom date range
- Severity filter: multi-select (Critical / High / Medium / Low)
- KEV toggle: All / KEV only / Non-KEV
- hasFix toggle: All / Fix available / No fix
- Team filter: typeahead multi-select (calls `GET /api/entities/team`)
- Image/Repository filter: typeahead (calls `GET /api/entities/image` or `GET /api/entities/repository` depending on widget lens)

"Save to dashboard" button: opens a sub-panel to pick destination dashboard (existing from My Dashboards list, or "New dashboard..."). Also: "Save & add to current dashboard" if viewing a saved dashboard.

Full from-scratch Widget Builder (change lens, measure, group-by, chart type) is explicitly deferred to Phase 4. The clone path is the only entry point in Phase 3.

---

## 14. Dashboard save and list

**API endpoints (new — to be added to Phase 2 API):**
```
GET    /api/dashboards              list all saved dashboards (name, id, updated_at)
GET    /api/dashboards/{id}         get dashboard with layout_json
POST   /api/dashboards              create { name, layout_json }
PUT    /api/dashboards/{id}         update { name?, layout_json? }
DELETE /api/dashboards/{id}         delete
```

**Backend storage:** new SQLite table `dashboards` in the existing `database.py` SQLite file (not DuckDB).

```sql
CREATE TABLE dashboards (
  id          TEXT PRIMARY KEY,          -- UUID v4
  name        TEXT NOT NULL,
  created_at  TEXT NOT NULL,             -- ISO 8601
  updated_at  TEXT NOT NULL,
  layout_json TEXT NOT NULL              -- JSON, see below
);
```

`layout_json` schema:
```json
{
  "version": 1,
  "widgets": [
    {
      "id": "fleet-critical-trend",
      "template_id": "fleet-critical-trend",
      "title": "Fleet Critical Trend",
      "col_span": 6,
      "query_overrides": {
        "time": { "last_n_snapshots": 90 },
        "filters": [{ "field": "severity", "operator": "eq", "value": "Critical" }]
      }
    }
  ]
}
```

`query_overrides` is merged with the template's default Query on render. Fields not in `query_overrides` use template defaults.

No auth / multi-user in v1. Single owner implicit. The default curated dashboard at `/dashboard` is hardcoded in the frontend (not from the database) — it is never editable or deleteable.

---

## 15. PDF export

**Endpoint:** `POST /api/export/pdf`

Request body options:
```json
{ "dashboard_id": "uuid" }
```
or
```json
{ "widget_id": "string", "query_overrides": { ... } }
```

Response: `application/pdf` with `Content-Disposition: attachment; filename="sysdig-analytics-{date}.pdf"`.

**Backend rendering:** `sas/pdf_generator.py`. Reuses existing reportlab pattern from the Streamlit sibling. ECharts SVG rendered server-side (via `pyecharts` or `kaleido` for SVG snapshot from ECharts JSON spec — to be determined at implementation time). No headless browser. No HTML-to-PDF conversion.

**Sysdig branding in PDF:**
- Page 1: Full-bleed Deep See (`#01353E`) cover with white Sysdig logo, report title, and date.
- All pages: header — Deep See banner with white "Analytics Studio" wordmark. Footer — left: "Sysdig Inc. Proprietary", center: `sysdig` bold, right: page number. Both separated from body by a 1px `--grey-20` line.
- Diagonal watermark: "DRAFT" at 3% opacity in Grey-60, rotated 45°, centered on each page. Configurable via `WATERMARK_TEXT` env var (empty string = no watermark).
- Chart area: reportlab primitives for simple charts (bar, line). For the Image Remediation Story (complex stacked + line), render the ECharts SVG snapshot at 72dpi, embed as vector in the PDF.

**Download surface:** "Export as PDF" in every widget's 3-dot menu. "Export Dashboard" button in the page header action row (right of timestamp). Both trigger the same endpoint with different request bodies.

---

## 16. Cosmetic sign-in

Route `/signin`. Protects all other routes via Next.js middleware (`middleware.ts`).

**Page design:** mimics Sysdig login UI. Deep See background. Centered card (white, 400px wide, `var(--radius)` corners). Card contents: Sysdig logo (white variant on Deep See bg) → black variant on white card. Email input + Password input (shadcn `Input` components with Lumin accent ring on focus). "Sign in" button (Deep See fill, white text, Lumin hover ring). Branding footer: "Sysdig Analytics Studio" in Grey-50.

**Auth logic:**
- Password checked against `process.env.SAS_DEMO_PASSWORD`. If env var is unset, any non-empty password succeeds (bypass mode for dev).
- On success: sets httpOnly cookie `sas_session` containing a JWT (`jsonwebtoken` npm package). JWT payload: `{ sub: "demo", iat, exp: iat + 86400 }`. Secret: `process.env.SAS_JWT_SECRET` (or `"dev-secret"` if unset).
- All non-`/signin` routes protected by `middleware.ts`: check for `sas_session` cookie, verify JWT, redirect to `/signin` if absent or invalid.
- Sign out: link in sidebar user area. Calls `GET /api/auth/signout` (a simple route handler that clears the cookie) then redirects to `/signin`.

No real Sysdig OIDC integration in Phase 3. Swappable in Phase 5+ — the middleware.ts auth check is the only touch point.

---

## 17. Typed API client

TypeScript types generated from FastAPI's `/openapi.json` at build time.

**Tool:** `openapi-ts` (formerly `openapi-typescript-codegen`) — free, well-maintained, generates clean TypeScript interfaces.

**Output location:** `sas/web/lib/api/` — generated types and fetch wrappers. This directory is git-ignored; regenerate on any backend change.

**npm script:** `"generate-api": "openapi-ts --input http://localhost:8000/openapi.json --output sas/web/lib/api"`.

**Usage pattern in components:**
```typescript
import { QueryRequest, QueryResult } from "@/lib/api/models";
import { ApiClient } from "@/lib/api/ApiClient";

const result: QueryResult = await ApiClient.postQuery(query);
```

End-to-end type safety: `Query` constructed in the Widget Builder form has the same TypeScript type as the `QueryRequest` schema from the backend. Type errors at build time, not runtime.

**Regeneration protocol:** after any change to FastAPI routes or dataclass shapes, run `npm run generate-api`. CI pipeline should also run this and fail if the generated output differs from committed types (optional — defer to Phase 4 CI work).

---

## 18. Phased delivery within Phase 3

Each sub-phase ends with something demonstrable. Budget estimates are maximums — stop and evaluate before starting next sub-phase.

### Phase 3.1 — Foundation (~$30)

Sonnet task. Output must be reviewable by Aaron before 3.2 starts.

- Next.js 14 App Router scaffold in `sas/web/` (TypeScript, strict mode)
- `tokens.css` + `tailwind.config.ts` (all tokens from §6 above, light + dark)
- App shell: sidebar (180px, Deep See, nav items, My Dashboards stub, user info) + page header (48px, as-of timestamp) + content area
- Cosmetic sign-in at `/signin` with middleware protection
- Typed API client generation script (`npm run generate-api`) wired to Phase 2 FastAPI
- One starter widget end-to-end: **Widget 2 — Fleet Critical Trend** rendering real Phase 2 API data. Proves the full vertical stack: `tokens.css` → card shell → ECharts → `POST /api/query` → live data.
- Dark mode toggle functional

Deliverable gate: Aaron can sign in, see the sidebar and header, and see Fleet Critical Trend rendering live data.

### Phase 3.2 — Widgets (~$25)

Sonnet tasks. Can parallelize across 2 workers: Worker A = thin widgets, Worker B = tier-1 widgets.

- **Worker A:** Widgets 3, 6, 8, 9, 10 (remaining thin widgets). All use card shell from 3.1. Findings Table and Image Inventory use TanStack Table.
- **Worker B:** Widgets 1, 4, 5, 7 (tier-1 custom layouts). Start with Widget 1 (flagship). See §9 for layout specifications.
- Axis-labels toggle implemented on Widgets 1, 2, 3, 6. Cadence logic from §10 (axis-labels toggle principle).
- Dashboard layout: 12-column CSS grid, widget span assignments, gap from token.

Deliverable gate: all 10 widgets render live data on `/dashboard`. Image Remediation Story is the demo centrepiece.

### Phase 3.3 — Drill-in and routing (~$15)

Opus-supervised (routing subtlety). Sonnet implements with close review.

- Nested URL structure from §5 fully wired
- Page-stack animation: framer-motion `AnimatePresence` + CSS perspective transforms (or fallback to plain routing per §11 fallback protocol)
- Breadcrumb strip component: appears/disappears on drill-in/unwind
- Zustand drill stack store
- Detail pages: Image Detail, CVE Detail, Team Detail, Workload Detail (each a focused single-entity view consuming `GET /api/entities/{lens}/{id}` and relevant widgets)

Deliverable gate: clicking an image row in Widget 8 navigates to a populated image detail page. Browser back returns to dashboard. URL is shareable.

### Phase 3.4 — Graph Explorer (2D) (~$15)

Sonnet task. Clear scope, limited integration surface.

- react-flow canvas in `/explore/graph`
- Entity node shapes and colors from §12 table
- `dagre` auto-layout
- Inspector panel (220px right panel)
- Click-to-recenter with `router.push`
- Navigation trail chips at bottom of canvas
- Typeahead entity search to seed initial graph
- 150-node truncation guard

Deliverable gate: search for a CVE, see its blast radius as a graph, click an image node, see inspector panel populate, click a workload node, see graph recenter.

### Phase 3.5 — Save, clone, export (~$20)

Sonnet tasks. Can parallelize: Worker A = save/clone, Worker B = PDF export.

- **Worker A:** Dashboard save/list/open (§14 SQLite schema + API endpoints + sidebar list + save button in page header). Widget Builder "lite" drawer (§13).
- **Worker B:** PDF export endpoint in FastAPI (§15) + download button wiring in frontend. Cover page + branding + at minimum Widget 2 renders correctly in PDF.

Deliverable gate: clone Fleet Critical Trend with a 30-day window, save to a new dashboard, navigate to it via sidebar, export the dashboard as a PDF with Sysdig branding.

---

## 19. What is deliberately out of Phase 3

| Item | Deferred to |
|---|---|
| Full from-scratch Widget Builder (lens / measure / group-by / chart type) | Phase 4 |
| Widget chart-type switcher (bar ↔ line ↔ area) | Phase 4 |
| 3D Graph Explorer | Phase 3.2 upgrade or Phase 4 — validate 2D first |
| Dashboard drag-to-resize / reorder widgets | Phase 4 |
| Real authentication (Sysdig OIDC) | Phase 5 |
| Admin pages (ownership rules editor, ingest history viewer) | Phase 4 |
| Multi-tenant / multi-user | Phase 5+ |
| Additional ECharts chart types (heatmap, scatter, timeline) | Future |
| Alerting / notifications | Out of product scope |
| CI pipeline + automated tests | Phase 4 |
| Gap-coverage strip under every X-axis | Phase 3.2 (included in widget build) |
| Stale-scan amber markers in image views | Phase 3.2 (included in Widget 8) |
| "View as table" accessibility toggle | Phase 4 |

---

## 20. Collaboration pattern

- **Opus** — scaffolding judgment, drill-in/routing integration (§10 stack), Graph Explorer edge cases (§11 recenter + trail), PDF branding QA, final review gate between each sub-phase.
- **Sonnet workers** — all implementation tasks within sub-phases. 80% of Phase 3 token spend. Each worker briefed with a focused task spec referencing sections of this document.
- **Sub-phase review gates** — Aaron reviews deliverable before next sub-phase starts. Opus does a code review pass at each gate. Sonnet does not proceed past a gate autonomously.
- **User manages commits manually** — per project convention. Do not auto-commit.
- **Total budget:** ~$85–110. $500 ceiling approved; ~$390 remaining for Phase 4+ and iteration rounds.

---

## 21. References

- Main design doc: `docs/superpowers/specs/2026-04-23-sas-design.md` (§2 Tenets, §3 Architecture, §4 Data model, §7 Query primitive, §8 Widget catalog, §9 Frontend UX, §9.1 Design tokens, §10 Phased delivery)
- Visual mockups (approved):
  - App shell: `.superpowers/brainstorm/59240-1777030047/content/layout-b-refined.html`
  - Flagship widget: `.superpowers/brainstorm/59240-1777030047/content/widget-remediation-story-v5.html`
  - Drill-in pattern: `.superpowers/brainstorm/59240-1777030047/content/page-stack-drillin.html`
  - Graph Explorer: `.superpowers/brainstorm/59240-1777030047/content/graph-explorer.html`
- Brand tokens: `sas/sysdig-brand.css`, `sas/branding.md`
- Phase 2 API contract: `sas/api/` + live OpenAPI at `/docs` (FastAPI auto-generated)
- Customer context: `docs/problem.md`
