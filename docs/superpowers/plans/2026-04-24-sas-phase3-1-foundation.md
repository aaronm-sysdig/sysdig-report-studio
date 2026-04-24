# SAS Phase 3.1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the Next.js 14 frontend for Sysdig Analytics Studio, wire Sysdig brand tokens, build the app shell (sidebar + page header), implement cosmetic sign-in with middleware auth protection, generate a typed API client from the live Phase 2 OpenAPI spec, and render one real widget — Fleet Critical Trend — end-to-end against live Phase 2 API data. Phase 3.1 is the "is the plumbing right?" phase.

**Architecture:** Next.js 14 App Router at `sas/web/`. TypeScript strict mode throughout. Tailwind CSS driven entirely by design tokens. shadcn/ui for UI primitives. Apache ECharts (via echarts-for-react) for charting. OpenAPI types generated from `http://localhost:8000/openapi.json` at build time.

**Tech stack:** Next.js 14+, TypeScript, Tailwind CSS, shadcn/ui, echarts-for-react, openapi-typescript, jose (JWT), Node 20+.

**Design references:**
- Spec: [`docs/superpowers/specs/2026-04-24-sas-phase3-frontend-design.md`](../specs/2026-04-24-sas-phase3-frontend-design.md) — §3 Stack, §6 Design tokens, §7 Layout grammar, §8 Widget card shell, §9 Widget 2, §15 Cosmetic sign-in (spec labels it §16), §16 Typed API client (spec labels it §17), §17 Phased delivery §3.1 (spec labels it §18).
- Brand tokens: [`sas/sysdig-brand.css`](../../sas/sysdig-brand.css) and [`sas/branding.md`](../../sas/branding.md).
- Phase 2 API: `sas/api/` + `sas/query/primitives.py`. Backend started via `.venv/bin/python -m sas.api.run` on port 8000.

**Budget estimate:** ~$30.

**Collaboration note:** All 12 tasks can be dispatched to Sonnet 4.6 workers. Tasks 5–7 (auth) benefit from a single worker for consistency. Tasks 1–4 and 8–12 are independently parallelisable after Task 1 completes.

---

## File Structure

Every file created or modified in Phase 3.1. One responsibility per file.

```
sas/
  web/                              new — Phase 3 root
    package.json
    tsconfig.json
    next.config.js
    tailwind.config.ts
    postcss.config.js
    .env.local.example
    middleware.ts                   auth protection
    app/
      layout.tsx                    root layout (fonts, theme attribute)
      page.tsx                      / → redirect to /dashboard
      signin/
        page.tsx                    cosmetic sign-in page
      dashboard/
        page.tsx                    renders AppShell + FleetCriticalTrend
      api/
        signin/
          route.ts                  Next.js API route — sets sas_session cookie
        auth/
          signout/
            route.ts                clears cookie, redirects to /signin
    components/
      app-shell/
        Sidebar.tsx
        PageHeader.tsx
        BreadcrumbStrip.tsx
        AppShell.tsx
      ui/                           shadcn components land here (auto-generated)
      widgets/
        WidgetCard.tsx              shared card shell
        FleetCriticalTrend.tsx      the one end-to-end widget in 3.1
    lib/
      api/
        types.ts                    generated from openapi-typescript (git-ignored)
        client.ts                   typed fetch wrapper
      auth/
        cookies.ts                  httpOnly cookie helpers
    styles/
      tokens.css                    design tokens
      globals.css                   Tailwind base + token import
```

---

## Task 1 — Next.js scaffold

**Files:**
- Create: `sas/web/` directory tree via `create-next-app`
- Create: `.env.local.example`
- Modify: `sas/web/package.json` (add echarts deps, openapi-typescript, jose)

- [ ] **Step 1: Verify Node version**

Run:
```bash
node --version
```
Expected: `v20.x.x` or higher. If lower, install Node 20 LTS via `nvm install 20 && nvm use 20` before continuing.

- [ ] **Step 2: Scaffold the Next.js app**

Run from the repo root `sas/` parent:
```bash
npx create-next-app@latest web \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --src-dir=no \
  --import-alias="@/*"
```
When prompted:
- "Would you like to use Turbopack?" → **No** (Turbopack still experimental for production builds)

Expected: scaffold created at `sas/web/`. Confirm with `ls sas/web/`.

- [ ] **Step 3: Install additional dependencies**

Run from `sas/web/`:
```bash
cd sas/web && npm install \
  echarts \
  echarts-for-react \
  jose \
  && npm install --save-dev \
  openapi-typescript
```
Expected: `added N packages` with no peer-dependency warnings for the listed packages.

- [ ] **Step 4: Add npm scripts to package.json**

Open `sas/web/package.json`. In the `"scripts"` block, add:
```json
"generate-api": "openapi-ts --input http://localhost:8000/openapi.json --output lib/api/types.ts --exportSchemas false"
```

The full `"scripts"` section should now be:
```json
"scripts": {
  "dev": "next dev",
  "build": "next build",
  "start": "next start",
  "lint": "next lint",
  "generate-api": "openapi-ts --input http://localhost:8000/openapi.json --output lib/api/types.ts --exportSchemas false"
}
```

- [ ] **Step 5: Create `.env.local.example`**

File: `sas/web/.env.local.example`
```
# Copy this to .env.local and fill in values before running.

# Shared demo password for the cosmetic sign-in page.
# If unset, any non-empty password is accepted (dev bypass).
SAS_DEMO_PASSWORD=changeme

# Secret used to sign the sas_session JWT.
# If unset, falls back to "dev-secret" (not safe for production).
SAS_JWT_SECRET=change-this-to-a-random-32-char-string

# Base URL for the Phase 2 FastAPI backend.
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

- [ ] **Step 6: Add `sas/web/lib/api/` to `.gitignore`**

Append to `sas/web/.gitignore`:
```
# Generated OpenAPI types — regenerate with npm run generate-api
/lib/api/types.ts
```

- [ ] **Step 7: Verify the empty scaffold runs**

Run:
```bash
cd sas/web && npm run dev
```
Open `http://localhost:3000` in a browser. Expected: the default Next.js "Get Started" page renders with no console errors.

Ctrl+C to stop the dev server.

- [ ] **Step 8: Commit**

```bash
git add sas/web/
git commit -m "feat(sas): phase 3.1 next.js scaffold with typescript and tailwind"
```

---

## Task 2 — Design tokens

**Files:**
- Create: `sas/web/styles/tokens.css`
- Create: `sas/web/styles/globals.css` (replaces scaffolded version)
- Modify: `sas/web/app/layout.tsx`

- [ ] **Step 1: Create the tokens file**

File: `sas/web/styles/tokens.css`

```css
/* ==========================================================
   SAS Design Tokens — single source of truth.
   All components reference these custom properties.
   Raw hex values never appear outside this file.
   ========================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  /* --- Sysdig brand palette (mirrors sas/sysdig-brand.css) --- */
  --white: #FFFFFF;
  --black: #000000;
  --deep-see: #01353E;
  --lumin: #BDF78B;

  --grey-10: #EAEBED;
  --grey-20: #D4D6D9;
  --grey-30: #BEC0C5;
  --grey-40: #A8ABB1;
  --grey-50: #92959D;
  --grey-60: #6E7178;
  --grey-70: #4A4D53;
  --grey-80: #26282E;
  --grey-90: #121217;

  --falco-blue: #00CBE2;
  --red: #FF7774;
  --orange: #FFA940;
  --yellow: #FDD835;
  --purple: #CA87DA;

  /* --- Severity --- */
  --severity-critical: var(--red);
  --severity-high: var(--orange);
  --severity-medium: var(--yellow);
  --severity-low: var(--grey-50);

  /* --- Semantic surface tokens — light mode (default) --- */
  --bg-base: var(--white);
  --bg-surface: var(--grey-10);
  --bg-sidebar: var(--deep-see);
  --bg-sidebar-active: rgba(189, 247, 139, 0.18);   /* 18% Lumin */
  --bg-sidebar-hover: rgba(255, 255, 255, 0.08);     /* 8% white */
  --fg-primary: var(--black);
  --fg-muted: var(--grey-60);
  --fg-on-sidebar: var(--white);
  --fg-on-sidebar-muted: var(--grey-40);
  --border-subtle: var(--grey-20);
  --border-strong: var(--grey-40);
  --accent: var(--lumin);

  /* --- Density --- */
  --h-row: 32px;
  --p-card: 14px;
  --gap-widget: 10px;
  --h-topbar: 48px;
  --h-sidebar-row: 30px;
  --h-breadcrumb: 28px;
  --w-sidebar: 180px;
  --radius: 8px;

  /* --- Elevation --- */
  --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.05);

  /* --- Motion --- */
  --dur-standard: 180ms;
  --ease-standard: cubic-bezier(0.2, 0, 0, 1);

  /* --- Typography --- */
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --text-label: 10px;
  --text-body-sm: 11px;
  --text-body: 13px;
  --text-body-md: 14px;
  --text-title: 16px;
}

/* --- Dark mode overrides (body[data-theme="dark"]) --- */
[data-theme="dark"] {
  --bg-base: var(--grey-80);
  --bg-surface: var(--grey-90);
  --bg-sidebar: var(--grey-90);
  --fg-primary: var(--white);
  --fg-muted: var(--grey-50);
  --border-subtle: var(--grey-70);
  --border-strong: var(--grey-60);
  /* Severity and brand tokens are unchanged in dark mode */
}
```

- [ ] **Step 2: Create globals.css**

File: `sas/web/styles/globals.css`

```css
@import "./tokens.css";

@tailwind base;
@tailwind components;
@tailwind utilities;

*,
*::before,
*::after {
  box-sizing: border-box;
}

html,
body {
  height: 100%;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  font-size: var(--text-body);
  color: var(--fg-primary);
  background-color: var(--bg-base);
  -webkit-font-smoothing: antialiased;
  transition: background-color var(--dur-standard) var(--ease-standard),
              color var(--dur-standard) var(--ease-standard);
}
```

- [ ] **Step 3: Wire tokens into tailwind.config.ts**

Replace the contents of `sas/web/tailwind.config.ts` with:

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class", '[data-theme="dark"]'],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brand palette — reference CSS vars so Tailwind classes stay token-aware
        "deep-see":    "var(--deep-see)",
        "lumin":       "var(--lumin)",
        "falco-blue":  "var(--falco-blue)",
        // Semantic
        "bg-base":     "var(--bg-base)",
        "bg-surface":  "var(--bg-surface)",
        "bg-sidebar":  "var(--bg-sidebar)",
        "fg-primary":  "var(--fg-primary)",
        "fg-muted":    "var(--fg-muted)",
        "border-subtle":"var(--border-subtle)",
        "border-strong":"var(--border-strong)",
        // Severity
        "severity-critical": "var(--severity-critical)",
        "severity-high":     "var(--severity-high)",
        "severity-medium":   "var(--severity-medium)",
        "severity-low":      "var(--severity-low)",
        // Greys
        "grey-10": "var(--grey-10)",
        "grey-20": "var(--grey-20)",
        "grey-40": "var(--grey-40)",
        "grey-50": "var(--grey-50)",
        "grey-60": "var(--grey-60)",
        "grey-70": "var(--grey-70)",
        "grey-80": "var(--grey-80)",
        "grey-90": "var(--grey-90)",
      },
      spacing: {
        "sidebar": "var(--w-sidebar)",
        "topbar":  "var(--h-topbar)",
        "row":     "var(--h-row)",
        "card":    "var(--p-card)",
        "widget-gap": "var(--gap-widget)",
        "breadcrumb": "var(--h-breadcrumb)",
        "sidebar-row": "var(--h-sidebar-row)",
      },
      borderRadius: {
        DEFAULT: "var(--radius)",
        "sas": "var(--radius)",
      },
      boxShadow: {
        "card": "var(--shadow-card)",
      },
      fontFamily: {
        "sans": ["var(--font-sans)"],
      },
      transitionDuration: {
        "standard": "var(--dur-standard)",
      },
    },
  },
  plugins: [],
};

export default config;
```

- [ ] **Step 4: Update root layout to import globals.css and set font**

Replace `sas/web/app/layout.tsx` with:

```typescript
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "@/styles/globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sysdig Analytics Studio",
  description: "Vulnerability analytics and reporting for Sysdig-monitored fleets.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.variable}>
        {children}
      </body>
    </html>
  );
}
```

> Note: `suppressHydrationWarning` is required because the dark-mode `data-theme` attribute is set client-side from `localStorage` on first render, which would otherwise cause a hydration mismatch.

- [ ] **Step 5: Verify build succeeds and body uses Inter**

Run:
```bash
cd sas/web && npm run build
```
Expected: `✓ Compiled successfully` (or equivalent) with no TypeScript errors.

Then run `npm run dev`, open `http://localhost:3000`, open browser DevTools → Elements. Confirm `body` has `font-family` resolving to `Inter`.

- [ ] **Step 6: Commit**

```bash
git add sas/web/styles/ sas/web/tailwind.config.ts sas/web/app/layout.tsx
git commit -m "feat(sas): design tokens and tailwind config wired to sysdig brand"
```

---

## Task 3 — shadcn/ui installation

**Files:**
- Create: `sas/web/components/ui/` (auto-generated by shadcn CLI)
- Modify: `sas/web/components.json` (shadcn config, auto-created)

- [ ] **Step 1: Initialise shadcn/ui**

Run from `sas/web/`:
```bash
npx shadcn-ui@latest init
```

When prompted, answer:
- "Which style would you like to use?" → **Default**
- "Which color would you like to use as base color?" → **Slate** (closest to Deep See; we override with tokens anyway)
- "Would you like to use CSS variables for colors?" → **Yes**
- "Where is your global CSS file?" → `styles/globals.css`
- "Are you using a custom tailwind prefix?" → **No**
- "Where is your tailwind.config located?" → `tailwind.config.ts`
- "Configure the import alias for components?" → `@/components`
- "Configure the import alias for utils?" → `@/lib/utils`

Expected: `components.json` created; `lib/utils.ts` created; no errors.

- [ ] **Step 2: Install required shadcn components**

Run from `sas/web/`:
```bash
npx shadcn-ui@latest add button card dialog input label dropdown-menu
```
Expected: files appear in `sas/web/components/ui/`. Each is a `.tsx` file.

- [ ] **Step 3: Override shadcn CSS variables to use our tokens**

shadcn generates `--background`, `--foreground`, etc. in `globals.css`. Replace those generated variables with mappings to our tokens. In `sas/web/styles/globals.css`, after the `@tailwind base` block, add:

```css
@layer base {
  :root {
    /* Map shadcn's expected variables to our tokens */
    --background:       255 255 255;   /* white */
    --foreground:       0 0 0;         /* black */
    --card:             255 255 255;
    --card-foreground:  0 0 0;
    --border:           212 214 217;   /* grey-20 */
    --input:            212 214 217;
    --ring:             189 247 139;   /* lumin */
    --radius:           8px;
    --primary:          1 53 62;       /* deep-see */
    --primary-foreground: 255 255 255;
    --muted:            234 235 237;   /* grey-10 */
    --muted-foreground: 110 113 120;   /* grey-60 */
  }

  [data-theme="dark"] {
    --background:       38 40 46;      /* grey-80 */
    --foreground:       255 255 255;
    --card:             18 18 23;      /* grey-90 */
    --card-foreground:  255 255 255;
    --border:           74 77 83;      /* grey-70 */
    --input:            74 77 83;
    --muted:            18 18 23;
    --muted-foreground: 146 149 157;   /* grey-50 */
  }
}
```

- [ ] **Step 4: Verify a Button renders with Sysdig colours**

Create a temporary test page `sas/web/app/test/page.tsx`:
```typescript
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function TestPage() {
  return (
    <div className="p-8 bg-bg-base min-h-screen">
      <Card className="max-w-sm shadow-card">
        <CardHeader>
          <CardTitle className="text-fg-primary">Sysdig Analytics Studio</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button className="bg-deep-see text-white hover:opacity-90">Sign in</Button>
          <Button variant="outline">Cancel</Button>
        </CardContent>
      </Card>
    </div>
  );
}
```

Run `npm run dev`, open `http://localhost:3000/test`. Confirm:
- Card renders with white background and subtle shadow.
- "Sign in" button is Deep See (`#01353E`) with white text.

Delete `sas/web/app/test/` after verification.

- [ ] **Step 5: Commit**

```bash
git add sas/web/components/ui/ sas/web/components.json sas/web/lib/utils.ts sas/web/styles/globals.css
git commit -m "feat(sas): shadcn/ui initialised and mapped to sysdig brand tokens"
```

---

## Task 4 — Typed API client generation

**Files:**
- Create: `sas/web/lib/api/client.ts`
- Create (generated): `sas/web/lib/api/types.ts` (git-ignored; generated at runtime)

- [ ] **Step 1: Ensure the Phase 2 backend is running**

Run (in a separate terminal):
```bash
.venv/bin/python -m sas.api.run
```
Expected: `Uvicorn running on http://0.0.0.0:8000`. Verify with:
```bash
curl -s http://localhost:8000/healthz | python3 -m json.tool
```
Expected output contains `"status": "ok"`.

- [ ] **Step 2: Generate TypeScript types from the OpenAPI spec**

Run from `sas/web/`:
```bash
npm run generate-api
```
Expected: `lib/api/types.ts` created. Verify it contains interfaces including `QueryIn`, `QueryResult`, `FilterIn`, `TimeWindowIn`, `OrderingIn`.

Spot-check the generated file:
```bash
grep -E "^export (interface|type) (QueryIn|QueryResult|FilterIn|TimeWindowIn)" sas/web/lib/api/types.ts
```
Expected: four matching lines.

- [ ] **Step 3: Create the typed API client wrapper**

File: `sas/web/lib/api/client.ts`

```typescript
/**
 * Typed fetch wrapper for the Phase 2 FastAPI backend.
 * All methods are async and throw on non-2xx responses.
 * Import types from ./types (generated — run `npm run generate-api` if missing).
 */

// Types generated from openapi.json — see .gitignore note
// If this file is missing, run: npm run generate-api
import type { QueryIn, components } from "./types";

export type QueryResult = components["schemas"]["QueryResult"] extends undefined
  ? {
      series: Array<{ key: Record<string, unknown>; x: string[]; y: number[] }>;
      dimensions: Record<string, unknown[]>;
      snapshot_range: [string, string];
      missing_days: string[];
      exec_time_ms: number;
    }
  : components["schemas"]["QueryResult"];

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function apiFetch<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status} on ${path}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API surface
// ---------------------------------------------------------------------------

/**
 * POST /api/query — execute a structured query and return time-series results.
 */
export async function runQuery(query: QueryIn): Promise<QueryResult> {
  return apiFetch<QueryResult>("/api/query", {
    method: "POST",
    body: JSON.stringify(query),
  });
}

/**
 * GET /api/widgets/catalog — list all registered widget templates.
 */
export async function getWidgetsCatalog(): Promise<unknown[]> {
  return apiFetch<unknown[]>("/api/widgets/catalog");
}

/**
 * GET /api/entities/{lens} — list entity values for a given lens.
 * Used for typeahead pickers (image, cve, team, repository, etc.).
 */
export async function getEntities(
  lens: string,
  params?: Record<string, string>
): Promise<unknown[]> {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  return apiFetch<unknown[]>(`/api/entities/${lens}${qs}`);
}
```

> **Note on generated types:** `openapi-typescript` emits a `components` namespace. If the FastAPI schema names `QueryIn` differently (e.g. `Body_run_query`), update the import alias. Always re-run `npm run generate-api` after any backend change.

- [ ] **Step 4: Verify TypeScript compiles without errors**

Run from `sas/web/`:
```bash
npx tsc --noEmit
```
Expected: no output (zero errors). If `types.ts` import errors appear, re-run `npm run generate-api` first.

- [ ] **Step 5: Commit**

```bash
git add sas/web/lib/api/client.ts sas/web/.gitignore sas/web/package.json
git commit -m "feat(sas): typed api client with openapi-typescript generation script"
```

---

## Task 5 — Cosmetic sign-in: backend cookie route

**Files:**
- Create: `sas/web/app/api/signin/route.ts`
- Create: `sas/web/app/api/auth/signout/route.ts`
- Create: `sas/web/lib/auth/cookies.ts`

- [ ] **Step 1: Create the cookie helper**

File: `sas/web/lib/auth/cookies.ts`

```typescript
/**
 * Cookie name and helper constants for SAS session auth.
 * The session is cosmetic — not production-grade security.
 */

export const SESSION_COOKIE = "sas_session";

export const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  path: "/",
  maxAge: 60 * 60 * 24, // 24 hours in seconds
};
```

- [ ] **Step 2: Create the sign-in API route**

File: `sas/web/app/api/signin/route.ts`

```typescript
import { NextRequest, NextResponse } from "next/server";
import { SignJWT } from "jose";
import { SESSION_COOKIE, COOKIE_OPTIONS } from "@/lib/auth/cookies";

const DEMO_PASSWORD = process.env.SAS_DEMO_PASSWORD;
const JWT_SECRET = new TextEncoder().encode(
  process.env.SAS_JWT_SECRET ?? "dev-secret"
);

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: { username?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid request body." },
      { status: 400 }
    );
  }

  const { password } = body;

  if (!password) {
    return NextResponse.json(
      { error: "Password is required." },
      { status: 400 }
    );
  }

  // If SAS_DEMO_PASSWORD is unset, any non-empty password succeeds (dev bypass).
  const isValid =
    DEMO_PASSWORD === undefined || DEMO_PASSWORD === ""
      ? password.length > 0
      : password === DEMO_PASSWORD;

  if (!isValid) {
    return NextResponse.json(
      { error: "Invalid credentials. Please try again." },
      { status: 401 }
    );
  }

  const now = Math.floor(Date.now() / 1000);
  const token = await new SignJWT({ sub: "demo" })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt(now)
    .setExpirationTime(now + 86400) // 24 hours
    .sign(JWT_SECRET);

  const response = NextResponse.json({ ok: true }, { status: 200 });
  response.cookies.set(SESSION_COOKIE, token, COOKIE_OPTIONS);
  return response;
}
```

- [ ] **Step 3: Create the sign-out route**

File: `sas/web/app/api/auth/signout/route.ts`

```typescript
import { NextResponse } from "next/server";
import { SESSION_COOKIE } from "@/lib/auth/cookies";

export async function GET(): Promise<NextResponse> {
  const response = NextResponse.redirect(
    new URL("/signin", process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000")
  );
  response.cookies.set(SESSION_COOKIE, "", {
    httpOnly: true,
    maxAge: 0,
    path: "/",
  });
  return response;
}
```

- [ ] **Step 4: Verify the sign-in route returns a cookie**

With `npm run dev` running, run:
```bash
curl -s -X POST http://localhost:3000/api/signin \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"anything"}' \
  -v 2>&1 | grep -E "(Set-Cookie|HTTP/)"
```
Expected output includes:
```
< HTTP/1.1 200 OK
< Set-Cookie: sas_session=eyJ...
```

- [ ] **Step 5: Verify wrong password returns 401**

```bash
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:3000/api/signin \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":""}' \
  -e ""
```
Expected: `400` (empty password).

If `SAS_DEMO_PASSWORD=secret` is set in `.env.local` and password is `wrong`:
```bash
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:3000/api/signin \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"wrong"}'
```
Expected: `401`.

- [ ] **Step 6: Commit**

```bash
git add sas/web/app/api/ sas/web/lib/auth/
git commit -m "feat(sas): cosmetic sign-in api route with jose jwt and httponly cookie"
```

---

## Task 6 — Sign-in page UI

**Files:**
- Create: `sas/web/app/signin/page.tsx`

- [ ] **Step 1: Create the sign-in page**

File: `sas/web/app/signin/page.tsx`

```typescript
"use client";

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function SignInPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const res = await fetch("/api/signin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (res.ok) {
        router.push("/dashboard");
      } else {
        const data = await res.json();
        setError(data.error ?? "Invalid credentials. Please try again.");
      }
    } catch {
      setError("Unable to connect. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ backgroundColor: "var(--deep-see)" }}
    >
      <div
        className="w-[400px] rounded-sas p-8 shadow-xl"
        style={{ backgroundColor: "var(--white)" }}
      >
        {/* Wordmark */}
        <div className="flex items-center gap-2 mb-6">
          {/* Sysdig wordmark — text fallback matching brand spec */}
          <span
            className="text-[22px] font-bold tracking-tight"
            style={{ color: "var(--deep-see)" }}
          >
            sysdig
          </span>
          {/* Lumin dot — brand accent */}
          <span
            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: "var(--lumin)" }}
            aria-hidden="true"
          />
        </div>

        {/* Heading with Lumin left-edge accent */}
        <div className="flex items-center gap-3 mb-6">
          <div
            className="w-1 h-6 rounded-full flex-shrink-0"
            style={{ backgroundColor: "var(--lumin)" }}
            aria-hidden="true"
          />
          <h1
            className="text-xl font-semibold"
            style={{ color: "var(--fg-primary)" }}
          >
            Sign in
          </h1>
        </div>

        <form onSubmit={handleSubmit} noValidate className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="username" style={{ color: "var(--fg-primary)" }}>
              Username
            </Label>
            <Input
              id="username"
              type="text"
              autoComplete="username"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              style={{
                borderColor: "var(--border-subtle)",
                "--tw-ring-color": "var(--lumin)",
              } as React.CSSProperties}
              className="focus-visible:ring-2"
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="password" style={{ color: "var(--fg-primary)" }}>
              Password
            </Label>
            <Input
              id="password"
              type="password"
              autoComplete="current-password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{
                borderColor: "var(--border-subtle)",
                "--tw-ring-color": "var(--lumin)",
              } as React.CSSProperties}
              className="focus-visible:ring-2"
            />
          </div>

          {error && (
            <p
              className="text-sm"
              role="alert"
              style={{ color: "var(--severity-critical)" }}
            >
              {error}
            </p>
          )}

          <Button
            type="submit"
            disabled={loading}
            className="w-full font-medium"
            style={{
              backgroundColor: "var(--deep-see)",
              color: "var(--white)",
            }}
          >
            {loading ? "Signing in…" : "Sign in"}
          </Button>
        </form>

        <p
          className="mt-6 text-center text-xs"
          style={{ color: "var(--fg-muted)" }}
        >
          Sysdig Analytics Studio
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create root redirect page**

Replace `sas/web/app/page.tsx` with:

```typescript
import { redirect } from "next/navigation";

export default function RootPage() {
  redirect("/dashboard");
}
```

- [ ] **Step 3: Visual verification**

Run `npm run dev`, navigate to `http://localhost:3000/signin`. Confirm:
- Deep See (`#01353E`) full-bleed background.
- White card centred on screen, 400px wide.
- "sysdig" wordmark with Lumin dot to the right.
- "Sign in" heading with a Lumin left-edge accent bar.
- Username and Password inputs.
- Sign in button (Deep See background, white text).
- Typing an incorrect password shows the error message.
- Correct password (or any password if `SAS_DEMO_PASSWORD` unset) redirects to `/dashboard` (which 404s at this point — expected).

- [ ] **Step 4: Commit**

```bash
git add sas/web/app/signin/ sas/web/app/page.tsx
git commit -m "feat(sas): cosmetic sign-in page with sysdig brand styling"
```

---

## Task 7 — Middleware for auth protection

**Files:**
- Create: `sas/web/middleware.ts`

- [ ] **Step 1: Create middleware.ts**

File: `sas/web/middleware.ts`

```typescript
import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { SESSION_COOKIE } from "@/lib/auth/cookies";

const JWT_SECRET = new TextEncoder().encode(
  process.env.SAS_JWT_SECRET ?? "dev-secret"
);

// Paths that do not require authentication.
const PUBLIC_PATHS = [
  "/signin",
  "/api/signin",
  "/api/auth/signout",
];

function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/"))
    || pathname.startsWith("/_next/")
    || pathname.startsWith("/favicon")
    || pathname.startsWith("/static/");
}

export async function middleware(req: NextRequest): Promise<NextResponse> {
  const { pathname } = req.nextUrl;

  if (isPublicPath(pathname)) {
    return NextResponse.next();
  }

  const token = req.cookies.get(SESSION_COOKIE)?.value;

  if (!token) {
    const signInUrl = req.nextUrl.clone();
    signInUrl.pathname = "/signin";
    return NextResponse.redirect(signInUrl);
  }

  try {
    await jwtVerify(token, JWT_SECRET);
    return NextResponse.next();
  } catch {
    // Token invalid or expired — clear the cookie and redirect to sign-in.
    const signInUrl = req.nextUrl.clone();
    signInUrl.pathname = "/signin";
    const response = NextResponse.redirect(signInUrl);
    response.cookies.set(SESSION_COOKIE, "", { maxAge: 0, path: "/" });
    return response;
  }
}

export const config = {
  // Run middleware on all paths except Next.js internals and static assets.
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
```

- [ ] **Step 2: Verify unauthenticated access redirects to sign-in**

With `npm run dev` running:
```bash
curl -s -o /dev/null -w "%{http_code}" \
  -L http://localhost:3000/dashboard
```
Expected: `200` (after following redirect to `/signin`). The final URL should be `/signin`.

To confirm redirect happens (not a direct serve):
```bash
curl -s -o /dev/null -w "%{redirect_url}" \
  http://localhost:3000/dashboard
```
Expected: `http://localhost:3000/signin`.

- [ ] **Step 3: Verify authenticated access passes through**

```bash
# 1. Get a session cookie
TOKEN=$(curl -s -X POST http://localhost:3000/api/signin \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"anything"}' \
  -c /tmp/sas-cookies.txt -b /tmp/sas-cookies.txt)

# 2. Access a protected route with the cookie
curl -s -o /dev/null -w "%{http_code}" \
  -b /tmp/sas-cookies.txt \
  http://localhost:3000/dashboard
```
Expected: `200` (not a redirect — the page renders, even if it 404s due to missing dashboard page).

- [ ] **Step 4: Commit**

```bash
git add sas/web/middleware.ts
git commit -m "feat(sas): next.js middleware for jwt-based route protection"
```

---

## Task 8 — App shell: Sidebar

**Files:**
- Create: `sas/web/components/app-shell/Sidebar.tsx`

- [ ] **Step 1: Create Sidebar.tsx**

File: `sas/web/components/app-shell/Sidebar.tsx`

```typescript
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Simple inline SVG icons — no external icon dep needed for 3.1
function IconDashboard({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
      <rect x="1" y="1" width="6" height="6" rx="1" />
      <rect x="9" y="1" width="6" height="6" rx="1" />
      <rect x="1" y="9" width="6" height="6" rx="1" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
    </svg>
  );
}

function IconExplore({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="7" cy="7" r="5" />
      <path d="M11 11l3 3" strokeLinecap="round" />
    </svg>
  );
}

function IconAdmin({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="5" r="2.5" />
      <path d="M2 13c0-3.3 2.7-6 6-6s6 2.7 6 6" strokeLinecap="round" />
    </svg>
  );
}

function IconSignOut({ className }: { className?: string }) {
  return (
    <svg className={className} width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M6 3H3a1 1 0 00-1 1v8a1 1 0 001 1h3" strokeLinecap="round" />
      <path d="M10 11l3-3-3-3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M13 8H6" strokeLinecap="round" />
    </svg>
  );
}

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", Icon: IconDashboard },
  { href: "/explore", label: "Explore", Icon: IconExplore },
  { href: "/admin", label: "Admin", Icon: IconAdmin },
];

export function Sidebar() {
  const pathname = usePathname();

  function isActive(href: string) {
    if (href === "/dashboard") return pathname === "/dashboard" || pathname.startsWith("/dashboard/");
    return pathname.startsWith(href);
  }

  return (
    <aside
      className="flex flex-col h-full flex-shrink-0"
      style={{
        width: "var(--w-sidebar)",
        backgroundColor: "var(--bg-sidebar)",
      }}
    >
      {/* Wordmark */}
      <div className="flex items-center gap-2 px-4 py-4 flex-shrink-0">
        <span
          className="text-lg font-bold tracking-tight"
          style={{ color: "var(--white)" }}
        >
          sysdig
        </span>
        <span
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: "var(--lumin)" }}
          aria-hidden="true"
        />
      </div>

      {/* Product label */}
      <div
        className="px-4 pb-3 text-[10px] font-medium tracking-widest uppercase"
        style={{ color: "var(--fg-on-sidebar-muted)" }}
      >
        Analytics Studio
      </div>

      {/* Nav items */}
      <nav className="flex-1 px-2 space-y-0.5 overflow-y-auto" aria-label="Main navigation">
        {NAV_ITEMS.map(({ href, label, Icon }) => {
          const active = isActive(href);
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-2.5 px-3 rounded-sas text-sm font-medium transition-colors duration-standard"
              style={{
                height: "var(--h-sidebar-row)",
                color: active ? "var(--white)" : "rgba(255,255,255,0.75)",
                backgroundColor: active
                  ? "var(--bg-sidebar-active)"
                  : "transparent",
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.backgroundColor =
                    "var(--bg-sidebar-hover)";
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.backgroundColor =
                    "transparent";
                }
              }}
              aria-current={active ? "page" : undefined}
            >
              <Icon className="flex-shrink-0 opacity-80" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* My Dashboards — stub for 3.1 */}
      <div className="px-2 pt-2 pb-1">
        <div
          className="px-3 py-1 text-[10px] font-medium tracking-widest uppercase"
          style={{ color: "var(--fg-on-sidebar-muted)" }}
        >
          My Dashboards
        </div>
        <p
          className="px-3 py-1 text-[11px] italic"
          style={{ color: "rgba(255,255,255,0.35)" }}
        >
          No saved dashboards yet.
        </p>
      </div>

      {/* User info — pinned at bottom */}
      <div
        className="flex items-center justify-between px-3 py-3 mt-auto border-t"
        style={{ borderColor: "rgba(255,255,255,0.1)" }}
      >
        <div className="flex items-center gap-2 min-w-0">
          {/* Avatar */}
          <div
            className="w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-bold"
            style={{
              backgroundColor: "var(--lumin)",
              color: "var(--deep-see)",
            }}
            aria-hidden="true"
          >
            D
          </div>
          <span
            className="text-[11px] truncate"
            style={{ color: "rgba(255,255,255,0.7)" }}
          >
            demo
          </span>
        </div>
        <a
          href="/api/auth/signout"
          className="flex-shrink-0 opacity-60 hover:opacity-100 transition-opacity"
          style={{ color: "var(--white)" }}
          title="Sign out"
          aria-label="Sign out"
        >
          <IconSignOut />
        </a>
      </div>
    </aside>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add sas/web/components/app-shell/Sidebar.tsx
git commit -m "feat(sas): sidebar component with deep see background and nav items"
```

---

## Task 9 — App shell: PageHeader, BreadcrumbStrip, and AppShell wrapper

**Files:**
- Create: `sas/web/components/app-shell/PageHeader.tsx`
- Create: `sas/web/components/app-shell/BreadcrumbStrip.tsx`
- Create: `sas/web/components/app-shell/AppShell.tsx`

- [ ] **Step 1: Create PageHeader.tsx**

File: `sas/web/components/app-shell/PageHeader.tsx`

```typescript
"use client";

import { useState, useEffect } from "react";

interface PageHeaderProps {
  title: string;
  asOf?: string;        // ISO date string; defaults to now if omitted
  children?: React.ReactNode; // optional action buttons (right slot)
}

function formatAsOf(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

function DarkModeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("sas-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = stored ? stored === "dark" : prefersDark;
    setDark(isDark);
    document.body.setAttribute("data-theme", isDark ? "dark" : "light");
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.body.setAttribute("data-theme", next ? "dark" : "light");
    localStorage.setItem("sas-theme", next ? "dark" : "light");
  }

  return (
    <button
      onClick={toggle}
      className="p-1 rounded opacity-60 hover:opacity-100 transition-opacity"
      style={{ color: "var(--fg-muted)" }}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? (
        // Sun icon
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="8" cy="8" r="3" />
          <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.1 3.1l1.4 1.4M11.5 11.5l1.4 1.4M11.5 3.1l-1.4 1.4M3.1 11.5l1.4 1.4" strokeLinecap="round" />
        </svg>
      ) : (
        // Moon icon
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M13.6 11A6 6 0 015 2.4a6 6 0 100 11.2 6 6 0 008.6-2.6z" />
        </svg>
      )}
    </button>
  );
}

export function PageHeader({ title, asOf, children }: PageHeaderProps) {
  const timestamp = asOf ?? new Date().toISOString();

  return (
    <header
      className="flex items-center justify-between flex-shrink-0 px-5 border-b"
      style={{
        height: "var(--h-topbar)",
        backgroundColor: "var(--bg-base)",
        borderColor: "var(--border-subtle)",
      }}
    >
      <span
        className="text-sm font-medium"
        style={{ color: "var(--fg-primary)" }}
      >
        {title}
      </span>

      <div className="flex items-center gap-3">
        <span
          className="text-xs"
          style={{ color: "var(--fg-muted)" }}
        >
          As of {formatAsOf(timestamp)}
        </span>
        <DarkModeToggle />
        {children}
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Create BreadcrumbStrip.tsx**

File: `sas/web/components/app-shell/BreadcrumbStrip.tsx`

```typescript
// BreadcrumbStrip — rendered only when drill depth > 0.
// In Phase 3.1 this is always empty (no drill-in yet).
// Phase 3.3 will populate crumbs from the Zustand drill stack.

interface Crumb {
  label: string;
  href: string;
}

interface BreadcrumbStripProps {
  crumbs?: Crumb[];
}

export function BreadcrumbStrip({ crumbs = [] }: BreadcrumbStripProps) {
  if (crumbs.length === 0) return null;

  return (
    <nav
      className="flex items-center px-5 gap-1.5 flex-shrink-0 border-b"
      style={{
        height: "var(--h-breadcrumb)",
        backgroundColor: "var(--bg-base)",
        borderColor: "var(--border-subtle)",
      }}
      aria-label="Breadcrumb"
    >
      {crumbs.map((crumb, i) => (
        <span key={crumb.href} className="flex items-center gap-1.5">
          {i > 0 && (
            <span
              className="text-xs"
              style={{ color: "var(--fg-muted)" }}
              aria-hidden="true"
            >
              /
            </span>
          )}
          <a
            href={crumb.href}
            className="text-xs hover:underline"
            style={{ color: i === crumbs.length - 1 ? "var(--fg-primary)" : "var(--fg-muted)" }}
          >
            {crumb.label}
          </a>
        </span>
      ))}
    </nav>
  );
}
```

- [ ] **Step 3: Create AppShell.tsx**

File: `sas/web/components/app-shell/AppShell.tsx`

```typescript
import { Sidebar } from "./Sidebar";
import { PageHeader } from "./PageHeader";
import { BreadcrumbStrip } from "./BreadcrumbStrip";

interface Crumb {
  label: string;
  href: string;
}

interface AppShellProps {
  pageTitle: string;
  asOf?: string;
  crumbs?: Crumb[];
  headerActions?: React.ReactNode;
  children: React.ReactNode;
}

export function AppShell({
  pageTitle,
  asOf,
  crumbs,
  headerActions,
  children,
}: AppShellProps) {
  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ backgroundColor: "var(--bg-base)" }}
    >
      {/* Left sidebar — fixed width */}
      <Sidebar />

      {/* Main content area */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <PageHeader title={pageTitle} asOf={asOf}>
          {headerActions}
        </PageHeader>
        <BreadcrumbStrip crumbs={crumbs} />
        <main
          className="flex-1 overflow-auto p-5"
          style={{ backgroundColor: "var(--bg-base)" }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add sas/web/components/app-shell/
git commit -m "feat(sas): app shell components — sidebar, page header, breadcrumb, wrapper"
```

---

## Task 10 — Widget card shell

**Files:**
- Create: `sas/web/components/widgets/WidgetCard.tsx`

- [ ] **Step 1: Create WidgetCard.tsx**

File: `sas/web/components/widgets/WidgetCard.tsx`

```typescript
"use client";

import { useState, useRef, useEffect } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";

interface WidgetCardProps {
  /** 10px uppercase category label, e.g. "FLEET METRICS" */
  label: string;
  /** 13px card title, e.g. "Fleet Critical Trend" */
  title: string;
  /** Optional footer narrative text. Truncated to 1 line with expand link. */
  footer?: string;
  /** Optional axis-labels toggle state — pass undefined to hide the toggle. */
  axisLabels?: boolean;
  onAxisLabelsChange?: (on: boolean) => void;
  children: React.ReactNode;
}

function ThreeDotIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
      <circle cx="8" cy="3" r="1.2" />
      <circle cx="8" cy="8" r="1.2" />
      <circle cx="8" cy="13" r="1.2" />
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="2" y="3" width="12" height="11" rx="1.5" />
      <path d="M5 1v4M11 1v4M2 7h12" strokeLinecap="round" />
    </svg>
  );
}

export function WidgetCard({
  label,
  title,
  footer,
  axisLabels,
  onAxisLabelsChange,
  children,
}: WidgetCardProps) {
  const [footerExpanded, setFooterExpanded] = useState(false);
  const [footerOverflows, setFooterOverflows] = useState(false);
  const footerRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    if (footerRef.current) {
      setFooterOverflows(
        footerRef.current.scrollWidth > footerRef.current.clientWidth
      );
    }
  }, [footer]);

  return (
    <div
      className="flex flex-col rounded-sas shadow-card transition-colors duration-standard"
      style={{
        backgroundColor: "var(--bg-base)",
        border: "1px solid var(--border-subtle)",
        padding: "var(--p-card)",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor = "var(--border-strong)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor = "var(--border-subtle)";
      }}
    >
      {/* Label row */}
      <div className="flex items-center justify-between mb-0.5" style={{ height: "24px" }}>
        <span
          className="text-[10px] font-medium tracking-widest uppercase"
          style={{ color: "var(--fg-muted)" }}
        >
          {label}
        </span>

        <div className="flex items-center gap-1">
          {/* Axis-labels toggle — only shown if prop is supplied */}
          {axisLabels !== undefined && onAxisLabelsChange && (
            <button
              onClick={() => onAxisLabelsChange(!axisLabels)}
              className="p-1 rounded transition-opacity"
              style={{
                color: axisLabels ? "var(--deep-see)" : "var(--fg-muted)",
                opacity: axisLabels ? 1 : 0.5,
              }}
              title={axisLabels ? "Hide axis labels" : "Show axis labels"}
              aria-label={axisLabels ? "Hide axis labels" : "Show axis labels"}
              aria-pressed={axisLabels}
            >
              <CalendarIcon />
            </button>
          )}

          {/* 3-dot action menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 opacity-50 hover:opacity-100"
                style={{ color: "var(--fg-muted)" }}
                aria-label="Widget actions"
              >
                <ThreeDotIcon />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="text-sm">
              <DropdownMenuItem disabled>
                Clone &amp; edit filters
              </DropdownMenuItem>
              <DropdownMenuItem disabled>
                Export as PDF
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => {
                  if (typeof navigator !== "undefined") {
                    navigator.clipboard.writeText(window.location.href);
                  }
                }}
              >
                Copy widget link
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Title row */}
      <div className="mb-3" style={{ height: "20px" }}>
        <h2
          className="text-[13px] font-medium leading-5 truncate"
          style={{ color: "var(--fg-primary)" }}
        >
          {title}
        </h2>
      </div>

      {/* Chart area */}
      <div className="flex-1 min-h-[180px]">
        {children}
      </div>

      {/* Optional footer */}
      {footer && (
        <div className="mt-2 flex items-baseline gap-1">
          <p
            ref={footerRef}
            className="text-[11px] flex-1"
            style={{
              color: "var(--fg-muted)",
              overflow: footerExpanded ? "visible" : "hidden",
              whiteSpace: footerExpanded ? "normal" : "nowrap",
              textOverflow: "ellipsis",
            }}
          >
            {footer}
          </p>
          {footerOverflows && !footerExpanded && (
            <button
              onClick={() => setFooterExpanded(true)}
              className="text-[11px] flex-shrink-0 underline"
              style={{ color: "var(--fg-muted)" }}
            >
              more
            </button>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd sas/web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add sas/web/components/widgets/WidgetCard.tsx
git commit -m "feat(sas): widget card shell with label, title, 3-dot menu, footer"
```

---

## Task 11 — Fleet Critical Trend widget (end-to-end vertical-stack proof)

**Files:**
- Create: `sas/web/components/widgets/FleetCriticalTrend.tsx`

The query this widget sends to `POST /api/query` mirrors the Phase 2 `QueryIn` Pydantic model exactly. Field names come from `sas/api/routes/query.py` and `sas/query/primitives.py`.

- [ ] **Step 1: Create FleetCriticalTrend.tsx**

File: `sas/web/components/widgets/FleetCriticalTrend.tsx`

```typescript
"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { WidgetCard } from "./WidgetCard";
import { runQuery } from "@/lib/api/client";
import type { QueryResult } from "@/lib/api/client";

// echarts-for-react uses browser APIs — must be loaded client-side only
const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// ---------------------------------------------------------------------------
// Query definition — Widget 2: Fleet Critical Trend
// Fields match QueryIn from sas/api/routes/query.py
// ---------------------------------------------------------------------------
const FLEET_CRITICAL_QUERY = {
  lens: "Image",
  traversal: [] as string[],
  time: {
    mode: "last_n_snapshots" as const,
    n: 90,
    granularity: "day" as const,
  },
  measure: "count_open",
  filters: [
    { field: "severity", operator: "eq", value: "Critical" },
  ],
  group_by: [] as string[],
  order_by: null,
  limit: null,
};

// ---------------------------------------------------------------------------
// Skeleton shimmer — fills chart area while loading
// ---------------------------------------------------------------------------
function ChartSkeleton() {
  return (
    <div
      className="w-full h-[220px] rounded animate-pulse"
      style={{ backgroundColor: "var(--bg-surface)" }}
      aria-label="Loading chart data…"
      role="status"
    />
  );
}

// ---------------------------------------------------------------------------
// ECharts option builder
// ---------------------------------------------------------------------------
function buildChartOption(
  result: QueryResult,
  axisLabels: boolean
): object {
  // Expect a single series with key {severity: "Critical"}
  const series = result.series[0] ?? { x: [], y: [] };
  const dates: string[] = series.x as string[];
  const counts: number[] = series.y as number[];

  // Determine x-axis label cadence based on snapshot count (spec §10)
  const n = dates.length;
  let labelInterval = 0; // show all (hidden by default)
  if (axisLabels) {
    if (n > 30) labelInterval = 6;       // ~weekly for 90-day
    else if (n > 7) labelInterval = 2;   // every 3 days for 30-day
    else labelInterval = 0;              // daily for 7-day
  }

  return {
    backgroundColor: "transparent",
    grid: {
      top: 12,
      right: 16,
      bottom: axisLabels ? 52 : 20,  // reserve 44px extra for rotated labels
      left: 48,
      containLabel: false,
    },
    xAxis: {
      type: "category",
      data: dates,
      axisLabel: {
        show: axisLabels,
        interval: labelInterval,
        rotate: n > 7 ? 45 : 0,
        fontSize: 10,
        color: "var(--fg-muted)",
      },
      axisLine: { lineStyle: { color: "var(--border-subtle)" } },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      minInterval: 1,
      axisLabel: {
        fontSize: 10,
        color: "var(--fg-muted)",
        formatter: (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v)),
      },
      splitLine: { lineStyle: { color: "var(--border-subtle)", type: "dashed" } },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      {
        type: "line",
        step: "end",              // step-line — honesty tenet (no smooth interpolation)
        data: counts,
        lineStyle: { color: "var(--deep-see)", width: 2 },
        itemStyle: { color: "var(--deep-see)" },
        symbol: "circle",
        symbolSize: 4,
        showSymbol: false,
        emphasis: { scale: false },
        // Gap markers — null values render as gaps (missing snapshots)
        connectNulls: false,
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "var(--bg-base)",
      borderColor: "var(--border-subtle)",
      textStyle: { color: "var(--fg-primary)", fontSize: 11 },
      formatter: (params: unknown[]) => {
        const p = (params as Array<{ axisValue: string; value: number }>)[0];
        if (!p) return "";
        return `<div style="font-size:11px">
          <div style="color:var(--fg-muted);margin-bottom:2px">${p.axisValue}</div>
          <div><b>${p.value?.toLocaleString("en-GB") ?? "—"}</b> critical open</div>
        </div>`;
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function FleetCriticalTrend() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [axisLabels, setAxisLabels] = useState(false);

  useEffect(() => {
    let cancelled = false;
    runQuery(FLEET_CRITICAL_QUERY)
      .then((r) => { if (!cancelled) setResult(r); })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load data.");
        }
      });
    return () => { cancelled = true; };
  }, []);

  const footer = result
    ? (() => {
        const series = result.series[0];
        if (!series || series.y.length === 0) return undefined;
        const latest = series.y[series.y.length - 1] as number;
        const earliest = series.y[0] as number;
        const delta = latest - earliest;
        const direction = delta < 0 ? "down" : delta > 0 ? "up" : "unchanged";
        const abs = Math.abs(delta);
        return direction === "unchanged"
          ? `Critical open findings are unchanged over this period.`
          : `Critical open findings are ${direction} by ${abs.toLocaleString("en-GB")} vs the start of this window.`;
      })()
    : undefined;

  return (
    <WidgetCard
      label="Fleet Metrics"
      title="Fleet Critical Trend"
      footer={footer}
      axisLabels={axisLabels}
      onAxisLabelsChange={setAxisLabels}
    >
      {error ? (
        <div
          className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--severity-critical)" }}
          role="alert"
        >
          Unable to load data: {error}
        </div>
      ) : result === null ? (
        <ChartSkeleton />
      ) : result.series.length === 0 || result.series[0]?.y.length === 0 ? (
        <div
          className="flex items-center justify-center h-[220px] text-sm"
          style={{ color: "var(--fg-muted)" }}
        >
          No critical findings in this window.
        </div>
      ) : (
        <ReactECharts
          option={buildChartOption(result, axisLabels)}
          style={{ height: "220px", width: "100%" }}
          notMerge
          lazyUpdate={false}
          theme={undefined}
        />
      )}
    </WidgetCard>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles without errors**

```bash
cd sas/web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add sas/web/components/widgets/FleetCriticalTrend.tsx
git commit -m "feat(sas): fleet critical trend widget with echarts step-line and live api data"
```

---

## Task 12 — Dashboard page wiring + README + smoke test

**Files:**
- Create: `sas/web/app/dashboard/page.tsx`
- Create: `sas/web/README.md`

- [ ] **Step 1: Create the dashboard page**

File: `sas/web/app/dashboard/page.tsx`

```typescript
import { AppShell } from "@/components/app-shell/AppShell";
import { FleetCriticalTrend } from "@/components/widgets/FleetCriticalTrend";

export default function DashboardPage() {
  return (
    <AppShell pageTitle="Dashboard">
      {/* 12-column CSS grid — widgets span 4, 6, or 12 columns */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(12, 1fr)",
          gap: "var(--gap-widget)",
        }}
      >
        {/* Widget 2: Fleet Critical Trend — 6-column span */}
        <div style={{ gridColumn: "span 6" }}>
          <FleetCriticalTrend />
        </div>

        {/* Remaining 6-column placeholder — ready for Task 3.2 widgets */}
        <div
          style={{ gridColumn: "span 6" }}
          className="flex items-center justify-center rounded-sas h-[280px]"
        >
          <span
            className="text-sm italic"
            style={{ color: "var(--fg-muted)" }}
          >
            More widgets coming in Phase 3.2
          </span>
        </div>
      </div>
    </AppShell>
  );
}
```

- [ ] **Step 2: Create the README**

File: `sas/web/README.md`

```markdown
# SAS Web — Sysdig Analytics Studio Frontend

Next.js 14 App Router frontend for the Sysdig Analytics Studio (SAS) project.

## Prerequisites

- Node 20+ (`node --version`)
- Phase 2 FastAPI backend running on port 8000

## Running the backend

From the repo root:
```bash
.venv/bin/python -m sas.api.run
```

## Running the frontend

```bash
cd sas/web

# 1. Install dependencies
npm install

# 2. Generate typed API client from live backend (backend must be running)
npm run generate-api

# 3. Copy env file and fill in values (or leave as-is for dev bypass)
cp .env.local.example .env.local

# 4. Start the dev server
npm run dev
```

Open http://localhost:3000. You will be redirected to the sign-in page.
Sign in with any non-empty password (dev bypass) or the value of `SAS_DEMO_PASSWORD`
in `.env.local`.

After signing in, the dashboard renders with the **Fleet Critical Trend** widget
showing live data from the Phase 2 API.

## Production build

```bash
npm run build
npm start
```

## Regenerating API types

After any change to the Phase 2 FastAPI routes or dataclass shapes:
```bash
npm run generate-api
```

The generated file (`lib/api/types.ts`) is git-ignored — always regenerate from a
live backend.
```

- [ ] **Step 3: Final production build smoke test**

With the Phase 2 backend running and `lib/api/types.ts` already generated:

```bash
cd sas/web && npm run build
```

Expected output includes:
```
✓ Compiled successfully
```

No TypeScript errors. No missing-module errors.

- [ ] **Step 4: End-to-end smoke test**

```bash
cd sas/web && npm run dev
```

Manual checklist:
1. Navigate to `http://localhost:3000` — should redirect to `/signin`.
2. Attempt sign-in with empty password — should show "Password is required." error.
3. Sign in with any non-empty password (dev mode) — should redirect to `/dashboard`.
4. Dashboard renders: Deep See sidebar on left (180px), page header with title "Dashboard" and "As of …" timestamp, Fleet Critical Trend widget card.
5. Widget card shows skeleton shimmer briefly, then renders a Deep See step-line chart with real data from the Phase 2 API.
6. Click the calendar icon on the widget card — axis labels appear on the chart.
7. Click the dark mode toggle in the page header — layout switches to dark tokens.
8. Sidebar "Admin" link navigates to `/admin` (404 expected — handled in Phase 3.2).
9. Sign out link (bottom of sidebar) clears session and redirects to `/signin`.
10. Direct navigation to `http://localhost:3000/dashboard` while signed out redirects to `/signin`.

All 10 checks passing = Phase 3.1 deliverable gate met.

- [ ] **Step 5: Commit**

```bash
git add sas/web/app/dashboard/ sas/web/README.md
git commit -m "feat(sas): dashboard page, app shell wired, phase 3.1 complete"
```

---

## Summary

Phase 3.1 delivers exactly what the spec's §18 deliverable gate requires: sign in, see the sidebar and header, see Fleet Critical Trend rendering live data.

| Task | Deliverable |
|---|---|
| 1 | Next.js 14 scaffold in `sas/web/` |
| 2 | `tokens.css` with full Sysdig brand palette, light + dark |
| 3 | shadcn/ui installed and mapped to tokens |
| 4 | `npm run generate-api` produces typed client from live OpenAPI spec |
| 5 | `/api/signin` cookie route with jose JWT |
| 6 | `/signin` page matching Sysdig brand |
| 7 | `middleware.ts` protecting all routes |
| 8 | Deep See sidebar (180px) with nav items |
| 9 | PageHeader (48px), BreadcrumbStrip (28px), AppShell wrapper |
| 10 | `WidgetCard` reusable shell (used by all 10 widgets in Phase 3.2+) |
| 11 | `FleetCriticalTrend` — full vertical stack: auth → API → typed result → ECharts |
| 12 | Dashboard page, README, production build + smoke test |

**Phase 3.2 entry condition:** Aaron signs off on the smoke test checklist in Task 12 Step 4. Do not start Phase 3.2 until sign-off is received.
