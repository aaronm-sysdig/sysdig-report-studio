# SAS Web — Sysdig Analytics Studio Frontend

Next.js 16 App Router frontend for the Sysdig Analytics Studio (SAS) project.

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

## Architecture notes

- **Routing:** Next.js 16 App Router. Middleware lives in `proxy.ts` (renamed from `middleware.ts` in NJ16).
- **Auth:** cosmetic JWT-in-cookie. Swap for real OIDC in Phase 5+.
- **Design tokens:** `styles/tokens.css` — single source of truth. Dark mode via `[data-theme="dark"]`.
- **Typed API client:** auto-generated from FastAPI OpenAPI spec. Regenerate with `npm run generate-api`.
