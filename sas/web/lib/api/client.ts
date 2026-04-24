/**
 * Typed fetch wrapper for the Phase 2 FastAPI backend.
 * All methods are async and throw on non-2xx responses.
 */

import type { components } from "./types";

// Re-export the commonly-used types so consumers don't need to dig through components
export type QueryIn = components["schemas"]["QueryIn"];
export type TimeWindowIn = components["schemas"]["TimeWindowIn"];
export type FilterIn = components["schemas"]["FilterIn"];
export type OrderingIn = components["schemas"]["OrderingIn"];
export type QueryResult = components["schemas"]["QueryResultOut"];
export type Series = components["schemas"]["SeriesOut"];

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
 */
export async function getEntities(
  lens: string,
  params?: Record<string, string>
): Promise<unknown[]> {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  return apiFetch<unknown[]>(`/api/entities/${lens}${qs}`);
}
