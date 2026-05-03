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

export type FindingsResponse = components["schemas"]["FindingsResponse"];

/**
 * GET /api/findings — paginated raw finding_state rows.
 */
export async function getFindings(opts: {
  limit?: number;
  offset?: number;
  severity?: string;
  state?: string;
  fix_available?: boolean;
  in_use?: boolean;
  public_exploit?: boolean;
}): Promise<FindingsResponse> {
  const params = new URLSearchParams();
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  if (opts.offset !== undefined) params.set("offset", String(opts.offset));
  if (opts.severity) params.set("severity", opts.severity);
  if (opts.state) params.set("state", opts.state);
  if (opts.fix_available !== undefined) params.set("fix_available", opts.fix_available ? "1" : "0");
  if (opts.in_use !== undefined) params.set("in_use", opts.in_use ? "1" : "0");
  if (opts.public_exploit !== undefined) params.set("public_exploit", opts.public_exploit ? "1" : "0");
  return apiFetch<FindingsResponse>(`/api/findings?${params.toString()}`);
}

export type WorkloadCountsResponse = components["schemas"]["WorkloadCountsResponse"];

/**
 * GET /api/workload-counts — CVE-level workload blast radius counts.
 */
export async function getWorkloadCounts(): Promise<WorkloadCountsResponse> {
  return apiFetch<WorkloadCountsResponse>("/api/workload-counts");
}
