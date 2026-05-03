"use client";

export interface WeightConfig {
  severityGate: string[];
  weights: {
    Critical: number;
    High: number;
    Medium: number;
    Low: number;
    Negligible: number;
    in_use: number;
    fix_available: number;
    public_exploit: number;
  };
}

export const DEFAULT_WEIGHTS: WeightConfig = {
  severityGate: ["Critical", "High"],
  weights: {
    Critical: 2,
    High: 1,
    Medium: 0,
    Low: 0,
    Negligible: 0,
    in_use: 1,
    fix_available: 1,
    public_exploit: 1,
  },
};

const STORAGE_KEY = "sas:weighted-weights";

export function loadWeights(): WeightConfig {
  if (typeof window === "undefined") return DEFAULT_WEIGHTS;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_WEIGHTS;
    const parsed = JSON.parse(raw);
    // Validate structure and merge with defaults to handle partial configs
    if (
      Array.isArray(parsed.severityGate) &&
      parsed.weights &&
      typeof parsed.weights.Critical === "number"
    ) {
      return {
        severityGate: parsed.severityGate,
        weights: { ...DEFAULT_WEIGHTS.weights, ...parsed.weights },
      };
    }
    return DEFAULT_WEIGHTS;
  } catch {
    return DEFAULT_WEIGHTS;
  }
}

export function saveWeights(config: WeightConfig): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // Storage full or unavailable — silent fail
  }
}

/** Future: migrate to user profile in database when multi-tenant auth is implemented. */
