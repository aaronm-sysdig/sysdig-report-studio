"use client";

import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { useMemo } from "react";
import type { DrillField, DrillMode, DrillFilter, UseDrillFilterReturn } from "./drill-types";

const DRILL_FIELDS: DrillField[] = ["cve", "package", "image"];

export function useDrillFilter(): UseDrillFilterReturn {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const filter = useMemo<DrillFilter>(() => {
    // Find which drill field is set in URL
    for (const field of DRILL_FIELDS) {
      const value = searchParams.get(field);
      if (value) {
        return {
          field,
          value,
          mode: (searchParams.get("mode") as DrillMode) || "findings",
        };
      }
    }
    return { mode: "findings" };
  }, [searchParams]);

  const isFiltered = !!filter.field && !!filter.value;

  const applyFilter = (field: DrillField, value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    // Clear any existing drill fields
    for (const f of DRILL_FIELDS) {
      params.delete(f);
    }
    params.delete("mode"); // reset mode
    // Set new filter
    params.set(field, value);
    router.push(`${pathname}?${params.toString()}`, { scroll: false });
  };

  const setMode = (mode: DrillMode) => {
    const params = new URLSearchParams(searchParams.toString());
    if (mode === "findings") {
      params.delete("mode");
    } else {
      params.set("mode", mode);
    }
    router.push(`${pathname}?${params.toString()}`, { scroll: false });
  };

  const clearFilter = () => {
    const params = new URLSearchParams(searchParams.toString());
    for (const f of DRILL_FIELDS) {
      params.delete(f);
    }
    params.delete("mode");
    // Push clean URL (or just pathname if no other params)
    const remaining = params.toString();
    router.push(remaining ? `${pathname}?${remaining}` : pathname, { scroll: false });
  };

  return { filter, applyFilter, setMode, clearFilter, isFiltered };
}
