import { Suspense } from "react";
import { AppShell } from "@/components/app-shell/AppShell";
import { FleetCriticalTrend } from "@/components/widgets/FleetCriticalTrend";
import { FleetSeveritySnapshot } from "@/components/widgets/FleetSeveritySnapshot";
import { FindingsTable } from "@/components/widgets/FindingsTable";
import { ImageInventoryGrid } from "@/components/widgets/ImageInventoryGrid";
import { ImageRemediationStory } from "@/components/widgets/ImageRemediationStory";
import { KevRansomwareExposure } from "@/components/widgets/KevRansomwareExposure";
import { NewFixedRegressed } from "@/components/widgets/NewFixedRegressed";
import { RepositoryTagHygiene } from "@/components/widgets/RepositoryTagHygiene";

export default function DashboardPage() {
  return (
    <AppShell pageTitle="Dashboard">
      {/* 12-column CSS grid — widgets span 6 or 12 columns */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(12, 1fr)",
          gap: "var(--gap-widget)",
        }}
      >
        {/* Row 1 — fleet severity snapshot, full width */}
        <div style={{ gridColumn: "span 12" }}>
          <FleetSeveritySnapshot />
        </div>

        {/* Row 2 — flagship widget, full width */}
        <div style={{ gridColumn: "span 12" }}>
          <ImageRemediationStory />
        </div>

        {/* Row 3 — fleet metrics side by side */}
        <div style={{ gridColumn: "span 6" }}>
          <FleetCriticalTrend />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <NewFixedRegressed />
        </div>

        {/* Row 4 — exposure + repository hygiene */}
        <div style={{ gridColumn: "span 6" }}>
          <KevRansomwareExposure />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <RepositoryTagHygiene />
        </div>

        {/* Row 5 — image inventory full width */}
        <div style={{ gridColumn: "span 12" }}>
          <ImageInventoryGrid />
        </div>

        {/* Row 6 — findings table full width */}
        <div style={{ gridColumn: "span 12" }}>
          <Suspense fallback={<div className="animate-pulse" style={{ minHeight: "200px" }} />}>
            <FindingsTable />
          </Suspense>
        </div>
      </div>
    </AppShell>
  );
}
