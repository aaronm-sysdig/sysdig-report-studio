import { AppShell } from "@/components/app-shell/AppShell";
import { FleetCriticalTrend } from "@/components/widgets/FleetCriticalTrend";
import { NewFixedRegressed } from "@/components/widgets/NewFixedRegressed";
import { KevRansomwareExposure } from "@/components/widgets/KevRansomwareExposure";
import { RepositoryTagHygiene } from "@/components/widgets/RepositoryTagHygiene";
import { ImageInventoryGrid } from "@/components/widgets/ImageInventoryGrid";
import { FindingsTable } from "@/components/widgets/FindingsTable";

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
        {/* Row 1 — fleet metrics side by side */}
        <div style={{ gridColumn: "span 6" }}>
          <FleetCriticalTrend />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <NewFixedRegressed />
        </div>

        {/* Row 2 — exposure + repository hygiene */}
        <div style={{ gridColumn: "span 6" }}>
          <KevRansomwareExposure />
        </div>
        <div style={{ gridColumn: "span 6" }}>
          <RepositoryTagHygiene />
        </div>

        {/* Row 3 — image inventory full width */}
        <div style={{ gridColumn: "span 12" }}>
          <ImageInventoryGrid />
        </div>

        {/* Row 4 — findings table full width */}
        <div style={{ gridColumn: "span 12" }}>
          <FindingsTable />
        </div>
      </div>
    </AppShell>
  );
}
