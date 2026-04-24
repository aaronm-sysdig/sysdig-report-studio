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
          style={{
            gridColumn: "span 6",
            borderRadius: "var(--radius)",
            border: "1px dashed var(--border-subtle)",
          }}
          className="flex items-center justify-center h-[280px]"
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
