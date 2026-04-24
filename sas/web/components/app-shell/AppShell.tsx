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
      <Sidebar />

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
