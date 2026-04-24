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
