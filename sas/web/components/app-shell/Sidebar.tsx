"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Simple inline SVG icons — no external icon dep needed for 3.1
function IconDashboard({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
      <rect x="1" y="1" width="6" height="6" rx="1" />
      <rect x="9" y="1" width="6" height="6" rx="1" />
      <rect x="1" y="9" width="6" height="6" rx="1" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
    </svg>
  );
}

function IconExplore({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="7" cy="7" r="5" />
      <path d="M11 11l3 3" strokeLinecap="round" />
    </svg>
  );
}

function IconAdmin({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="5" r="2.5" />
      <path d="M2 13c0-3.3 2.7-6 6-6s6 2.7 6 6" strokeLinecap="round" />
    </svg>
  );
}

function IconSignOut({ className }: { className?: string }) {
  return (
    <svg className={className} width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M6 3H3a1 1 0 00-1 1v8a1 1 0 001 1h3" strokeLinecap="round" />
      <path d="M10 11l3-3-3-3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M13 8H6" strokeLinecap="round" />
    </svg>
  );
}

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", Icon: IconDashboard },
  { href: "/explore", label: "Explore", Icon: IconExplore },
  { href: "/admin", label: "Admin", Icon: IconAdmin },
];

export function Sidebar() {
  const pathname = usePathname();

  function isActive(href: string) {
    if (href === "/dashboard") return pathname === "/dashboard" || pathname.startsWith("/dashboard/");
    return pathname.startsWith(href);
  }

  return (
    <aside
      className="flex flex-col h-full flex-shrink-0"
      style={{
        width: "var(--w-sidebar)",
        backgroundColor: "var(--bg-sidebar)",
      }}
    >
      {/* Wordmark */}
      <div className="flex items-center gap-2 px-4 py-4 flex-shrink-0">
        <span
          className="text-lg font-bold tracking-tight"
          style={{ color: "var(--white)" }}
        >
          sysdig
        </span>
        <span
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: "var(--lumin)" }}
          aria-hidden="true"
        />
      </div>

      {/* Product label */}
      <div
        className="px-4 pb-3 text-[10px] font-medium tracking-widest uppercase"
        style={{ color: "var(--fg-on-sidebar-muted)" }}
      >
        Analytics Studio
      </div>

      {/* Nav items */}
      <nav className="flex-1 px-2 space-y-0.5 overflow-y-auto" aria-label="Main navigation">
        {NAV_ITEMS.map(({ href, label, Icon }) => {
          const active = isActive(href);
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-2.5 px-3 text-sm font-medium transition-colors"
              style={{
                height: "var(--h-sidebar-row)",
                borderRadius: "var(--radius)",
                transitionDuration: "var(--dur-standard)",
                color: active ? "var(--white)" : "rgba(255,255,255,0.75)",
                backgroundColor: active
                  ? "var(--bg-sidebar-active)"
                  : "transparent",
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.backgroundColor =
                    "var(--bg-sidebar-hover)";
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.backgroundColor =
                    "transparent";
                }
              }}
              aria-current={active ? "page" : undefined}
            >
              <Icon className="flex-shrink-0 opacity-80" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* My Dashboards — stub for 3.1 */}
      <div className="px-2 pt-2 pb-1">
        <div
          className="px-3 py-1 text-[10px] font-medium tracking-widest uppercase"
          style={{ color: "var(--fg-on-sidebar-muted)" }}
        >
          My Dashboards
        </div>
        <p
          className="px-3 py-1 text-[11px] italic"
          style={{ color: "rgba(255,255,255,0.35)" }}
        >
          No saved dashboards yet.
        </p>
      </div>

      {/* User info — pinned at bottom */}
      <div
        className="flex items-center justify-between px-3 py-3 mt-auto border-t"
        style={{ borderColor: "rgba(255,255,255,0.1)" }}
      >
        <div className="flex items-center gap-2 min-w-0">
          {/* Avatar */}
          <div
            className="w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-bold"
            style={{
              backgroundColor: "var(--lumin)",
              color: "var(--deep-see)",
            }}
            aria-hidden="true"
          >
            D
          </div>
          <span
            className="text-[11px] truncate"
            style={{ color: "rgba(255,255,255,0.7)" }}
          >
            demo
          </span>
        </div>
        <a
          href="/api/auth/signout"
          className="flex-shrink-0 opacity-60 hover:opacity-100 transition-opacity"
          style={{ color: "var(--white)" }}
          title="Sign out"
          aria-label="Sign out"
        >
          <IconSignOut />
        </a>
      </div>
    </aside>
  );
}
