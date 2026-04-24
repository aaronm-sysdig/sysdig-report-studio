"use client";

import { useState, useEffect } from "react";

interface PageHeaderProps {
  title: string;
  asOf?: string;
  children?: React.ReactNode;
}

function formatAsOf(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

function DarkModeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("sas-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = stored ? stored === "dark" : prefersDark;
    setDark(isDark);
    document.body.setAttribute("data-theme", isDark ? "dark" : "light");
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.body.setAttribute("data-theme", next ? "dark" : "light");
    localStorage.setItem("sas-theme", next ? "dark" : "light");
  }

  return (
    <button
      onClick={toggle}
      className="p-1 rounded opacity-60 hover:opacity-100 transition-opacity"
      style={{ color: "var(--fg-muted)" }}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="8" cy="8" r="3" />
          <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.1 3.1l1.4 1.4M11.5 11.5l1.4 1.4M11.5 3.1l-1.4 1.4M3.1 11.5l1.4 1.4" strokeLinecap="round" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M13.6 11A6 6 0 015 2.4a6 6 0 100 11.2 6 6 0 008.6-2.6z" />
        </svg>
      )}
    </button>
  );
}

export function PageHeader({ title, asOf, children }: PageHeaderProps) {
  const timestamp = asOf ?? new Date().toISOString();

  return (
    <header
      className="flex items-center justify-between flex-shrink-0 px-5 border-b"
      style={{
        height: "var(--h-topbar)",
        backgroundColor: "var(--bg-base)",
        borderColor: "var(--border-subtle)",
      }}
    >
      <span
        className="text-sm font-medium"
        style={{ color: "var(--fg-primary)" }}
      >
        {title}
      </span>

      <div className="flex items-center gap-3">
        <span
          className="text-xs"
          style={{ color: "var(--fg-muted)" }}
        >
          As of {formatAsOf(timestamp)}
        </span>
        <DarkModeToggle />
        {children}
      </div>
    </header>
  );
}
