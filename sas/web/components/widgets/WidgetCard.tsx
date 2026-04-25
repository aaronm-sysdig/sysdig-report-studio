"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface WidgetCardProps {
  /** 10px uppercase category label, e.g. "FLEET METRICS" */
  label: string;
  /** 13px card title, e.g. "Fleet Critical Trend" */
  title: string;
  /** Optional footer narrative text. Truncated to 1 line with expand link. */
  footer?: string;
  /** Optional axis-labels toggle state — pass undefined to hide the toggle. */
  axisLabels?: boolean;
  onAxisLabelsChange?: (on: boolean) => void;
  children: React.ReactNode;
}

function ThreeDotIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
      <circle cx="8" cy="3" r="1.2" />
      <circle cx="8" cy="8" r="1.2" />
      <circle cx="8" cy="13" r="1.2" />
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="2" y="3" width="12" height="11" rx="1.5" />
      <path d="M5 1v4M11 1v4M2 7h12" strokeLinecap="round" />
    </svg>
  );
}

function ExpandIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ContractIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M6 2v4H2M10 2v4h4M6 14v-4H2M10 14v-4h4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function WidgetCard({
  label,
  title,
  footer,
  axisLabels,
  onAxisLabelsChange,
  children,
}: WidgetCardProps) {
  const [footerExpanded, setFooterExpanded] = useState(false);
  const [footerOverflows, setFooterOverflows] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const footerRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    if (footerRef.current) {
      setFooterOverflows(
        footerRef.current.scrollWidth > footerRef.current.clientWidth
      );
    }
  }, [footer]);

  useEffect(() => {
    if (!isFullScreen) return;

    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsFullScreen(false);
    };
    window.addEventListener("keydown", onKey);

    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", onKey);
    };
  }, [isFullScreen]);

  return (
    <div
      className="flex flex-col"
      style={
        isFullScreen
          ? {
              position: "fixed",
              inset: 0,
              zIndex: 50,
              backgroundColor: "var(--bg-base)",
              border: "none",
              borderRadius: 0,
              padding: "20px 24px",
              transitionDuration: "var(--dur-standard)",
              transitionProperty: "border-color",
            }
          : {
              backgroundColor: "var(--bg-base)",
              border: "1px solid var(--border-subtle)",
              padding: "var(--p-card)",
              borderRadius: "var(--radius)",
              boxShadow: "var(--shadow-card)",
              transitionDuration: "var(--dur-standard)",
              transitionProperty: "border-color",
            }
      }
      onMouseEnter={(e) => {
        if (!isFullScreen) {
          (e.currentTarget as HTMLElement).style.borderColor = "var(--border-strong)";
        }
      }}
      onMouseLeave={(e) => {
        if (!isFullScreen) {
          (e.currentTarget as HTMLElement).style.borderColor = "var(--border-subtle)";
        }
      }}
    >
      {/* Label row */}
      <div className="flex items-center justify-between mb-0.5" style={{ height: "24px" }}>
        <span
          className="text-[10px] font-medium tracking-widest uppercase"
          style={{ color: "var(--fg-muted)" }}
        >
          {label}
        </span>

        <div className="flex items-center gap-1">
          {/* Axis-labels toggle */}
          {axisLabels !== undefined && onAxisLabelsChange && (
            <button
              onClick={() => onAxisLabelsChange(!axisLabels)}
              className="p-1 rounded transition-opacity"
              style={{
                color: axisLabels ? "var(--deep-see)" : "var(--fg-muted)",
                opacity: axisLabels ? 1 : 0.5,
              }}
              title={axisLabels ? "Hide axis labels" : "Show axis labels"}
              aria-label={axisLabels ? "Hide axis labels" : "Show axis labels"}
              aria-pressed={axisLabels}
            >
              <CalendarIcon />
            </button>
          )}

          {/* Full-screen toggle */}
          <button
            onClick={() => setIsFullScreen((v) => !v)}
            className="p-1 rounded transition-opacity"
            style={{ color: "var(--fg-muted)", opacity: 0.5 }}
            title={isFullScreen ? "Exit full screen (Esc)" : "Full screen"}
            aria-label={isFullScreen ? "Exit full screen" : "Enter full screen"}
            aria-pressed={isFullScreen}
            onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.opacity = "1")}
            onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.opacity = "0.5")}
          >
            {isFullScreen ? <ContractIcon /> : <ExpandIcon />}
          </button>

          {/* 3-dot action menu */}
          <DropdownMenu>
            <DropdownMenuTrigger
              className="inline-flex h-6 w-6 items-center justify-center rounded-lg opacity-50 transition-opacity hover:bg-muted hover:opacity-100"
              style={{ color: "var(--fg-muted)" }}
              aria-label="Widget actions"
            >
              <ThreeDotIcon />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="text-sm">
              <DropdownMenuItem disabled>
                Clone &amp; edit filters
              </DropdownMenuItem>
              <DropdownMenuItem disabled>
                Export as PDF
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => {
                  if (typeof navigator !== "undefined") {
                    navigator.clipboard.writeText(window.location.href);
                  }
                }}
              >
                Copy widget link
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Title row */}
      <div className="mb-3" style={{ height: "20px" }}>
        <h2
          className="text-[13px] font-medium leading-5 truncate"
          style={{ color: "var(--fg-primary)" }}
        >
          {title}
        </h2>
      </div>

      {/* Chart area */}
      <div className="flex-1 min-h-[180px]">
        {children}
      </div>

      {/* Optional footer */}
      {footer && (
        <div className="mt-2 flex items-baseline gap-1">
          <p
            ref={footerRef}
            className="text-[11px] flex-1"
            style={{
              color: "var(--fg-muted)",
              overflow: footerExpanded ? "visible" : "hidden",
              whiteSpace: footerExpanded ? "normal" : "nowrap",
              textOverflow: "ellipsis",
            }}
          >
            {footer}
          </p>
          {footerOverflows && !footerExpanded && (
            <button
              onClick={() => setFooterExpanded(true)}
              className="text-[11px] flex-shrink-0 underline"
              style={{ color: "var(--fg-muted)" }}
            >
              more
            </button>
          )}
        </div>
      )}
    </div>
  );
}
