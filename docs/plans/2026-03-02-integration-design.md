# Integration Design: Sysdig Report Studio + Sysdig Analytics Suite

**Date:** 2026-03-02
**Branch:** vuln-reg-posture
**Status:** Approved

## Overview

Merge Prakash's `sysdig-coding` analytics app into `sysdig-report-studio` to form a single unified reporting and analytics platform. The result is one app with clearly separate sections, a shared sidebar navigation, and a single global auth config.

## Goals

- Single unified Streamlit app covering both analytics exploration and report building
- No loss of functionality from either repo
- Shared global config (API token + region) — enter once, applies everywhere
- Clean seam preserved for future wiring of analytics charts into the report builder
- Prakash's analytics taken as-is (simplification deferred to a later stage)

## Architecture

### Approach: Multi-module, single entry point (Option B)

A thin `app.py` handles navigation and global config only. All page logic lives in dedicated modules. Existing modules (`database.py`, `scheduler.py`, `pdf_generator.py`, `charts.py`) are untouched.

### File Structure

```
sysdig-report-studio/
├── app.py                  # Entry point: navigation routing + global config sidebar
├── config.py               # Expanded: unified regions, API helpers, snapshot dir path
├── database.py             # Unchanged
├── scheduler.py            # Unchanged
├── pdf_generator.py        # Unchanged
├── charts.py               # Unchanged
│
├── report_studio.py        # Our report builder (extracted from app.py)
├── posture_analytics.py    # From Prakash's posture_analytics_page()
├── registry_analytics.py   # From Prakash's vuln_analytics_page()
├── cve_risk.py             # From Prakash's cve_risk_page()
└── engineering_fix.py      # From Prakash's engineering_fix_page()
```

Files retired (superseded by richer Prakash versions):
- `posture.py` → replaced by `posture_analytics.py`
- `registry_vulns.py` → replaced by `registry_analytics.py`

## Navigation

Sidebar radio navigation extending Prakash's "Select Tool" pattern:

```
🛡️ Sysdig Platform
──────────────────
  Region:    [ AU1 ▼ ]
  API Token: [••••••••]
──────────────────
Select Tool:
  ○ 📋 Posture Analytics
  ○ 🔍 Registry Vulnerabilities
  ○ 📊 CVE Risk Overview
  ○ 🔧 Engineering Fix View
  ○ 📄 Report Studio
──────────────────
[tool-specific sidebar options below]
```

Tool-specific sidebar options (file uploads, date pickers, etc.) render below the nav divider, consistent with both apps' existing behaviour.

## Global Config

- Centralised in `config.py` via a `get_api_config()` helper
- Reads `st.session_state` for `api_token` and `base_url`
- All modules call this helper — no per-page auth inputs
- If config is not set, modules display a friendly prompt directing the user to the sidebar
- Snapshot directory path (`~/sysdig-vuln-data/`) moved to a constant in `config.py`

## The `go.Figure` Seam

All four analytics modules follow a two-layer structure:

```
posture_analytics.py
├── chart functions    → return go.Figure  (no st.* calls inside)
└── render_page()      → calls st.plotly_chart(fig) on those figures
```

This is a minimal structural change to Prakash's code — logic is untouched, we only separate "build figure" from "display figure". This preserves a natural integration point for the report builder to call `fig.to_image()` in a future phase without requiring a rewrite.

`report_studio.py` already follows this pattern via `charts.py`.

## Data Flow & State Management

Session state keys are namespaced by module to avoid collisions.

| Section | Data Source | State Lifetime |
|---|---|---|
| Posture Analytics | Uploaded CSV/GZ files | Session |
| Registry Vulnerabilities | API fetch → disk snapshots | Persistent (disk) |
| CVE Risk Overview | API fetch | Session |
| Engineering Fix View | API fetch | Session |
| Report Studio | SYSQL API + SQLite DB | Persistent (DB) |

No cross-module data sharing. Each section is self-contained.

## Requirements

`requirements.txt` additions:
- `streamlit-sortables>=0.3.0` (Prakash uses it; not currently listed in ours)

Retained from ours:
- `reportlab>=4.0.0`
- `pyyaml>=6.0`

## What Changes

| File | Action |
|---|---|
| `app.py` | Stripped to navigation + global config only |
| `config.py` | Expanded with `get_api_config()` and snapshot dir constant |
| `report_studio.py` | New — report builder logic extracted from `app.py` |
| `posture_analytics.py` | New — from Prakash's `posture_analytics_page()` |
| `registry_analytics.py` | New — from Prakash's `vuln_analytics_page()` |
| `cve_risk.py` | New — from Prakash's `cve_risk_page()` |
| `engineering_fix.py` | New — from Prakash's `engineering_fix_page()` |
| `posture.py` | Retired |
| `registry_vulns.py` | Retired |
| `requirements.txt` | Add `streamlit-sortables` |

## Future Work (Out of Scope)

- Wiring analytics chart functions into the report builder (seam is preserved, work deferred)
- Simplifying Overall project
