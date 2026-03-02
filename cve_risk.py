"""
CVE Risk Overview page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
All chart helpers already return go.Figure — no seam refactor needed.
"""
import re
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import get_api_config

# ── CVE constants ──────────────────────────────────────────────────────────────
CVE_DEFAULT_BASE   = "https://app.au1.sysdig.com"
CVE_API_TIMEOUT    = 20
CVE_BY_CVE_PATH    = "/api/secure/analytics/v1/data/vulnerabilities/findings/by-cve"
CVE_EPSS_THRESHOLD = 0.50   # 50 %
CVE_TOP_N          = 50

CVE_SEVERITY_COLOR = {
    "Critical":   "#9B3FBF",
    "High":       "#E53935",
    "Medium":     "#FB8C00",
    "Low":        "#1E88E5",
    "Negligible": "#78909C",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0bec5", size=12),
    margin=dict(t=40, b=20, l=20, r=20),
)


def _cve_headers(token: str) -> dict:
    return {
        "Authorization":    f"Bearer {token}",
        "Accept":           "application/json",
        "X-Sysdig-Product": "SDS",
    }


def _fetch_top_cves(base: str, token: str) -> list:
    """Paginate /by-cve; collect items with EPSS >= threshold."""
    hdrs   = _cve_headers(token)
    params = {"severity_in": "critical,high,medium", "limit": 200}
    qualifying = []
    while True:
        r = requests.get(f"{base}{CVE_BY_CVE_PATH}", headers=hdrs,
                         params=params, timeout=CVE_API_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("data", []):
            if float(item.get("epssScore") or 0) >= CVE_EPSS_THRESHOLD:
                qualifying.append(item)
        meta   = payload.get("meta") or {}
        cursor = payload.get("cursor") or {}
        if (len(qualifying) >= CVE_TOP_N
                or not meta.get("hasMore")
                or not cursor.get("next")):
            break
        params = {**params, "cursor": cursor["next"]}
    qualifying.sort(key=lambda x: float(x.get("epssScore") or 0), reverse=True)
    return qualifying[:CVE_TOP_N]


def _normalize_cve(item: dict) -> dict:
    epss        = float(item.get("epssScore") or 0)
    cvss        = float(item.get("cvssScore") or 0)
    exploitable = bool(item.get("hasExploit"))
    kev         = bool(item.get("hasCisaKev"))
    risk_score  = round(epss * 40 + (cvss / 10) * 30 + exploitable * 20 + kev * 10, 1)
    return {
        "cveId":         item.get("name", "Unknown"),
        "severity":      (item.get("severity") or "Unknown").capitalize(),
        "epssScore":     epss,
        "cvssScore":     cvss,
        "fixAvailable":  bool(item.get("isFixAvailable")),
        "exploitable":   exploitable,
        "hasCisaKev":    kev,
        "findingsCount": int(item.get("findingsCount") or 0),
        "inUse":         bool(item.get("inUse") or item.get("isInUse") or False),
        "riskScore":     risk_score,
    }


def _load_cves_with_progress(base: str, token: str, status_ctx) -> tuple:
    status_ctx.write(
        f"**Querying Findings API** — top CVEs with EPSS > "
        f"{CVE_EPSS_THRESHOLD*100:.0f}% (critical/high/medium)…"
    )
    items = _fetch_top_cves(base, token)
    if not items:
        return [], [f"No CVEs with EPSS > {CVE_EPSS_THRESHOLD*100:.0f}% found."]
    normalised = [_normalize_cve(it) for it in items]
    in_use  = sum(1 for c in normalised if c["inUse"])
    not_use = sum(1 for c in normalised if not c["inUse"])
    status_ctx.write(
        f"  ✓ **{len(normalised)}** CVE(s) — "
        f"**{in_use}** in-use · **{not_use}** not-in-use"
    )
    return normalised, []


def _cve_chart_severity_donut(df) -> go.Figure:
    counts = df["severity"].value_counts().reset_index()
    counts.columns = ["severity", "count"]
    colors = [CVE_SEVERITY_COLOR.get(s, "#78909C") for s in counts["severity"]]
    fig = go.Figure(go.Pie(
        labels=counts["severity"], values=counts["count"],
        marker=dict(colors=colors, line=dict(width=2, color="#12161f")),
        hole=0.55, textinfo="label+value", textfont=dict(size=12),
        hovertemplate="%{label}: %{value} CVEs (%{percent})<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
    return fig


def _cve_chart_fix_donut(df) -> go.Figure:
    fix_yes = int(df["fixAvailable"].sum())
    fix_no  = len(df) - fix_yes
    fig = go.Figure(go.Pie(
        labels=["Fix Available", "No Fix Yet"], values=[fix_yes, fix_no],
        marker=dict(colors=["#00C853", "#E53935"], line=dict(width=2, color="#12161f")),
        hole=0.55, textinfo="label+value", textfont=dict(size=12),
        hovertemplate="%{label}: %{value} CVEs (%{percent})<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
    return fig


def _cve_chart_epss_dist(df) -> go.Figure:
    bins   = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    df2 = df.copy()
    df2["epss_bucket"] = pd.cut(df2["epssScore"], bins=bins,
                                labels=labels, include_lowest=True)
    counts = (
        df2.groupby(["epss_bucket", "severity"], observed=True)
        .size().reset_index(name="count")
    )
    sev_order = ["Critical", "High", "Medium", "Low", "Negligible"]
    counts["severity"] = pd.Categorical(counts["severity"],
                                        categories=sev_order, ordered=True)
    counts = counts.sort_values(["epss_bucket", "severity"])
    fig = px.bar(
        counts, x="epss_bucket", y="count", color="severity",
        color_discrete_map=CVE_SEVERITY_COLOR, barmode="stack",
        labels={"epss_bucket": "EPSS Range", "count": "CVE Count", "severity": "Severity"},
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      xaxis=dict(gridcolor="#1e2d3d"),
                      yaxis=dict(gridcolor="#1e2d3d", title="CVE Count"),
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=1.02, xanchor="right", x=1))
    return fig


def _cve_chart_key_flags(df) -> go.Figure:
    cats   = ["Exploitable", "CISA KEV", "Has Fix"]
    values = [int(df["exploitable"].sum()),
              int(df["hasCisaKev"].sum()),
              int(df["fixAvailable"].sum())]
    fig = go.Figure(go.Bar(
        x=cats, y=values,
        marker=dict(color=["#E53935", "#9B3FBF", "#00C853"], line=dict(width=0)),
        text=values, textposition="outside",
        hovertemplate="%{x}: %{y} CVEs<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      yaxis=dict(gridcolor="#1e2d3d", title="CVE Count"),
                      xaxis=dict(gridcolor="#1e2d3d"), showlegend=False)
    return fig


def _cve_render_section(cves: list, label: str, header_class: str) -> None:
    st.markdown(f'''<div class="section-hdr {header_class}">{label}</div>''',
                unsafe_allow_html=True)
    if not cves:
        st.info("No CVEs in this category.")
        return
    df = pd.DataFrame(cves)
    for _col, _default in [("exploitable", False), ("hasCisaKev", False),
                            ("fixAvailable", False), ("cvssScore", 0.0),
                            ("epssScore", 0.0)]:
        if _col not in df.columns:
            df[_col] = _default
    total       = len(df)
    avg_epss    = df["epssScore"].mean() * 100
    exploitable = int(df["exploitable"].sum())
    kev         = int(df["hasCisaKev"].sum())
    fixable     = int(df["fixAvailable"].sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CVEs",        total)
    c2.metric("Avg EPSS",   f"{avg_epss:.1f}%")
    c3.metric("Exploitable", exploitable, "known exploits")
    c4.metric("CISA KEV",    kev,         "actively exploited")
    c5.metric("Has Fix",     fixable,     f"{fixable/total*100:.0f}% fixable")
    st.markdown("<br>", unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown(
            "<div style='text-align:center;color:#90a4ae;font-size:.83rem;"
            "margin-bottom:4px'>Severity Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(_cve_chart_severity_donut(df), use_container_width=True,
                        config={"displayModeBar": False})
    with r1c2:
        st.markdown(
            "<div style='text-align:center;color:#90a4ae;font-size:.83rem;"
            "margin-bottom:4px'>CVEs by EPSS Range &amp; Severity</div>",
            unsafe_allow_html=True)
        st.plotly_chart(_cve_chart_epss_dist(df), use_container_width=True,
                        config={"displayModeBar": False})
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown(
            "<div style='text-align:center;color:#90a4ae;font-size:.83rem;"
            "margin-bottom:4px'>Fix Availability</div>", unsafe_allow_html=True)
        st.plotly_chart(_cve_chart_fix_donut(df), use_container_width=True,
                        config={"displayModeBar": False})
    with r2c2:
        st.markdown(
            "<div style='text-align:center;color:#90a4ae;font-size:.83rem;"
            "margin-bottom:4px'>Key Risk Flags</div>", unsafe_allow_html=True)
        st.plotly_chart(_cve_chart_key_flags(df), use_container_width=True,
                        config={"displayModeBar": False})


def render_page():
    """CVE Risk Overview — top CVEs with EPSS > 50%, split In-Use / Not-In-Use."""

    api_token, api_base = get_api_config()
    api_base = api_base.rstrip("/")

    if not api_token:
        st.info("Set your API token in the sidebar to get started.")
        return

    # ── Shared CSS for CVE pages ──────────────────────────────────────────────
    st.markdown("""
<style>
.section-hdr { font-size:1.1rem;font-weight:700;margin:0 0 4px;padding-bottom:6px; }
.section-inuse  { color:#ef9a9a;border-bottom:3px solid #E53935; }
.section-notuse { color:#fff176;border-bottom:3px solid #FB8C00; }
.section-divider { border:none;border-top:1px solid #1e2d3d;margin:36px 0; }
.stat-card { background:#1a1f2e;border-radius:10px;padding:14px 18px;
             border:1px solid #2a3040;text-align:center; }
.stat-val { font-size:1.8rem;font-weight:700;color:#fff; }
.stat-lbl { font-size:.72rem;color:#78909c;text-transform:uppercase;
            letter-spacing:.05em; }
.error-banner { background:#2e1a1a;border:1px solid #e53935;border-radius:8px;
                padding:10px 18px;color:#ef9a9a;font-size:.88rem;margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    for _k, _v in [("t1_cves", []), ("t1_loaded", False), ("t1_errors", [])]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### CVE Risk Settings")
        st.markdown("---")
        if st.button("🔄 Refresh CVE data", use_container_width=True, key="cve_refresh"):
            st.session_state.t1_cves   = []
            st.session_state.t1_loaded = False
            st.session_state.t1_errors = []
            st.rerun()
        st.markdown(
            f"<small style='color:#546e7a'>Timeout {CVE_API_TIMEOUT}s · "
            f"EPSS ≥ {CVE_EPSS_THRESHOLD*100:.0f}% · Top {CVE_TOP_N}<br>"
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</small>",
            unsafe_allow_html=True,
        )

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin-bottom:20px">
  <h1 style="color:#fff;font-size:1.8rem;font-weight:700;margin:0 0 4px">
    📊 CVE Risk Overview
  </h1>
  <p style="color:#78909c;font-size:.9rem;margin:0">
    Top CVEs with EPSS &gt; 50% — split by runtime exposure.
    Sourced from Sysdig analytics API (NVD CVSS v3 severity).
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Data fetch ────────────────────────────────────────────────────────────
    if not st.session_state.t1_loaded:
        with st.status("📡 Loading vulnerability data…", expanded=True) as _st:
            try:
                _cves, _errs = _load_cves_with_progress(api_base, api_token, _st)
                st.session_state.t1_cves   = _cves
                st.session_state.t1_errors = _errs
                st.session_state.t1_loaded = True
                if _errs:
                    _st.update(label=f"⚠️ {_errs[0]}", state="error")
                else:
                    _st.update(label=f"✅ Loaded {len(_cves)} CVE(s)",
                               state="complete", expanded=False)
            except Exception as _exc:
                st.session_state.t1_errors = [str(_exc)]
                st.session_state.t1_loaded = True
                _st.update(label=f"❌ {_exc}", state="error")

    for err in st.session_state.t1_errors:
        st.markdown(f'<div class="error-banner">⚠️ {err}</div>', unsafe_allow_html=True)

    all_cves = st.session_state.t1_cves
    if not all_cves:
        return

    in_use_cves  = [c for c in all_cves if c.get("inUse")]
    not_use_cves = [c for c in all_cves if not c.get("inUse")]

    # ── Overall summary ───────────────────────────────────────────────────────
    st.markdown("### Overall Summary")
    st.caption(
        "ℹ️ Severity sourced from Sysdig analytics API (NVD CVSS v3). "
        "May differ from Sysdig Vulnerability Findings page (vendor-adjusted ratings)."
    )
    ov1, ov2, ov3, ov4, ov5, ov6 = st.columns(6)
    ov1.metric("Total CVEs",  len(all_cves),                                    "EPSS > 50%")
    ov2.metric("In Use",      len(in_use_cves),                                 "runtime exposure")
    ov3.metric("Not In Use",  len(not_use_cves),                                "not actively running")
    ov4.metric("Exploitable", sum(1 for c in all_cves if c.get("exploitable")), "known exploits")
    ov5.metric("CISA KEV",    sum(1 for c in all_cves if c.get("hasCisaKev")),  "actively exploited")
    ov6.metric("Has Fix",     sum(1 for c in all_cves if c.get("fixAvailable")))

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    _cve_render_section(in_use_cves,  "🔴 In Use — Fix Now",    "section-inuse")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    _cve_render_section(not_use_cves, "🟡 Not In Use — Monitor", "section-notuse")
