"""
Engineering Fix View page for Sysdig Report Studio.

Live API fetch via SysQL query (same mechanism as Report Studio) with an
optional CSV fallback. Produces both an engineering action list and an
executive-level narrative analysis.
"""
from __future__ import annotations

import json
import urllib.parse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime

from config import get_sysdig_host

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0bec5", size=12),
    margin=dict(t=40, b=20, l=20, r=20),
)

EXPECTED_COLS = {
    "clusterName", "findings", "imageReference",
    "imageRegistry", "imageRepository", "imageTag",
    "namespaceName", "resourceName",
}  # used by _eng_load_from_records

SYSQL_QUERY_TEMPLATE = """\
MATCH Vulnerability AS vuln AFFECTS RuntimeResource AS resource OVER RuntimeMetadata AS metadata
MATCH metadata RUNS_IMAGE Image AS image
WHERE vuln.name = '{cve}' AND vuln.hasFix = true AND vuln.inUse != NULL AND vuln.hasExploit = true AND vuln.acceptedRisk = false
RETURN resource.clusterName AS clusterName, resource.namespaceName AS namespaceName, resource.name AS resourceName, image.imageReference AS imageReference, image.registry AS imageRegistry, image.repository AS imageRepository, image.tag AS imageTag, count(DISTINCT vuln.globalId) AS findings
ORDER BY findings DESC
LIMIT 10000;"""


# ---------------------------------------------------------------------------
# API fetch (mirrors fetch_sysql_data in report_studio.py)
# ---------------------------------------------------------------------------

def _fetch_sysql(region: str, api_token: str, query: str) -> tuple[list[dict] | None, str | None]:
    """Execute a SysQL query against the Sysdig API."""
    host = get_sysdig_host(region)
    base_url = f"https://{host}/api/sysql/v2/query"
    encoded_query = urllib.parse.quote(query)
    url = f"{base_url}?q={encoded_query}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "items" in result:
            items = result["items"]
            if "entities" in result and items:
                allowed_fields = list(result["entities"].keys())
                items = [
                    {k: item.get(k) for k in allowed_fields if k in item}
                    for item in items
                ]
            return items, None
        else:
            return None, "Unexpected API response format (no 'items' key)"

    except requests.exceptions.HTTPError as e:
        return None, f"API Error: {e.response.status_code} - {e.response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Connection error: Could not reach Sysdig API"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except json.JSONDecodeError:
        return None, "Invalid JSON response from API"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _t2_hbar(items: dict, x_label: str, color: str) -> go.Figure:
    df = pd.DataFrame(sorted(items.items(), key=lambda x: x[1]),
                      columns=["Label", "Value"])
    fig = go.Figure(go.Bar(
        x=df["Value"], y=df["Label"], orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        text=df["Value"], textposition="outside",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(200, len(items) * 38 + 60),
                      xaxis=dict(title=x_label, gridcolor="#1e2d3d"),
                      yaxis=dict(showgrid=False))
    return fig


def _eng_load_from_records(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from SysQL API records, same shape as CSV load."""
    df = pd.DataFrame(records)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"API response is missing columns: {', '.join(sorted(missing))}")
    df["findings"] = pd.to_numeric(df["findings"], errors="coerce").fillna(0).astype(int)
    df["imageLabel"] = df["imageRepository"].str.split("/").str[-1] + ":" + df["imageTag"]
    return df


def _eng_image_summary(df) -> pd.DataFrame:
    agg = (
        df.groupby(["imageRegistry", "imageRepository", "imageTag",
                    "imageReference", "imageLabel"])
        .agg(workloads      =("resourceName",  "nunique"),
             clusters       =("clusterName",   "nunique"),
             namespaces     =("namespaceName", "nunique"),
             total_findings =("findings",      "sum"),
             cluster_list   =("clusterName",   lambda x: ", ".join(sorted(x.unique()))),
             ns_list        =("namespaceName", lambda x: ", ".join(sorted(x.unique()))))
        .reset_index()
        .sort_values("total_findings", ascending=False)
        .reset_index(drop=True)
    )
    agg.insert(0, "Priority", range(1, len(agg) + 1))
    return agg


def _eng_repo_summary(df) -> pd.DataFrame:
    return (
        df.groupby("imageRepository")
        .agg(unique_tags    =("imageTag",      "nunique"),
             workloads      =("resourceName",  "nunique"),
             clusters       =("clusterName",   "nunique"),
             total_findings =("findings",      "sum"),
             tags           =("imageTag",      lambda x: ", ".join(sorted(x.unique()))))
        .reset_index()
        .sort_values("total_findings", ascending=False)
        .reset_index(drop=True)
    )


def _eng_top_images_bar(img_df, n: int = 25) -> go.Figure:
    top = img_df.head(n).sort_values("total_findings")
    fig = go.Figure(go.Bar(
        x=top["total_findings"], y=top["imageLabel"], orientation="h",
        marker=dict(color="#E53935", line=dict(width=0)),
        text=top["total_findings"], textposition="outside",
        customdata=top[["workloads", "clusters", "imageReference"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Total findings: %{x}<br>"
            "Workloads: %{customdata[0]}<br>Clusters: %{customdata[1]}<br>"
            "<i>%{customdata[2]}</i><extra></extra>"
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(320, len(top) * 30 + 60),
                      xaxis=dict(title="Total Findings", gridcolor="#1e2d3d"),
                      yaxis=dict(showgrid=False, tickfont=dict(size=10)))
    return fig


def _eng_cluster_bar(df) -> go.Figure:
    counts = (
        df.groupby("clusterName")["resourceName"].nunique()
        .reset_index().rename(columns={"resourceName": "workloads"})
        .sort_values("workloads")
    )
    fig = go.Figure(go.Bar(
        x=counts["workloads"], y=counts["clusterName"], orientation="h",
        marker=dict(color="#9B3FBF", line=dict(width=0)),
        text=counts["workloads"], textposition="outside",
        hovertemplate="%{y}: %{x} affected workloads<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(260, len(counts) * 32 + 60),
                      xaxis=dict(title="Affected Workloads", gridcolor="#1e2d3d"),
                      yaxis=dict(showgrid=False))
    return fig


def _eng_registry_donut(df) -> go.Figure:
    counts = df.groupby("imageRegistry")["imageReference"].nunique().reset_index()
    counts.columns = ["registry", "images"]
    palette = ["#00BFA5", "#E53935", "#9B3FBF", "#FB8C00",
               "#1E88E5", "#00C853", "#FF6F00", "#7C4DFF"]
    fig = go.Figure(go.Pie(
        labels=counts["registry"], values=counts["images"],
        marker=dict(colors=[palette[i % len(palette)] for i in range(len(counts))],
                    line=dict(width=2, color="#12161f")),
        hole=0.55, textinfo="label+value", textfont=dict(size=10),
        hovertemplate="%{label}<br>%{value} unique images (%{percent})<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                      title=dict(text="Unique vulnerable images by registry",
                                 font=dict(size=12, color="#90a4ae"), x=0.5))
    return fig


def _eng_cluster_image_heatmap(df, top_n: int = 30) -> go.Figure:
    top_labels = (
        df.groupby("imageLabel")["findings"].sum()
        .nlargest(top_n).index.tolist()
    )
    sub   = df[df["imageLabel"].isin(top_labels)]
    pivot = (
        sub.groupby(["imageLabel", "clusterName"])["resourceName"].nunique()
        .reset_index()
        .pivot(index="imageLabel", columns="clusterName", values="resourceName")
        .fillna(0).astype(int)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    z     = pivot.values.tolist()
    text  = [[str(int(v)) if v > 0 else "" for v in row] for row in z]
    fig   = go.Figure(go.Heatmap(
        z=z, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#12161f"], [0.3, "#1a2744"], [0.7, "#9B3FBF"], [1, "#E53935"]],
        showscale=True,
        colorbar=dict(title=dict(text="Workloads", font=dict(color="#90a4ae")),
                      tickfont=dict(color="#90a4ae")),
        hovertemplate=(
            "<b>%{y}</b><br>Cluster: %{x}<br>Workloads: %{z}<extra></extra>"
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(400, len(pivot) * 22 + 120),
                      xaxis=dict(title="Cluster", tickangle=-30,
                                 gridcolor="#1e2d3d", tickfont=dict(size=10)),
                      yaxis=dict(title="Image", autorange="reversed",
                                 gridcolor="#1e2d3d", tickfont=dict(size=10)))
    return fig


def _eng_findings_hist(df) -> go.Figure:
    counts = df["findings"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str), y=counts.values,
        marker=dict(color="#FB8C00", line=dict(width=0)),
        text=counts.values, textposition="outside",
        hovertemplate="Findings per workload: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      xaxis=dict(title="Findings per Workload", gridcolor="#1e2d3d"),
                      yaxis=dict(title="Workload Count", gridcolor="#1e2d3d"))
    return fig


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _exec_summary(df: pd.DataFrame, cve: str, img_df: pd.DataFrame):
    """Render an executive-level narrative analysis of the CVE exposure."""
    total_workloads  = df["resourceName"].nunique()
    total_images     = len(img_df)
    total_clusters   = df["clusterName"].nunique()
    total_namespaces = df["namespaceName"].nunique()
    total_findings   = int(df["findings"].sum())

    # Risk classification
    if total_clusters >= 5 or total_workloads >= 50:
        risk_label, risk_color, risk_bg = "CRITICAL", "#ef5350", "#2d1515"
    elif total_clusters >= 3 or total_workloads >= 20:
        risk_label, risk_color, risk_bg = "HIGH", "#ffa726", "#2d1e0f"
    elif total_workloads >= 5:
        risk_label, risk_color, risk_bg = "MEDIUM", "#ffeb3b", "#2a270d"
    else:
        risk_label, risk_color, risk_bg = "LOW", "#66bb6a", "#0d2010"

    # Fix efficiency — patching top 3 images removes what % of workload exposure?
    top3 = img_df.head(3)
    top3_workloads = int(top3["workloads"].sum())
    top3_pct = (top3_workloads / total_workloads * 100) if total_workloads > 0 else 0

    # Most exposed cluster
    cluster_wl = (
        df.groupby("clusterName")["resourceName"].nunique()
        .sort_values(ascending=False)
    )
    top_cluster      = cluster_wl.index[0] if len(cluster_wl) else "N/A"
    top_cluster_wl   = int(cluster_wl.iloc[0]) if len(cluster_wl) else 0

    # Namespace concentration — top namespace share
    ns_wl = df.groupby("namespaceName")["resourceName"].nunique().sort_values(ascending=False)
    top_ns     = ns_wl.index[0] if len(ns_wl) else "N/A"
    top_ns_pct = (int(ns_wl.iloc[0]) / total_workloads * 100) if total_workloads > 0 else 0

    # ── Risk banner ────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:{risk_bg};border-radius:12px;padding:20px 24px;
            border-left:5px solid {risk_color};margin-bottom:8px">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
    <span style="background:{risk_color};color:#fff;font-weight:700;font-size:.78rem;
                 padding:4px 12px;border-radius:4px;letter-spacing:.1em">{risk_label} RISK</span>
    <span style="color:#b0bec5;font-size:.9rem;font-weight:600">{cve} &nbsp;·&nbsp; Runtime Exposure Analysis</span>
  </div>
  <div style="color:#cfd8dc;font-size:.92rem;line-height:1.7">
    <b>{cve}</b> is an actively exploitable vulnerability with a confirmed fix available.
    Runtime scanning has identified <b>{total_workloads:,} affected workload{'s' if total_workloads != 1 else ''}</b>
    across <b>{total_clusters} cluster{'s' if total_clusters != 1 else ''}</b> and
    <b>{total_namespaces} namespace{'s' if total_namespaces != 1 else ''}</b>,
    running <b>{total_images} unique vulnerable image{'s' if total_images != 1 else ''}</b>.
    All findings are in active use, have a known exploit, and have no accepted risk exemption —
    making this a <b style="color:{risk_color}">priority remediation item</b>.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Key insight cards ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
<div style="background:#1a1f2e;border-radius:10px;padding:16px 20px;height:100%">
  <div style="color:#90a4ae;font-size:.75rem;font-weight:600;letter-spacing:.08em;
              margin-bottom:6px">⚡ QUICK WIN</div>
  <div style="color:#fff;font-size:1rem;line-height:1.5">
    Rebuilding the <b>top {min(3, total_images)} image{'s' if min(3, total_images) != 1 else ''}</b>
    eliminates exposure in
    <b style="color:#00C853;font-size:1.15rem">{top3_pct:.0f}%</b>
    of affected workloads ({top3_workloads:,} of {total_workloads:,}).
  </div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div style="background:#1a1f2e;border-radius:10px;padding:16px 20px;height:100%">
  <div style="color:#90a4ae;font-size:.75rem;font-weight:600;letter-spacing:.08em;
              margin-bottom:6px">🌐 HIGHEST EXPOSURE CLUSTER</div>
  <div style="color:#fff;font-size:1rem;line-height:1.5">
    <b style="color:#ffa726">{top_cluster}</b> has the most impact with
    <b>{top_cluster_wl:,} affected workload{'s' if top_cluster_wl != 1 else ''}</b>.
    Prioritise patching here first.
  </div>
</div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
<div style="background:#1a1f2e;border-radius:10px;padding:16px 20px;height:100%">
  <div style="color:#90a4ae;font-size:.75rem;font-weight:600;letter-spacing:.08em;
              margin-bottom:6px">📍 CONCENTRATION HOTSPOT</div>
  <div style="color:#fff;font-size:1rem;line-height:1.5">
    <b style="color:#7b61ff">{top_ns}</b> namespace accounts for
    <b>{top_ns_pct:.0f}%</b> of affected workloads —
    a focused patch here delivers outsized risk reduction.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # ── Priority fix table ─────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.95rem;font-weight:700;color:#fff;margin:16px 0 6px'>"
        "🎯 Recommended Fix Priority</div>"
        "<div style='color:#78909c;font-size:.82rem;margin-bottom:10px'>"
        "Ordered by blast radius. Fixing the top rows gives the fastest risk reduction.</div>",
        unsafe_allow_html=True,
    )

    top10 = img_df.head(10).copy()
    top10["Cumulative Workload Coverage"] = (
        top10["workloads"].cumsum() / total_workloads * 100
    ).map("{:.0f}%".format)

    fix_tbl = top10[["Priority", "imageLabel", "workloads", "clusters",
                      "total_findings", "Cumulative Workload Coverage"]].rename(columns={
        "imageLabel":                   "Image (name:tag)",
        "workloads":                    "Workloads",
        "clusters":                     "Clusters",
        "total_findings":               "Findings",
        "Cumulative Workload Coverage": "Cumulative Coverage",
    })

    st.dataframe(fix_tbl, use_container_width=True, hide_index=True)

    # ── Recommended actions ────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.95rem;font-weight:700;color:#fff;margin:16px 0 6px'>"
        "📋 Recommended Actions</div>",
        unsafe_allow_html=True,
    )

    action_items = [
        f"**Immediate (24–48 h):** Rebuild and redeploy the top {min(3, total_images)} image(s) listed above — "
        f"this resolves ~{top3_pct:.0f}% of the exposure with minimal blast radius.",
        f"**Short-term (this sprint):** Patch all {total_images} unique vulnerable image(s) and roll out "
        f"to the {total_workloads:,} affected workload(s). Start with cluster **{top_cluster}** "
        f"({top_cluster_wl} workloads).",
        f"**Process:** Enforce image signing and admission control to prevent vulnerable images from "
        f"being redeployed. Add `{cve}` to your vulnerability exception review backlog.",
        f"**Verification:** Re-run this query after patching to confirm zero remaining findings.",
    ]
    for item in action_items:
        st.markdown(f"- {item}")


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render_page(api_token: str = "", region: str = "US East"):
    """Engineering Fix View — SysQL API fetch or CSV fallback."""

    st.markdown("""
<style>
.section-divider { border:none;border-top:1px solid #1e2d3d;margin:36px 0; }
.stCodeBlock { font-size: .8rem !important; }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-bottom:22px">
  <h1 style="color:#fff;font-size:1.8rem;font-weight:700;margin:0 0 6px">
    🔧 Engineering Fix View
  </h1>
  <p style="color:#78909c;font-size:.87rem;margin:0">
    Enter a CVE to fetch live runtime exposure data directly from Sysdig.
    Produces an executive summary and a per-image action list.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── CVE input + query display ──────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        cve = st.text_input(
            "CVE Identifier",
            value="",
            placeholder="CVE-2023-44487",
            help="Enter the CVE ID to query. Leave blank to use the placeholder query.",
            key="eng_cve_input",
        )
        cve_for_query = cve.strip() or "CVE-2023-44487"

        query = SYSQL_QUERY_TEMPLATE.format(cve=cve_for_query)

        fetch_disabled = not api_token
        fetch_clicked = st.button(
            "🔍 Fetch from Sysdig API",
            disabled=fetch_disabled,
            key="eng_fetch_btn",
            help="Requires an API token in the Global Config sidebar." if fetch_disabled else f"Fetch {cve_for_query} data from {region}",
            use_container_width=True,
        )
        if fetch_disabled:
            st.caption("⚠️ Enter an API token in the sidebar to enable live fetch.")

    with right_col:
        st.markdown(
            "<div style='color:#90a4ae;font-size:.78rem;font-weight:600;"
            "letter-spacing:.06em;margin-bottom:4px'>SYSQL QUERY</div>",
            unsafe_allow_html=True,
        )
        st.code(query, language="sql")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Resolve data: API fetch → CSV fallback → nothing ──────────────────────
    df_eng = None
    data_source = None

    # Check for a previously fetched result in session state
    cached_key = f"eng_data_{cve_for_query}"

    if fetch_clicked:
        with st.spinner(f"Querying Sysdig for {cve_for_query}…"):
            records, err = _fetch_sysql(region, api_token, query)
        if err:
            st.error(f"API fetch failed: {err}")
        elif not records:
            st.warning(f"No findings returned for **{cve_for_query}**. "
                       "The CVE may not be present in your runtime environment, "
                       "or all findings may be exempted.")
        else:
            try:
                df_eng = _eng_load_from_records(records)
                st.session_state[cached_key] = df_eng
                st.success(f"Fetched **{len(df_eng):,}** rows for `{cve_for_query}` from Sysdig ({region})")
                data_source = f"Sysdig API · {cve_for_query} · {region}"
            except Exception as e:
                st.error(f"Could not parse API response: {e}")

    elif cached_key in st.session_state:
        df_eng = st.session_state[cached_key]
        data_source = f"Sysdig API · {cve_for_query} · {region} (cached)"

    # ── Render analysis if we have data ───────────────────────────────────────
    if df_eng is not None:
        img_df  = _eng_image_summary(df_eng)
        repo_df = _eng_repo_summary(df_eng)
        ts = datetime.now().strftime("%Y%m%d_%H%M")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Executive Summary ──────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:12px'>"
            "📊 Executive Summary</div>",
            unsafe_allow_html=True,
        )
        _exec_summary(df_eng, cve_for_query, img_df)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Impact Summary metrics ─────────────────────────────────────────────
        st.markdown("### Impact Summary")
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("Workloads Affected",  df_eng["resourceName"].nunique())
        e2.metric("Unique Images",        len(img_df), "to rebuild/patch")
        e3.metric("Image Repositories",   df_eng["imageRepository"].nunique())
        e4.metric("Clusters",             df_eng["clusterName"].nunique())
        e5.metric("Namespaces",           df_eng["namespaceName"].nunique())
        e6.metric("Total Findings",       int(df_eng["findings"].sum()))

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:4px'>"
            "🎯 What to Fix — Image Action List</div>"
            "<div style='color:#78909c;font-size:.82rem;margin-bottom:14px'>"
            "Each row is one image that needs to be rebuilt/patched. "
            "Priority is ranked by total findings.</div>",
            unsafe_allow_html=True,
        )

        with st.expander("📂 Group by Repository", expanded=False):
            st.dataframe(repo_df.rename(columns={
                "imageRepository": "Repository",
                "unique_tags":     "Vulnerable Tags",
                "workloads":       "Workloads",
                "clusters":        "Clusters",
                "total_findings":  "Total Findings",
                "tags":            "Tag Versions",
            }), use_container_width=True, hide_index=True)

        action_cols = {
            "Priority": "Priority", "imageLabel": "Image (name:tag)",
            "imageRegistry": "Registry", "workloads": "Workloads",
            "clusters": "Clusters", "namespaces": "Namespaces",
            "total_findings": "Total Findings", "cluster_list": "Cluster Names",
            "imageReference": "Full Image Reference",
        }
        action_df = img_df[list(action_cols.keys())].rename(columns=action_cols)

        def _sty_priority(v):
            if v <= 3:  return "color:#ef5350;font-weight:700"
            if v <= 10: return "color:#ffa726;font-weight:700"
            return "color:#90a4ae"

        st.dataframe(action_df.style.applymap(_sty_priority, subset=["Priority"]),
                     use_container_width=True, hide_index=True,
                     height=min(600, 36 * len(action_df) + 60))

        dl1, _ = st.columns([1, 5])
        with dl1:
            st.download_button("⬇️ Export action list",
                               data=action_df.to_csv(index=False),
                               file_name=f"fix_action_list_{ts}.csv",
                               mime="text/csv", key="dl_eng_action")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:14px'>"
            "📊 Visual Breakdown</div>", unsafe_allow_html=True)

        vt1, vt2, vt3 = st.tabs(["🖼️ Top Images", "🌐 Cluster Spread", "📦 Registry & Findings"])

        with vt1:
            st.plotly_chart(_eng_top_images_bar(img_df), use_container_width=True,
                            config={"displayModeBar": False})
        with vt2:
            vc1, vc2 = st.columns(2)
            with vc1:
                st.plotly_chart(_eng_cluster_bar(df_eng), use_container_width=True,
                                config={"displayModeBar": False})
            with vc2:
                ns_counts = (
                    df_eng.groupby("namespaceName")["resourceName"].nunique()
                    .reset_index().rename(columns={"resourceName": "workloads"})
                    .set_index("namespaceName")["workloads"].to_dict()
                )
                st.plotly_chart(_t2_hbar(ns_counts, "Workloads", "#1E88E5"),
                                use_container_width=True, config={"displayModeBar": False})
        with vt3:
            vc1, vc2 = st.columns(2)
            with vc1:
                st.plotly_chart(_eng_registry_donut(df_eng), use_container_width=True,
                                config={"displayModeBar": False})
            with vc2:
                st.plotly_chart(_eng_findings_hist(df_eng), use_container_width=True,
                                config={"displayModeBar": False})

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:4px'>"
            "🟥 Image × Cluster Heatmap</div>"
            "<div style='color:#78909c;font-size:.82rem;margin-bottom:14px'>"
            "Which images are running in which clusters. Each cell = workload count.</div>",
            unsafe_allow_html=True)
        st.plotly_chart(_eng_cluster_image_heatmap(df_eng), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:4px'>"
            "🔍 Per-Image Workload Detail</div>"
            "<div style='color:#78909c;font-size:.82rem;margin-bottom:14px'>"
            "Select an image to see every workload that needs redeployment.</div>",
            unsafe_allow_html=True)

        sel_image = st.selectbox("Select image", options=img_df["imageLabel"].tolist(),
                                 key="eng_img_sel")
        if sel_image:
            sub_df  = df_eng[df_eng["imageLabel"] == sel_image].copy()
            sel_ref = sub_df["imageReference"].iloc[0]
            st.markdown(
                f"<div style='background:#1a1f2e;border-radius:8px;padding:12px 18px;"
                f"border-left:4px solid #E53935;margin-bottom:14px'>"
                f"<div style='color:#90a4ae;font-size:.78rem;margin-bottom:2px'>Full image reference</div>"
                f"<div style='color:#fff;font-family:monospace;font-size:.9rem'>{sel_ref}</div>"
                f"</div>", unsafe_allow_html=True)
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Workloads",      sub_df["resourceName"].nunique())
            sm2.metric("Clusters",       sub_df["clusterName"].nunique())
            sm3.metric("Namespaces",     sub_df["namespaceName"].nunique())
            sm4.metric("Total Findings", int(sub_df["findings"].sum()))
            detail_df = (
                sub_df[["clusterName", "namespaceName", "resourceName", "findings"]]
                .drop_duplicates()
                .sort_values(["clusterName", "namespaceName", "resourceName"])
                .rename(columns={"clusterName": "Cluster", "namespaceName": "Namespace",
                                 "resourceName": "Workload", "findings": "Findings"})
                .reset_index(drop=True)
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True,
                         height=min(500, 36 * len(detail_df) + 60))
            st.download_button(
                f"⬇️ Export workloads for {sel_image}",
                data=detail_df.to_csv(index=False),
                file_name=f"workloads_{sel_image.replace(':', '_').replace('/', '_')}_{ts}.csv",
                mime="text/csv", key="dl_eng_detail")

    else:
        st.markdown(
            "<div style='background:#12161f;border-radius:10px;padding:20px 24px;"
            "border:1px dashed #37474f;color:#78909c;font-size:.87rem;line-height:1.7'>"
            "Enter a CVE above and click <b>Fetch from Sysdig API</b> to load live runtime exposure data."
            "</div>",
            unsafe_allow_html=True,
        )
