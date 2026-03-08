"""
Posture Analytics page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
Chart functions return go.Figure objects; render_page() handles all st.plotly_chart() calls.
"""
import gzip
import io
import shutil
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import SYSDIG_REGIONS

POSTURE_REPORT_NAME = "[PG] Posture"
REPORTING_API = "/api/platform/reporting/v1"


# ── API helpers (same pattern as Runtime Vulnerabilities page) ────────────────

def _auto_detect_base_url(token: str) -> Optional[str]:
    """Cycle through all known Sysdig regions and return the first base_url where the token is valid."""
    cache_key = f"_posture_base_url_{token[:8]}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for host in SYSDIG_REGIONS.values():
        base_url = f"https://{host}"
        try:
            r = requests.get(f"{base_url}/api/platform/reporting/v1/reports", headers=headers, timeout=6)
            if r.status_code == 200:
                st.session_state[cache_key] = base_url
                return base_url
        except Exception:
            continue
    return None


def _api_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _find_report_id(base_url: str, token: str) -> Optional[tuple]:
    """Return (reportId, reportName) for the Posture report, or None."""
    r = requests.get(
        f"{base_url}{REPORTING_API}/reports",
        headers=_api_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    reports_list = data if isinstance(data, list) else data.get("reports", [])
    for report in reports_list:
        name = report.get("name", "") or report.get("reportName", "")
        if POSTURE_REPORT_NAME.lower() in name.lower():
            rid = report.get("id") or report.get("reportId")
            return int(rid), name
    return None


def _trigger_job(base_url: str, token: str, report_id: int, report_name: str) -> int:
    now = int(time.time())
    payload = {
        "reportId":         report_id,
        "isReportTemplate": False,
        "reportFormat":     "csv",
        "jobName":          report_name,
        "fileName":         report_name.replace(" ", "_").replace("[", "").replace("]", ""),
        "jobType":          "ON_DEMAND",
        "zones":            [],
        "timeFrame":        {"from": now - 86400, "to": now},
        "scheduledOn":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = requests.post(
        f"{base_url}{REPORTING_API}/jobs",
        headers=_api_headers(token),
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["id"]


def _poll_job(base_url: str, token: str, job_id: int, status_ph) -> Optional[str]:
    for attempt in range(40):
        time.sleep(4)
        r = requests.get(
            f"{base_url}{REPORTING_API}/jobs/{job_id}",
            headers=_api_headers(token),
            timeout=30,
        )
        r.raise_for_status()
        job = r.json()
        status = job.get("status", "UNKNOWN")
        status_ph.caption(f"Job status: **{status}** (check {attempt + 1}/40)…")
        if status == "COMPLETED":
            return job.get("fullFilePath")
        if status in ("FAILED", "CANCELLED"):
            return None
    return None


def fetch_from_api(token: str, base_url: str = None) -> tuple:
    """
    Fetch the [PG] Posture report via Sysdig Reporting API on-demand.
    Returns (df_full, df_fail, report_name) or raises on error.
    """
    if not base_url:
        base_url = _auto_detect_base_url(token)
    if not base_url:
        raise RuntimeError("Could not detect your Sysdig region. Check that your token is valid.")

    result = _find_report_id(base_url, token)
    if result is None:
        raise RuntimeError(f"Report '{POSTURE_REPORT_NAME}' not found in Reports Manager for this account.")
    report_id, report_name = result

    job_id = _trigger_job(base_url, token, report_id, report_name)
    status_ph = st.empty()
    status_ph.caption(f"Job created (id={job_id}). Waiting for completion…")

    file_url = _poll_job(base_url, token, job_id, status_ph)
    status_ph.empty()

    if not file_url:
        raise RuntimeError("Job did not complete successfully within the timeout.")

    dl = requests.get(
        file_url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=120,
    )
    dl.raise_for_status()

    buf = io.BytesIO()
    shutil.copyfileobj(dl.raw, buf)
    buf.seek(0)
    data = buf.read()

    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as f:
        df_full = pd.read_csv(f)

    df_fail = df_full[df_full['Result'] == 'Fail'].copy()
    return df_full, df_fail, report_name


def create_executive_charts(df: pd.DataFrame, group_by: str = 'Zones'):
    """
    Create executive-level dashboard charts for posture analytics.

    Generates visualizations showing who contributes most to compliance
    failures, designed for executive stakeholders to identify priority
    areas for remediation.

    Args:
        df: DataFrame containing failing control records
        group_by: Column to group data by ('Zones' for owners, 'Account Id' for accounts)

    Returns:
        tuple: (pie_chart, bar_chart, total_failures, unique_owners,
               unique_accounts, top_owners_df, all_owner_stats_df)
    """
    total_failures = len(df)
    unique_owners = df[group_by].nunique()
    unique_accounts = df['Account Id'].nunique()

    # Aggregate by owner - use different secondary column based on grouping
    secondary_col = 'Account Id' if group_by == 'Zones' else 'Zones'
    owner_stats = df.groupby(group_by).agg({
        'Control ID': 'count',
        secondary_col: lambda x: list(x.unique()),
        'Control Name': lambda x: x.nunique()
    }).reset_index()
    owner_stats.columns = ['Owner', 'Total Failures', 'Related Items', 'Unique Controls']
    owner_stats['Percentage'] = (owner_stats['Total Failures'] / total_failures * 100).round(1)
    owner_stats = owner_stats.sort_values('Total Failures', ascending=False)

    # Top contributors
    top_n = 10
    top_owners = owner_stats.head(top_n).copy()
    others_count = owner_stats.iloc[top_n:]['Total Failures'].sum() if len(owner_stats) > top_n else 0
    others_pct = round(others_count / total_failures * 100, 1)
    top_total_pct = top_owners['Percentage'].sum()

    # Pie chart - convert to string for Account IDs
    pie_labels = [f"{str(o)[:20]}..." if len(str(o)) > 20 else str(o) for o in top_owners['Owner']]
    pie_values = list(top_owners['Total Failures'])
    pie_text = [f"{p}%" for p in top_owners['Percentage']]

    if others_count > 0:
        pie_labels.append(f'Others ({len(owner_stats) - top_n} people)')
        pie_values.append(others_count)
        pie_text.append(f"{others_pct}%")

    colors_pie = ['#e74c3c', '#c0392b', '#e67e22', '#d35400', '#f39c12',
                  '#f1c40f', '#27ae60', '#2ecc71', '#3498db', '#2980b9', '#95a5a6']

    fig_pie = go.Figure(go.Pie(
        labels=pie_labels,
        values=pie_values,
        text=pie_text,
        textinfo='label+text',
        textposition='outside',
        marker_colors=colors_pie[:len(pie_labels)],
        hole=0.4,
        sort=False
    ))
    fig_pie.add_annotation(
        text=f"<b>{total_failures:,}</b><br>Total",
        x=0.5, y=0.5, font_size=16, showarrow=False
    )
    fig_pie.update_layout(
        title=dict(text='<b>Who is Contributing to Compliance Failures?</b>', x=0.5, font=dict(size=16)),
        height=500,
        margin=dict(t=60, b=40, l=40, r=40),
        showlegend=False
    )

    # Horizontal bar chart
    top_owners_sorted = top_owners.sort_values('Total Failures', ascending=True).reset_index(drop=True)
    bar_labels = [(str(o)[:25] + '...' if len(str(o)) > 25 else str(o)) for o in top_owners_sorted['Owner'].tolist()]
    bar_values = top_owners_sorted['Total Failures'].tolist()
    bar_pcts = top_owners_sorted['Percentage'].tolist()
    bar_related = top_owners_sorted['Related Items'].tolist()
    related_label = 'Accounts' if group_by == 'Zones' else 'Zones'

    fig_bar = go.Figure(go.Bar(
        x=bar_values,
        y=bar_labels,
        orientation='h',
        marker_color='#e74c3c',
        text=[f"{v:,} ({p}%)" for v, p in zip(bar_values, bar_pcts)],
        textposition='inside',
        textfont=dict(color='white', size=12),
        insidetextanchor='end',
        hovertext=[f"{o}<br>Failures: {v:,}<br>% of Total: {p}%<br>{related_label}: {', '.join(map(str, a[:3]))}"
                   for o, v, p, a in zip(bar_labels, bar_values, bar_pcts, bar_related)],
        hoverinfo='text'
    ))
    fig_bar.update_layout(
        title=dict(text=f'<b>Top {top_n} Contributors = {top_total_pct:.0f}% of All Failures</b>', x=0.5, font=dict(size=16)),
        height=500,
        margin=dict(t=60, b=60, l=200, r=60),
        xaxis=dict(title='Total Failures', tickformat=',d', rangemode='tozero'),
        yaxis=dict(type='category'),
        plot_bgcolor='#fafafa'
    )

    return fig_pie, fig_bar, total_failures, unique_owners, unique_accounts, top_owners, owner_stats


def create_person_charts(df: pd.DataFrame, top_owners: pd.DataFrame, group_by: str = 'Zones'):
    """
    Create detailed breakdown charts for top contributors.

    For each of the top 5 contributors, generates a horizontal bar chart
    showing their most frequently failing controls by severity.

    Args:
        df: DataFrame containing failing control records
        top_owners: DataFrame with top contributing owners
        group_by: Column used for grouping ('Zones' or 'Account Id')

    Returns:
        list[tuple]: List of (owner_name, plotly_figure) pairs
    """
    person_charts = []
    top_5_owners = top_owners.sort_values('Total Failures', ascending=False).head(5)

    for _, row in top_5_owners.iterrows():
        owner = row['Owner']
        owner_df = df[df[group_by] == owner]

        all_controls = owner_df.groupby(['Control Name', 'Control Severity']).size().reset_index(name='Count')
        total_unique_controls = len(all_controls)

        # Severity rank: High outranks everything regardless of count
        sev_rank = {'High': 3, 'Medium': 2, 'Low': 1, 'Info': 0}
        all_controls['_sev_rank'] = all_controls['Control Severity'].map(sev_rank).fillna(0)

        # Pick top 8 prioritising severity first, then count within each tier
        top8 = all_controls.sort_values(['_sev_rank', 'Count'], ascending=[False, False]).head(8)

        # Re-sort for chart display: lowest rank at bottom, highest (High) at top
        controls = top8.sort_values(['_sev_rank', 'Count'], ascending=[True, True]).reset_index(drop=True)
        controls = controls.drop(columns='_sev_rank')

        severity_colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#3498db', 'Info': '#95a5a6'}

        ctrl_labels = [(c[:40] + '...' if len(c) > 40 else c) for c in controls['Control Name'].tolist()]
        ctrl_values = controls['Count'].tolist()
        ctrl_severities = controls['Control Severity'].tolist()
        ctrl_colors = [severity_colors.get(s, '#95a5a6') for s in ctrl_severities]

        fig_person = go.Figure(go.Bar(
            x=ctrl_values,
            y=ctrl_labels,
            orientation='h',
            marker_color=ctrl_colors,
            text=[f"{c} ({s})" for c, s in zip(ctrl_values, ctrl_severities)],
            textposition='inside',
            textfont=dict(color='white', size=11),
            insidetextanchor='end',
            hovertext=controls['Control Name'].tolist(),
            hoverinfo='text+x'
        ))

        related_str = ', '.join(map(str, row['Related Items'][:3]))
        if len(row['Related Items']) > 3:
            related_str += f" (+{len(row['Related Items'])-3} more)"
        related_label = 'Accounts' if group_by == 'Zones' else 'Zones'

        fig_person.update_layout(
            title=dict(
                text=f"<b>{str(owner)[:35]}</b><br>{row['Total Failures']:,} failures ({row['Percentage']}%) across {total_unique_controls} controls<br>{related_label}: {related_str}",
                x=0.5, font=dict(size=13)
            ),
            height=400,
            margin=dict(t=90, b=40, l=250, r=60),
            xaxis=dict(title='Total Failures', tickformat=',d', rangemode='tozero'),
            yaxis=dict(tickfont=dict(size=10)),
            plot_bgcolor='#fafafa'
        )
        person_charts.append((owner, fig_person))

    return person_charts


def create_security_charts(df: pd.DataFrame, group_by: str = 'Zones'):
    """
    Create security team dashboard charts for detailed drill-down.

    Generates three visualizations:
    1. Treemap: Hierarchical view of Owner > Severity > Control
    2. Heatmap: Owner vs Control failure matrix
    3. Stacked bar: Severity breakdown by owner

    Args:
        df: DataFrame containing failing control records
        group_by: Column used for grouping ('Zones' or 'Account Id')

    Returns:
        tuple: (treemap_figure, heatmap_figure, severity_bar_figure)
    """
    # Treemap for hierarchical drill-down
    treemap_data = df.groupby([group_by, 'Control Severity', 'Control Name']).size().reset_index(name='Count')

    fig_treemap = px.treemap(
        treemap_data,
        path=[group_by, 'Control Severity', 'Control Name'],
        values='Count',
        color='Control Severity',
        color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#3498db', 'Info': '#95a5a6'},
        title='<b>Security Posture Drill-Down</b><br><sup>Click to explore: Owner > Severity > Control</sup>'
    )
    fig_treemap.update_layout(height=700)
    fig_treemap.update_traces(textinfo='label+value')

    # Heatmap
    pivot = df.pivot_table(
        index=group_by,
        columns='Control Name',
        values='Resource ID',
        aggfunc='count',
        fill_value=0
    )

    top_owners = df[group_by].value_counts().head(20).index
    top_controls = df['Control Name'].value_counts().head(15).index

    pivot_filtered = pivot.loc[
        pivot.index.isin(top_owners),
        pivot.columns.isin(top_controls)
    ]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_filtered.values,
        x=[c[:40] + '...' if len(c) > 40 else c for c in pivot_filtered.columns],
        y=[str(i) for i in pivot_filtered.index],
        colorscale='Reds',
        text=pivot_filtered.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Owner: %{y}<br>Control: %{x}<br>Failures: %{z}<extra></extra>'
    ))
    fig_heatmap.update_layout(
        title='<b>Owner vs Control Failure Matrix</b>',
        xaxis_title='Control Name',
        yaxis_title='Owner',
        height=600,
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10), type='category')
    )

    # Severity breakdown
    owner_severity = df.groupby([group_by, 'Control Severity']).size().reset_index(name='Count')
    top_10_owners = df[group_by].value_counts().head(10).index.tolist()
    owner_severity_filtered = owner_severity[owner_severity[group_by].isin(top_10_owners)]

    fig_severity = go.Figure()
    for severity in ['High', 'Medium', 'Low', 'Info']:
        sev_data = owner_severity_filtered[owner_severity_filtered['Control Severity'] == severity]
        if not sev_data.empty:
            fig_severity.add_trace(go.Bar(
                name=severity,
                x=sev_data[group_by].astype(str),
                y=sev_data['Count'],
                marker_color={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#3498db', 'Info': '#95a5a6'}[severity],
                text=sev_data['Count'],
                textposition='inside'
            ))

    fig_severity.update_layout(
        barmode='stack',
        title='<b>Severity Breakdown by Owner (Top 10)</b>',
        height=500,
        xaxis=dict(tickangle=45, type='category'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig_treemap, fig_heatmap, fig_severity


def create_downloadable_reports(df: pd.DataFrame, owner_stats: pd.DataFrame, group_by: str = 'Zones'):
    """
    Create downloadable CSV reports for offline analysis.

    Generates two reports:
    1. Owner Summary: High-level stats per owner
    2. Actionable Report: Detailed breakdown by owner/account/control

    Args:
        df: DataFrame containing failing control records
        owner_stats: DataFrame with aggregated owner statistics
        group_by: Column used for grouping

    Returns:
        tuple: (owner_export_df, actionable_report_df)
    """
    # Owner summary export
    owner_export = owner_stats.copy()
    owner_export['Related Items'] = owner_export['Related Items'].apply(lambda x: ', '.join(map(str, x)))

    # Actionable report
    action_report = []
    for owner in df[group_by].unique():
        owner_df = df[df[group_by] == owner]
        accounts = owner_df.groupby(['Account Name', 'Account Id']).size().reset_index(name='Failures')

        for _, acc in accounts.iterrows():
            acc_df = owner_df[(owner_df['Account Name'] == acc['Account Name']) &
                             (owner_df['Account Id'] == acc['Account Id'])]

            controls = acc_df.groupby(['Control Name', 'Control Severity', 'Control ID']).agg({
                'Resource Name': ['count', lambda x: ', '.join(x.unique()[:3])]
            }).reset_index()
            controls.columns = ['Control Name', 'Severity', 'Control ID', 'Count', 'Sample Resources']

            for _, ctrl in controls.iterrows():
                action_report.append({
                    'Owner': owner,
                    'Account Name': acc['Account Name'],
                    'Account Id': acc['Account Id'],
                    'Control Name': ctrl['Control Name'],
                    'Control ID': ctrl['Control ID'],
                    'Severity': ctrl['Severity'],
                    'Failure Count': ctrl['Count'],
                    'Sample Resources': ctrl['Sample Resources']
                })

    action_df = pd.DataFrame(action_report)

    return owner_export, action_df


def render_page():
    """
    Render the Posture Analytics page for compliance report analysis.

    This page provides:
    - CSV file upload for posture compliance reports
    - Executive dashboard showing top contributors to failures
    - Security drill-down with treemap and heatmap views
    - Trend analysis when multiple reports are uploaded
    - Downloadable CSV reports for offline use
    """
    st.title("Sysdig Posture Report Analytics")

    group_by = st.selectbox(
        "Group failures by",
        options=['Zones', 'Account Id'],
        index=0,
        help="Group failures by Zones (owner) or by Account Id",
        key="posture_group_by",
    )

    df = None
    df_full = None
    source_label = ""

    # ── Fetch from Sysdig API ──────────────────────────────────────────────────
    token = st.session_state.get("global_api_token", "")
    if not token:
        st.warning("No API token configured. Enter your API token in the sidebar.")
        return

    with st.spinner("Detecting region…"):
        base_url = _auto_detect_base_url(token)
    if not base_url:
        st.warning("Could not detect your Sysdig region. Check that your token is valid.")
        return

    st.caption(f"Region auto-detected: `{base_url}` · Report: `{POSTURE_REPORT_NAME}`")

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        fetch_btn = st.button("Fetch from API", type="primary", use_container_width=True)
    with col_note:
        if "posture_label" in st.session_state:
            st.caption(f"Last loaded: {st.session_state['posture_label']}")

    if fetch_btn:
        with st.spinner("Fetching posture report from Sysdig Reports Manager…"):
            try:
                df_full, df, report_name = fetch_from_api(token, base_url)
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
                source_label = f"API fetch — {report_name} ({ts})"
                st.session_state["posture_df_full"] = df_full
                st.session_state["posture_df"]      = df
                st.session_state["posture_label"]   = source_label
                st.session_state["posture_token"]   = token
                st.success(f"Fetched {len(df_full):,} rows.")
            except Exception as exc:
                st.error(f"API fetch failed: {exc}")
                return

    if df is None:
        if st.session_state.get("posture_token") == token:
            df_full = st.session_state.get("posture_df_full")
            df      = st.session_state.get("posture_df")
            source_label = st.session_state.get("posture_label", "")
        else:
            for k in ("posture_df_full", "posture_df", "posture_label", "posture_token"):
                st.session_state.pop(k, None)

    if df is None:
        st.info("Click **Fetch from API** to pull the latest posture report on-demand.")
        return

    if df is None or df.empty:
        st.info("No failing controls found in this report.")
        return

    # Display metrics
    st.markdown("---")
    fig_pie, fig_bar, total_failures, unique_owners, unique_accounts, top_owners, owner_stats = create_executive_charts(df, group_by)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Failures", f"{total_failures:,}")
    group_label = "Unique Zones" if group_by == 'Zones' else "Unique Accounts"
    col2.metric(group_label, f"{unique_owners}")
    col3.metric("Total Accounts", f"{unique_accounts}")

    tab1, tab2, tab3 = st.tabs(["Executive Dashboard", "Security Drill-Down", "Download Reports"])
    exec_tab, security_tab, download_tab = tab1, tab2, tab3

    with exec_tab:
        st.markdown("### Executive Summary: Who Should We Engage First?")
        st.markdown(f"*Showing data from: {source_label}*")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("### Top 5 Contributors - What Controls to Fix First")

        # Coloured severity pills — click to filter, click again to clear
        st.markdown("""
<style>
/* Unselected pill colours */
div[data-testid="stPills"] button:nth-child(1) { color:#e74c3c !important; border-color:#e74c3c !important; }
div[data-testid="stPills"] button:nth-child(2) { color:#f39c12 !important; border-color:#f39c12 !important; }
div[data-testid="stPills"] button:nth-child(3) { color:#3498db !important; border-color:#3498db !important; }
div[data-testid="stPills"] button:nth-child(4) { color:#95a5a6 !important; border-color:#95a5a6 !important; }
/* Selected pill — solid fill */
div[data-testid="stPills"] button[aria-pressed="true"]:nth-child(1) { background:#e74c3c !important; color:#fff !important; }
div[data-testid="stPills"] button[aria-pressed="true"]:nth-child(2) { background:#f39c12 !important; color:#fff !important; }
div[data-testid="stPills"] button[aria-pressed="true"]:nth-child(3) { background:#3498db !important; color:#fff !important; }
div[data-testid="stPills"] button[aria-pressed="true"]:nth-child(4) { background:#95a5a6 !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)

        sev_filter = st.pills(
            "Severity filter",
            options=["High", "Medium", "Low", "Info"],
            default=None,
            label_visibility="collapsed",
            key="posture_sev_pills",
        )

        # None = all severities; otherwise filter df to selected severity
        df_for_charts = df if sev_filter is None else df[df['Control Severity'] == sev_filter]

        person_charts = create_person_charts(df_for_charts, top_owners, group_by)

        cols = st.columns(2)
        for i, (owner, fig) in enumerate(person_charts):
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

    with security_tab:
        st.markdown("### Security Posture Drill-Down")

        fig_treemap, fig_heatmap, fig_severity = create_security_charts(df, group_by)

        st.plotly_chart(fig_treemap, use_container_width=True)

        st.markdown("---")
        st.markdown("### Owner vs Control Failure Matrix")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")
        st.markdown("### Severity Breakdown by Owner")
        st.plotly_chart(fig_severity, use_container_width=True)

    with download_tab:
        st.markdown("### Download Reports")

        owner_export, action_df = create_downloadable_reports(df, owner_stats, group_by)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Owner Summary")
            st.dataframe(owner_export.head(10), use_container_width=True)
            csv1 = owner_export.to_csv(index=False)
            st.download_button(
                label="Download Owner Summary CSV",
                data=csv1,
                file_name="owner_summary.csv",
                mime="text/csv"
            )

        with col2:
            st.markdown("#### Actionable Report")
            st.dataframe(action_df.head(10), use_container_width=True)
            csv2 = action_df.to_csv(index=False)
            st.download_button(
                label="Download Actionable Report CSV",
                data=csv2,
                file_name="actionable_report.csv",
                mime="text/csv"
            )
