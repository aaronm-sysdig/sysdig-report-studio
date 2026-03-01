"""
Registry Vulnerability Analytics page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
Chart functions return go.Figure / dict-of-figures; render_page() handles all st.plotly_chart() calls.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from streamlit_sortables import sort_items

from config import get_api_config, VULN_DATA_DIR


# Color mapping for vulnerability severity levels (used in charts)
SEVERITY_COLORS = {
    'Critical': '#9b59b6',  # Purple for critical issues
    'High': '#e74c3c',      # Red for high severity
    'Medium': '#f39c12',    # Orange for medium severity
    'Low': '#3498db',       # Blue for low severity
    'Negligible': '#95a5a6',  # Gray for negligible/informational
}

# Ordered list of severity levels from most to least severe
SEVERITY_ORDER = ['Critical', 'High', 'Medium', 'Low', 'Negligible']

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0bec5", size=12),
    margin=dict(t=40, b=20, l=20, r=20),
)


def fetch_registry_results(api_token: str, base_url: str, limit: int = 100) -> list[dict]:
    """
    Fetch all registry vulnerability scan results from Sysdig API.

    Uses cursor-based pagination to retrieve all results across multiple
    API calls. Each page contains up to 'limit' results.

    Args:
        api_token: Bearer token for Sysdig API authentication
        base_url: Base URL for the Sysdig API (e.g. https://secure.sysdig.com)
        limit: Number of results per page (default: 100)

    Returns:
        list[dict]: List of image vulnerability records from the registry scanner

    Raises:
        requests.HTTPError: If API request fails
        requests.Timeout: If request exceeds 60 second timeout
    """
    url = f"{base_url}/secure/vulnerability/v1/registry-results"
    headers = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}
    all_results = []
    cursor = None

    # Paginate through all results using cursor-based pagination
    while True:
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        body = resp.json()

        # Extract data from response and accumulate results
        data = body.get("data", [])
        all_results.extend(data)

        # Check for next page cursor
        page_info = body.get("page", {})
        cursor = page_info.get("next")
        if not cursor:
            break

    return all_results


def save_results_to_disk(results: list[dict], folder: Path = VULN_DATA_DIR) -> Path:
    """
    Save fetched vulnerability results to disk as a timestamped JSON file.

    Creates a snapshot file that can be used later for trend analysis
    or offline viewing.

    Args:
        results: List of vulnerability scan results from API
        folder: Directory to save the snapshot (default: VULN_DATA_DIR from config)

    Returns:
        Path: Path to the saved JSON file
    """
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = folder / f"registry_vuln_{ts}.json"

    # Create payload with metadata and results
    payload = {
        "fetched_at": datetime.now().isoformat(),
        "total_images": len(results),
        "data": results,
    }
    filepath.write_text(json.dumps(payload, default=str))
    return filepath


def list_saved_snapshots(folder: Path = VULN_DATA_DIR) -> list[dict]:
    """
    List all saved vulnerability snapshot files.

    Scans the snapshot directory for JSON files and returns metadata
    about each snapshot, sorted by date (newest first).

    Args:
        folder: Directory containing snapshot files

    Returns:
        list[dict]: List of snapshot metadata including path, filename,
                   fetch timestamp, and image count
    """
    if not folder.exists():
        return []

    files = sorted(folder.glob("registry_vuln_*.json"), reverse=True)
    snapshots = []

    for f in files:
        try:
            meta = json.loads(f.read_text())
            snapshots.append({
                "path": f,
                "filename": f.name,
                "fetched_at": meta.get("fetched_at", "unknown"),
                "total_images": meta.get("total_images", 0),
            })
        except (json.JSONDecodeError, KeyError):
            # Skip corrupted or invalid files
            continue

    return snapshots


def load_snapshot(filepath) -> list[dict]:
    """
    Load vulnerability data from a JSON snapshot file.

    Supports both file paths and file-like objects (for uploaded files).

    Args:
        filepath: Path to JSON file or file-like object

    Returns:
        tuple: (data list, fetched_at timestamp string)
    """
    # Handle file-like objects (e.g., Streamlit uploaded files)
    if hasattr(filepath, 'read'):
        raw = filepath.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        payload = json.loads(raw)
    else:
        # Handle file path
        payload = json.loads(Path(filepath).read_text())

    return payload.get("data", []), payload.get("fetched_at", "unknown")


def normalize_image_data(results: list[dict]) -> pd.DataFrame:
    """
    Convert raw Sysdig API results into a normalized DataFrame for analysis.

    Transforms the nested API response structure into a flat table format
    suitable for charting and data analysis. Handles field name variations
    across different API versions.

    Args:
        results: List of raw image scan results from Sysdig API

    Returns:
        pd.DataFrame: Normalized data with columns for image info, vulnerability
                     counts by severity, and metadata. Sorted by total
                     vulnerabilities (descending).
    """
    rows = []

    for r in results:
        # Extract vulnerability counts by severity
        # Handle field name variations across API versions
        vuln_sev = r.get("vulnTotalBySeverity",
                         r.get("vulnsBySev",
                                r.get("vulnTotalBySev", {})))
        fix_sev = r.get("fixableVulnsBySeverity",
                        r.get("fixableVulnsBySev", {}))

        # Extract counts for each severity level (handle case variations)
        crit = vuln_sev.get("critical", vuln_sev.get("Critical", 0))
        high = vuln_sev.get("high", vuln_sev.get("High", 0))
        med = vuln_sev.get("medium", vuln_sev.get("Medium", 0))
        low = vuln_sev.get("low", vuln_sev.get("Low", 0))
        neg = vuln_sev.get("negligible", vuln_sev.get("Negligible", 0))
        total_vulns = crit + high + med + low + neg

        # Extract fixable vulnerability counts
        fix_crit = fix_sev.get("critical", fix_sev.get("Critical", 0))
        fix_high = fix_sev.get("high", fix_sev.get("High", 0))
        fix_med = fix_sev.get("medium", fix_sev.get("Medium", 0))
        fix_low = fix_sev.get("low", fix_sev.get("Low", 0))
        fix_neg = fix_sev.get("negligible", fix_sev.get("Negligible", 0))
        total_fixable = fix_crit + fix_high + fix_med + fix_low + fix_neg

        # Get image pull string (the full image reference)
        pull_string = r.get("pullString", r.get("imagePullString", ""))

        # Parse repository and tag from pullString
        # Format: "registry/repo/image:tag" -> repo="registry/repo/image", tag="tag"
        parsed_repo = pull_string
        parsed_tag = ""
        if ":" in pull_string:
            parts = pull_string.rsplit(":", 1)
            parsed_repo = parts[0]
            parsed_tag = parts[1]

        # Build normalized row with all relevant fields
        row = {
            "image_id": r.get("imageId", r.get("resultId", "")),
            "result_id": r.get("resultId", ""),
            "pull_string": pull_string,
            "repository": parsed_repo,
            "tag": parsed_tag or r.get("tag", ""),
            "vendor": r.get("vendor", ""),
            "created_at": r.get("createdAt", ""),
            # Vulnerability counts by severity
            "critical": crit,
            "high": high,
            "medium": med,
            "low": low,
            "negligible": neg,
            # Fixable vulnerability counts
            "fix_critical": fix_crit,
            "fix_high": fix_high,
            "fix_medium": fix_med,
            "fix_low": fix_low,
            "fix_negligible": fix_neg,
            # Aggregate counts
            "total_vulns": total_vulns,
            "total_fixable": total_fixable,
            "total_unfixable": total_vulns - total_fixable,
            # Additional metadata
            "exploit_count": r.get("exploitCount", r.get("exploitableCount", 0)),
            "policy_status": r.get("policyStatus", r.get("policyEvaluation", "")),
            "in_use": r.get("inUse", False),
        }

        # Create a short display name for charts (just image name:tag)
        name_part = parsed_repo.split("/")[-1] if "/" in parsed_repo else parsed_repo
        row["display_name"] = f"{name_part}:{parsed_tag}" if parsed_tag else name_part
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort by total vulnerabilities for priority-based analysis
    df = df.sort_values("total_vulns", ascending=False).reset_index(drop=True)
    return df


def create_vuln_executive_charts(df: pd.DataFrame):
    """
    Create executive dashboard charts for vulnerability analytics.

    Generates multiple visualizations for analyzing container image
    vulnerabilities from registry scans:
    - Top 5 most vulnerable images (stacked bar)
    - Severity distribution (donut chart)
    - Vulnerabilities by vendor
    - Severity breakdown by vendor
    - Priority patching list (Critical + High)
    - Vulnerability distribution histogram

    Args:
        df: Normalized DataFrame from normalize_image_data()

    Returns:
        dict: Chart figures keyed by name (e.g., 'fig_top5', 'fig_severity')
              Also includes aggregate stats: 'total_images', 'total_vulns', 'total_critical'
    """
    if df.empty:
        return {}

    total_images = len(df)
    total_vulns = int(df['total_vulns'].sum())
    total_critical = int(df['critical'].sum())

    charts = {
        "total_images": total_images,
        "total_vulns": total_vulns,
        "total_critical": total_critical,
    }

    # --- 1. Top 5 most vulnerable images (stacked horizontal bar) ---
    top5 = df.head(5).copy()
    top5_sorted = top5.sort_values('total_vulns', ascending=True)
    # Truncate long labels
    top5_sorted = top5_sorted.copy()
    top5_sorted['short_name'] = top5_sorted['display_name'].apply(lambda n: n[:35] + '...' if len(n) > 35 else n)

    fig_top5 = go.Figure()
    for sev in SEVERITY_ORDER:
        col = sev.lower()
        fig_top5.add_trace(go.Bar(
            y=top5_sorted['short_name'],
            x=top5_sorted[col],
            name=sev,
            orientation='h',
            marker_color=SEVERITY_COLORS[sev],
            text=top5_sorted[col],
            textposition='inside',
            hovertext=top5_sorted['display_name'],
            hoverinfo='text+x',
        ))
    fig_top5.update_layout(
        barmode='stack',
        title=dict(text='<b>Top 5 Most Vulnerable Images</b>', x=0.5, xanchor='center', font=dict(size=16)),
        height=500,
        margin=dict(t=60, b=80, l=40, r=40),
        xaxis=dict(title='Vulnerability Count'),
        yaxis=dict(type='category', automargin=True),
        legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5),
        plot_bgcolor='#fafafa',
    )
    charts["fig_top5"] = fig_top5

    # --- 2. Severity distribution donut ---
    sev_totals = {s: int(df[s.lower()].sum()) for s in SEVERITY_ORDER}
    sev_labels = [s for s in SEVERITY_ORDER if sev_totals[s] > 0]
    sev_values = [sev_totals[s] for s in sev_labels]
    sev_colors = [SEVERITY_COLORS[s] for s in sev_labels]

    fig_sev = go.Figure(go.Pie(
        labels=sev_labels,
        values=sev_values,
        marker_colors=sev_colors,
        hole=0.45,
        textinfo='percent',
        textposition='inside',
        textfont=dict(color='white', size=12),
        sort=False,
    ))
    fig_sev.add_annotation(
        text=f"<b>{total_vulns:,}</b><br>Total", x=0.5, y=0.5, font_size=15, showarrow=False
    )
    fig_sev.update_layout(
        title=dict(text='<b>Vulnerability Severity Distribution</b>', x=0.5, xanchor='center', font=dict(size=16)),
        height=450,
        margin=dict(t=80, b=40, l=40, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
    )
    charts["fig_severity"] = fig_sev

    # --- 3. Vulnerability count by vendor ---
    vendor_vulns = df.groupby('vendor').agg(
        images=('vendor', 'size'),
        total_vulns=('total_vulns', 'sum'),
        critical=('critical', 'sum'),
        high=('high', 'sum'),
    ).reset_index().sort_values('total_vulns', ascending=False).head(10)

    if not vendor_vulns.empty:
        vendor_sorted = vendor_vulns.sort_values('total_vulns', ascending=True)
        fig_vendor = go.Figure(go.Bar(
            y=vendor_sorted['vendor'],
            x=vendor_sorted['total_vulns'],
            orientation='h',
            marker_color='#e67e22',
            text=[f"{v:,} ({i} images)" for v, i in zip(vendor_sorted['total_vulns'], vendor_sorted['images'])],
            textposition='inside',
            textfont=dict(color='white', size=12),
            insidetextanchor='end',
        ))
        fig_vendor.update_layout(
            title=dict(text='<b>Vulnerabilities by Registry Vendor</b>', x=0.5, xanchor='center', font=dict(size=16)),
            height=400,
            margin=dict(t=80, b=40, l=40, r=40),
            xaxis=dict(title='Total Vulnerabilities'),
            yaxis=dict(type='category', automargin=True),
            plot_bgcolor='#fafafa',
        )
        charts["fig_vendor"] = fig_vendor

    # --- 4. Severity breakdown per vendor (stacked bar) ---
    if not vendor_vulns.empty:
        top_vendors = vendor_vulns.head(8).sort_values('total_vulns', ascending=True)
        fig_vendor_sev = go.Figure()
        for sev in SEVERITY_ORDER:
            col = sev.lower()
            vendor_sev_data = df.groupby('vendor')[col].sum().reset_index()
            vendor_sev_data = vendor_sev_data[vendor_sev_data['vendor'].isin(top_vendors['vendor'])]
            # Re-sort to match
            vendor_sev_data = vendor_sev_data.set_index('vendor').loc[top_vendors['vendor']].reset_index()
            fig_vendor_sev.add_trace(go.Bar(
                y=vendor_sev_data['vendor'],
                x=vendor_sev_data[col],
                name=sev,
                orientation='h',
                marker_color=SEVERITY_COLORS[sev],
            ))
        fig_vendor_sev.update_layout(
            barmode='stack',
            title=dict(text='<b>Severity Breakdown by Vendor</b>', x=0.5, xanchor='center', font=dict(size=16)),
            height=450,
            margin=dict(t=60, b=80, l=40, r=40),
            xaxis=dict(title='Vulnerability Count'),
            yaxis=dict(type='category', automargin=True),
            legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5),
            plot_bgcolor='#fafafa',
        )
        charts["fig_vendor_severity"] = fig_vendor_sev

    # --- 5. Images with most critical + high vulns (priority to patch) ---
    df_priority = df[['display_name', 'critical', 'high', 'pull_string']].copy()
    df_priority['crit_high'] = df_priority['critical'] + df_priority['high']
    df_priority = df_priority[df_priority['crit_high'] > 0].sort_values('crit_high', ascending=False).head(10)

    if not df_priority.empty:
        priority_sorted = df_priority.sort_values('crit_high', ascending=True).copy()
        priority_sorted['short_name'] = priority_sorted['display_name'].apply(lambda n: n[:35] + '...' if len(n) > 35 else n)
        fig_priority = go.Figure()
        fig_priority.add_trace(go.Bar(
            y=priority_sorted['short_name'],
            x=priority_sorted['critical'],
            name='Critical',
            orientation='h',
            marker_color=SEVERITY_COLORS['Critical'],
            hovertext=priority_sorted['display_name'],
            hoverinfo='text+x',
        ))
        fig_priority.add_trace(go.Bar(
            y=priority_sorted['short_name'],
            x=priority_sorted['high'],
            name='High',
            orientation='h',
            marker_color=SEVERITY_COLORS['High'],
            hovertext=priority_sorted['display_name'],
            hoverinfo='text+x',
        ))
        fig_priority.update_layout(
            barmode='stack',
            title=dict(text='<b>Top 10 Images to Patch (Critical + High)</b>', x=0.5, xanchor='center', font=dict(size=16)),
            height=550,
            margin=dict(t=60, b=80, l=40, r=40),
            xaxis=dict(title='Vulnerability Count'),
            yaxis=dict(type='category', automargin=True),
            legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
            plot_bgcolor='#fafafa',
        )
        charts["fig_priority"] = fig_priority

    # --- 6. Vulnerability density histogram (vulns per image distribution) ---
    fig_hist = go.Figure(go.Histogram(
        x=df['total_vulns'],
        nbinsx=30,
        marker_color='#3498db',
    ))
    fig_hist.update_layout(
        title=dict(text='<b>Vulnerability Distribution Across Images</b>', x=0.5, xanchor='center', font=dict(size=16)),
        height=500,
        margin=dict(t=80, b=60, l=60, r=40),
        xaxis=dict(title='Vulnerabilities per Image'),
        yaxis=dict(title='Number of Images'),
        plot_bgcolor='#fafafa',
    )
    charts["fig_histogram"] = fig_hist

    return charts


def create_vuln_trend_charts(snapshots: list[tuple[str, pd.DataFrame]]):
    """
    Create trend charts comparing vulnerability data across multiple snapshots.

    Visualizes how vulnerability counts change over time to track
    remediation progress or identify new issues.

    Args:
        snapshots: List of (date_string, dataframe) tuples, sorted by date

    Returns:
        dict: Contains trend figures and summary DataFrame:
              - 'fig_total_trend': Total vulnerabilities over time
              - 'fig_severity_trend': Severity breakdown over time
              - 'fig_image_trend': Top 5 images tracked over time
              - 'summary_df': Change summary table
              Returns empty dict if fewer than 2 snapshots provided
    """
    if len(snapshots) < 2:
        return {}

    charts = {}

    # Aggregate totals per snapshot
    trend_rows = []
    for date_str, df in snapshots:
        row = {"date": date_str, "total_vulns": int(df['total_vulns'].sum()), "total_images": len(df)}
        for sev in SEVERITY_ORDER:
            row[sev] = int(df[sev.lower()].sum())
        row["fixable"] = int(df['total_fixable'].sum())
        row["exploitable_images"] = int((df['exploit_count'] > 0).sum())
        trend_rows.append(row)

    trend_df = pd.DataFrame(trend_rows).sort_values("date")

    # --- 1. Total vulnerabilities over time ---
    fig_total = go.Figure(go.Scatter(
        x=trend_df['date'], y=trend_df['total_vulns'],
        mode='lines+markers+text',
        text=trend_df['total_vulns'].apply(lambda v: f"{v:,}"),
        textposition='top center',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10),
    ))
    fig_total.update_layout(
        title=dict(text='<b>Total Vulnerabilities Over Time</b>', x=0.5, xanchor='center', font=dict(size=16)),
        height=400,
        xaxis=dict(title='Snapshot Date'),
        yaxis=dict(title='Total Vulnerabilities'),
        plot_bgcolor='#fafafa',
    )
    charts["fig_total_trend"] = fig_total

    # --- 2. Severity breakdown over time (stacked area) ---
    fig_sev_trend = go.Figure()
    for sev in reversed(SEVERITY_ORDER):
        fig_sev_trend.add_trace(go.Scatter(
            x=trend_df['date'], y=trend_df[sev],
            name=sev, mode='lines',
            stackgroup='one',
            line=dict(color=SEVERITY_COLORS[sev]),
        ))
    fig_sev_trend.update_layout(
        title=dict(text='<b>Severity Breakdown Over Time</b>', x=0.5, xanchor='center', font=dict(size=16)),
        height=400,
        xaxis=dict(title='Snapshot Date'),
        yaxis=dict(title='Vulnerability Count'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        plot_bgcolor='#fafafa',
    )
    charts["fig_severity_trend"] = fig_sev_trend

    # --- 3. Top 5 images tracked over time ---
    # Use images from the latest snapshot
    latest_df = snapshots[-1][1]
    top5_names = latest_df.head(5)['display_name'].tolist()

    fig_img_trend = go.Figure()
    for img_name in top5_names:
        img_vals = []
        for date_str, df in snapshots:
            match = df[df['display_name'] == img_name]
            img_vals.append(int(match['total_vulns'].sum()) if not match.empty else 0)
        fig_img_trend.add_trace(go.Scatter(
            x=trend_df['date'].tolist(), y=img_vals,
            mode='lines+markers',
            name=img_name[:30],
        ))
    fig_img_trend.update_layout(
        title=dict(text='<b>Top 5 Images - Vulnerability Trend</b>', x=0.5, xanchor='center', font=dict(size=16)),
        height=400,
        xaxis=dict(title='Snapshot Date'),
        yaxis=dict(title='Total Vulnerabilities'),
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
        hovermode='x unified',
        plot_bgcolor='#fafafa',
    )
    charts["fig_image_trend"] = fig_img_trend

    # --- 4. Summary table with change indicators ---
    if len(trend_df) >= 2:
        first = trend_df.iloc[0]
        last = trend_df.iloc[-1]
        summary_rows = []
        for sev in SEVERITY_ORDER:
            f_val = int(first[sev])
            l_val = int(last[sev])
            change = l_val - f_val
            pct = ((change / f_val) * 100) if f_val > 0 else 0
            trend = "Improving" if change < 0 else ("Worsening" if change > 0 else "Stable")
            arrow = "v" if change < 0 else ("^" if change > 0 else "-")
            summary_rows.append({
                "Severity": sev,
                "First Snapshot": f_val,
                "Latest Snapshot": l_val,
                "Change": change,
                "% Change": f"{pct:+.1f}%",
                "Trend": arrow,
            })
        charts["summary_df"] = pd.DataFrame(summary_rows)

    return charts


# =============================================================================
# VULNERABILITY DASHBOARD UI CONFIGURATION
# =============================================================================

# Default widget display order for the vulnerability dashboard
DEFAULT_VULN_WIDGET_ORDER = [
    "Top 5 Most Vulnerable Images",
    "Severity Distribution",
    "Vulnerabilities by Vendor",
    "Severity Breakdown by Vendor",
    "Top 10 Images to Patch",
    "Vulnerability Distribution",
]

DEFAULT_VULN_WIDGET_WIDTHS = {name: "half" for name in DEFAULT_VULN_WIDGET_ORDER}

# Map widget names to chart keys returned by create_vuln_executive_charts
WIDGET_CHART_KEYS = {
    "Top 5 Most Vulnerable Images": "fig_top5",
    "Severity Distribution": "fig_severity",
    "Vulnerabilities by Vendor": "fig_vendor",
    "Severity Breakdown by Vendor": "fig_vendor_severity",
    "Top 10 Images to Patch": "fig_priority",
    "Vulnerability Distribution": "fig_histogram",
}


def _init_vuln_layout_state():
    """Initialize session state for dashboard layout if not already set."""
    if "vuln_widget_order" not in st.session_state:
        st.session_state.vuln_widget_order = list(DEFAULT_VULN_WIDGET_ORDER)
    if "vuln_widget_widths" not in st.session_state:
        st.session_state.vuln_widget_widths = dict(DEFAULT_VULN_WIDGET_WIDTHS)
    if "vuln_widget_visible" not in st.session_state:
        st.session_state.vuln_widget_visible = set(DEFAULT_VULN_WIDGET_ORDER)


def _render_dashboard_widgets(charts: dict):
    """Render widgets in user-configured order and width."""
    order = st.session_state.vuln_widget_order
    widths = st.session_state.vuln_widget_widths
    visible = st.session_state.vuln_widget_visible

    half_buffer = []  # (name, fig) pairs waiting for a column partner

    def _flush_half_buffer():
        nonlocal half_buffer
        if not half_buffer:
            return
        if len(half_buffer) == 1:
            name, fig = half_buffer[0]
            col1, col2 = st.columns(2)
            with col1:
                with st.expander(name, expanded=True):
                    st.plotly_chart(fig, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                with st.expander(half_buffer[0][0], expanded=True):
                    st.plotly_chart(half_buffer[0][1], use_container_width=True)
            with col2:
                with st.expander(half_buffer[1][0], expanded=True):
                    st.plotly_chart(half_buffer[1][1], use_container_width=True)
        half_buffer = []

    for widget_name in order:
        if widget_name not in visible:
            continue

        chart_key = WIDGET_CHART_KEYS.get(widget_name)
        if not chart_key or chart_key not in charts:
            continue

        fig = charts[chart_key]
        width = widths.get(widget_name, "half")

        if width == "full":
            _flush_half_buffer()
            with st.expander(widget_name, expanded=True):
                st.plotly_chart(fig, use_container_width=True)
        else:
            half_buffer.append((widget_name, fig))
            if len(half_buffer) == 2:
                _flush_half_buffer()

    _flush_half_buffer()


def render_page():
    """
    Render the Vulnerability Analytics page.

    This page provides:
    - API integration to fetch registry scan results from Sysdig
    - Interactive dashboard with customizable widget layouts
    - Trend analysis when multiple snapshots are available
    - Data explorer for searching and filtering images
    - Export functionality for CSV and JSON formats
    """
    st.title("Registry Vulnerability Analytics")
    st.markdown("Fetch and analyze container image vulnerabilities from the Sysdig registry scanner.")

    _init_vuln_layout_state()
    api_token, base_url = get_api_config()

    if not api_token:
        st.info("Set your API token in the sidebar to get started.")
        return

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Vulnerability Scanner")

        st.markdown("---")
        query_freq = st.selectbox(
            "Recommended query frequency",
            options=["Daily", "Weekly"],
            index=1,
            help="How often you plan to fetch fresh data. This is informational only.",
        )

        fetch_clicked = st.button(
            "Fetch Latest Data",
            disabled=not api_token,
            help="Query the Sysdig API and save results locally",
            type="primary",
        )

        st.markdown("---")
        st.subheader("Saved Snapshots")
        snapshots = list_saved_snapshots()
        if snapshots:
            st.markdown(f"**{len(snapshots)}** snapshot(s) in `~/sysdig-vuln-data/`")
            for s in snapshots[:5]:
                st.text(f"  {s['filename'][:35]}  ({s['total_images']} imgs)")
        else:
            st.markdown("No snapshots saved yet. Click **Fetch Latest Data** to start.")

        st.markdown("---")
        st.subheader("Upload Snapshots for Trends")
        uploaded_snapshots = st.file_uploader(
            "Upload saved JSON snapshots",
            type=["json"],
            accept_multiple_files=True,
            help="Upload previously downloaded snapshot JSONs to compare over time.",
            key="vuln_uploader",
        )

        # --- Dashboard layout controls ---
        st.markdown("---")
        st.subheader("Dashboard Layout")

        st.caption("Drag to reorder widgets:")
        new_order = sort_items(
            st.session_state.vuln_widget_order,
            direction="vertical",
            key="vuln_sort",
        )
        if new_order != st.session_state.vuln_widget_order:
            st.session_state.vuln_widget_order = new_order
            st.rerun()

        st.markdown("---")
        st.caption("Show / hide widgets:")
        visible_selection = st.multiselect(
            "Visible widgets",
            options=st.session_state.vuln_widget_order,
            default=list(st.session_state.vuln_widget_visible),
            key="vuln_visible_select",
            label_visibility="collapsed",
        )
        st.session_state.vuln_widget_visible = set(visible_selection)

        st.markdown("---")
        st.caption("Widget width:")
        for wname in st.session_state.vuln_widget_order:
            if wname not in st.session_state.vuln_widget_visible:
                continue
            current = st.session_state.vuln_widget_widths.get(wname, "half")
            is_full = st.toggle(
                f"{wname}",
                value=(current == "full"),
                key=f"width_{wname}",
                help="Toggle full width",
            )
            st.session_state.vuln_widget_widths[wname] = "full" if is_full else "half"

        st.markdown("---")
        if st.button("Reset Layout", key="reset_layout"):
            st.session_state.vuln_widget_order = list(DEFAULT_VULN_WIDGET_ORDER)
            st.session_state.vuln_widget_widths = dict(DEFAULT_VULN_WIDGET_WIDTHS)
            st.session_state.vuln_widget_visible = set(DEFAULT_VULN_WIDGET_ORDER)
            st.rerun()

    # --- Handle fetch ---
    if fetch_clicked and api_token:
        with st.spinner("Fetching registry results from Sysdig API (this may take a moment)..."):
            try:
                results = fetch_registry_results(api_token, base_url)
                filepath = save_results_to_disk(results)
                st.success(f"Fetched **{len(results)}** images. Saved to `{filepath.name}`")
                st.rerun()
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.status_code} - {e.response.text[:300]}")
                return
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return

    # --- Determine data sources ---
    # Collect all available snapshots: local files + uploaded files
    all_snapshots = []  # list of (date_str, df)

    # Load from local saved files
    for s in snapshots:
        try:
            data, fetched_at = load_snapshot(s["path"])
            df = normalize_image_data(data)
            if not df.empty:
                all_snapshots.append((fetched_at, df))
        except Exception:
            continue

    # Load from uploaded files
    if uploaded_snapshots:
        for uf in uploaded_snapshots:
            try:
                data, fetched_at = load_snapshot(uf)
                df = normalize_image_data(data)
                if not df.empty:
                    all_snapshots.append((fetched_at, df))
            except Exception:
                continue

    # Deduplicate by date string and sort
    seen_dates = set()
    unique_snapshots = []
    for date_str, df in all_snapshots:
        if date_str not in seen_dates:
            seen_dates.add(date_str)
            unique_snapshots.append((date_str, df))
    all_snapshots = sorted(unique_snapshots, key=lambda x: x[0])

    if not all_snapshots:
        st.info("No data available. Use **Fetch Latest Data** in the sidebar to query the Sysdig API, or upload saved JSON snapshots.")
        return

    # Use the most recent snapshot for the dashboard
    latest_date, latest_df = all_snapshots[-1]
    has_trends = len(all_snapshots) >= 2

    # --- Build tabs ---
    if has_trends:
        tab_dash, tab_trend, tab_explore, tab_dl = st.tabs(
            ["Dashboard", "Trend Analysis", "Data Explorer", "Download"]
        )
    else:
        tab_dash, tab_explore, tab_dl = st.tabs(
            ["Dashboard", "Data Explorer", "Download"]
        )

    # === Dashboard tab ===
    with tab_dash:
        charts = create_vuln_executive_charts(latest_df)
        if not charts:
            st.warning("No vulnerability data to display.")
            return

        st.markdown(f"*Snapshot: {latest_date}*")

        # KPI row (always pinned at top)
        c1, c2, c3, c4, c5 = st.columns(5)
        total_high = int(latest_df['high'].sum())
        total_med = int(latest_df['medium'].sum())

        c1.metric("Images Scanned", f"{charts['total_images']:,}")
        c2.metric("Total Vulns", f"{charts['total_vulns']:,}")
        c3.metric("Critical", f"{charts['total_critical']:,}")
        c4.metric("High", f"{total_high:,}")
        c5.metric("Medium", f"{total_med:,}")

        st.markdown("---")

        # Customizable widget area
        _render_dashboard_widgets(charts)

    # === Trend Analysis tab ===
    if has_trends:
        with tab_trend:
            st.markdown(f"### Vulnerability Trends ({len(all_snapshots)} snapshots)")
            trend_charts = create_vuln_trend_charts(all_snapshots)

            if "fig_total_trend" in trend_charts:
                st.plotly_chart(trend_charts["fig_total_trend"], use_container_width=True)

            if "summary_df" in trend_charts:
                st.markdown("---")
                st.markdown("### Change Summary")

                def color_trend(val):
                    if val == 'v':
                        return 'color: green; font-weight: bold'
                    elif val == '^':
                        return 'color: red; font-weight: bold'
                    return ''

                st.dataframe(
                    trend_charts["summary_df"].style.applymap(color_trend, subset=['Trend']),
                    use_container_width=True, hide_index=True,
                )

            if "fig_severity_trend" in trend_charts:
                st.markdown("---")
                st.plotly_chart(trend_charts["fig_severity_trend"], use_container_width=True)

            if "fig_image_trend" in trend_charts:
                st.markdown("---")
                st.plotly_chart(trend_charts["fig_image_trend"], use_container_width=True)

    # === Data Explorer tab ===
    with tab_explore:
        st.markdown("### All Scanned Images")
        st.markdown(f"*{len(latest_df)} images from snapshot {latest_date}*")

        # Search filter
        search = st.text_input("Search images", placeholder="Filter by name, vendor, or pull string...")
        display_df = latest_df.copy()
        if search:
            mask = (
                display_df['display_name'].str.contains(search, case=False, na=False) |
                display_df['vendor'].str.contains(search, case=False, na=False) |
                display_df['pull_string'].str.contains(search, case=False, na=False)
            )
            display_df = display_df[mask]

        show_cols = ['display_name', 'vendor', 'critical', 'high', 'medium', 'low',
                     'negligible', 'total_vulns', 'created_at']
        st.dataframe(
            display_df[show_cols].rename(columns={
                'display_name': 'Image', 'vendor': 'Vendor',
                'critical': 'Critical', 'high': 'High', 'medium': 'Medium', 'low': 'Low',
                'negligible': 'Negligible', 'total_vulns': 'Total',
                'created_at': 'Scanned At',
            }),
            use_container_width=True, hide_index=True, height=600,
        )

    # === Download tab ===
    with tab_dl:
        st.markdown("### Export Data")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Image Summary CSV")
            csv_data = latest_df.to_csv(index=False)
            st.download_button(
                label="Download Image Summary",
                data=csv_data,
                file_name=f"vuln_image_summary_{latest_date[:10]}.csv",
                mime="text/csv",
            )

        with col2:
            st.markdown("#### Raw JSON Snapshot")
            # Re-read the latest local snapshot if available
            if snapshots:
                raw = Path(snapshots[0]["path"]).read_text()
                st.download_button(
                    label="Download Latest JSON Snapshot",
                    data=raw,
                    file_name=snapshots[0]["filename"],
                    mime="application/json",
                )
            else:
                st.info("Fetch data first to enable JSON download.")
