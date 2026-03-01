"""
Posture Analytics page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
Chart functions return go.Figure objects; render_page() handles all st.plotly_chart() calls.
"""
import gzip
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def extract_date_from_filename(filename: str) -> datetime:
    """
    Extract date from a filename containing an ISO date pattern.

    Args:
        filename: Name of the file (e.g., 'Report_2026-01-31T03_25_25.610Z.csv.gz')

    Returns:
        datetime: Extracted date, or current date if no pattern found

    Example:
        >>> extract_date_from_filename('Report_2026-01-31T03_25_25.csv')
        datetime(2026, 1, 31)
    """
    # Match ISO date pattern: YYYY-MM-DD optionally followed by time
    pattern = r'(\d{4}-\d{2}-\d{2})T?(\d{2}[_:]\d{2}[_:]\d{2})?'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y-%m-%d')
    # Fallback to current date if no date found
    return datetime.now()


def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load posture report CSV data from an uploaded file.

    Supports multiple file formats:
    - Plain CSV files (.csv)
    - Gzipped CSV files (.csv.gz)
    - ZIP archives containing CSV files (.zip)

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        tuple: (full_dataframe, failing_controls_only_dataframe)

    Raises:
        ValueError: If ZIP archive contains no CSV files
    """
    filename = uploaded_file.name

    # Handle ZIP archives
    if filename.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            # Find CSV files in the zip (including gzipped ones)
            csv_files = [f for f in z.namelist() if f.endswith('.csv') or f.endswith('.csv.gz')]
            if not csv_files:
                raise ValueError("No CSV files found in the zip archive")

            # Use the first CSV file found
            csv_name = csv_files[0]
            if csv_name.endswith('.gz'):
                with z.open(csv_name) as zf:
                    with gzip.open(zf, 'rt') as f:
                        df = pd.read_csv(f)
            else:
                with z.open(csv_name) as zf:
                    df = pd.read_csv(zf)

    # Handle gzipped CSV files
    elif filename.endswith('.gz'):
        with gzip.open(uploaded_file, 'rt') as f:
            df = pd.read_csv(f)

    # Handle plain CSV files
    else:
        df = pd.read_csv(uploaded_file)

    # Filter to only failing controls for analysis
    df_fail = df[df['Result'] == 'Fail'].copy()

    return df, df_fail


def load_multiple_files(uploaded_files, group_by: str = 'Zones') -> pd.DataFrame:
    """
    Load multiple posture report CSV files for trend analysis.

    Combines data from multiple reports (typically from different dates)
    into a single DataFrame suitable for tracking changes over time.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        group_by: Column to group failures by ('Zones' or 'Account Id')

    Returns:
        pd.DataFrame: Combined data with columns for Owner, Total Failures,
                     Unique Controls, Report Date, and Filename
    """
    all_data = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        report_date = extract_date_from_filename(filename)

        if filename.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv') or f.endswith('.csv.gz')]
                if not csv_files:
                    continue  # Skip zip files without CSVs
                csv_name = csv_files[0]
                if csv_name.endswith('.gz'):
                    with z.open(csv_name) as zf:
                        with gzip.open(zf, 'rt') as f:
                            df = pd.read_csv(f)
                else:
                    with z.open(csv_name) as zf:
                        df = pd.read_csv(zf)
        elif filename.endswith('.gz'):
            with gzip.open(uploaded_file, 'rt') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(uploaded_file)

        # Filter to only failing controls
        df_fail = df[df['Result'] == 'Fail'].copy()

        # Aggregate by group_by column
        summary = df_fail.groupby(group_by).agg({
            'Control ID': 'count',
            'Control Name': 'nunique'
        }).reset_index()
        summary.columns = ['Owner', 'Total Failures', 'Unique Controls']
        summary['Report Date'] = report_date
        summary['Filename'] = filename

        all_data.append(summary)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['Owner', 'Report Date'])
        return combined
    return pd.DataFrame()


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

        controls = all_controls.sort_values('Count', ascending=True).tail(8).reset_index(drop=True)

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


def create_trend_charts(trend_data: pd.DataFrame):
    """
    Create trend analysis charts for tracking failures over time.

    Visualizes how compliance failures change across multiple report
    snapshots, helping identify improvement or regression.

    Args:
        trend_data: DataFrame with aggregated failure data per owner/date

    Returns:
        tuple: (line_chart, stacked_area_chart, summary_dataframe)
               Returns (None, None, None) if trend_data is empty
    """
    if trend_data.empty:
        return None, None

    # Get top 10 owners by total failures across all reports
    top_owners = trend_data.groupby('Owner')['Total Failures'].sum().nlargest(10).index.tolist()
    trend_filtered = trend_data[trend_data['Owner'].isin(top_owners)]

    # Line chart - Total Failures over time per owner
    fig_trend = go.Figure()

    for owner in top_owners:
        owner_data = trend_filtered[trend_filtered['Owner'] == owner].sort_values('Report Date')
        fig_trend.add_trace(go.Scatter(
            x=owner_data['Report Date'],
            y=owner_data['Total Failures'],
            mode='lines+markers',
            name=str(owner)[:25] + '...' if len(str(owner)) > 25 else str(owner),
            hovertemplate=f'{owner}<br>Date: %{{x}}<br>Failures: %{{y}}<extra></extra>'
        ))

    fig_trend.update_layout(
        title='<b>Failure Trend Over Time (Top 10 Contributors)</b><br><sup>Goal: See these lines go down!</sup>',
        xaxis_title='Report Date',
        yaxis_title='Total Failures',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
    )

    # Summary table with change indicators
    summary_data = []
    for owner in top_owners:
        owner_data = trend_filtered[trend_filtered['Owner'] == owner].sort_values('Report Date')
        if len(owner_data) >= 2:
            first_val = owner_data.iloc[0]['Total Failures']
            last_val = owner_data.iloc[-1]['Total Failures']
            change = last_val - first_val
            pct_change = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
            trend = '↓' if change < 0 else ('↑' if change > 0 else '→')
        else:
            first_val = owner_data.iloc[0]['Total Failures'] if len(owner_data) > 0 else 0
            last_val = first_val
            change = 0
            pct_change = 0
            trend = '→'

        summary_data.append({
            'Owner': str(owner),
            'First Report': int(first_val),
            'Latest Report': int(last_val),
            'Change': int(change),
            '% Change': f"{pct_change:.1f}%",
            'Trend': trend
        })

    summary_df = pd.DataFrame(summary_data)

    # Stacked area chart for overall trend
    pivot_trend = trend_filtered.pivot_table(
        index='Report Date',
        columns='Owner',
        values='Total Failures',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    fig_area = go.Figure()
    for owner in top_owners:
        if owner in pivot_trend.columns:
            fig_area.add_trace(go.Scatter(
                x=pivot_trend['Report Date'],
                y=pivot_trend[owner],
                mode='lines',
                name=str(owner)[:20] + '...' if len(str(owner)) > 20 else str(owner),
                stackgroup='one',
                hovertemplate=f'{owner}<br>Failures: %{{y}}<extra></extra>'
            ))

    fig_area.update_layout(
        title='<b>Cumulative Failure Trend (Stacked Area)</b>',
        xaxis_title='Report Date',
        yaxis_title='Total Failures',
        height=400,
        hovermode='x unified',
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
    )

    return fig_trend, fig_area, summary_df


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
    st.markdown("Upload your posture report CSV to generate executive and security dashboards.")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Data")
        st.markdown("**Drag & drop multiple files** to see trends over time")
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type=['csv', 'gz', 'zip'],
            accept_multiple_files=True,
            help="Upload one or more CSV/gzipped CSV/zip files. Multiple files enable trend analysis."
        )

        if uploaded_files:
            st.success(f"Loaded {len(uploaded_files)} file(s)")
            with st.expander("View uploaded files"):
                for f in uploaded_files:
                    date = extract_date_from_filename(f.name)
                    st.text(f"  {f.name[:40]}... ({date.strftime('%Y-%m-%d')})")

        st.header("Grouping Options")
        group_by = st.selectbox(
            "Group failures by",
            options=['Zones', 'Account Id'],
            index=0,
            help="Select how to group failures: by Zones (owner) or by Account Id"
        )

    if not uploaded_files:
        st.info("Please upload CSV file(s) using the sidebar to get started.")

        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("""
        1. Export your posture report(s) from Sysdig as CSV
        2. **Drag & drop** one or more files into the upload area
        3. View the generated dashboards below
        4. **Upload multiple reports** from different dates to see failure trends over time
        5. Download summary reports as needed
        """)
        return

    # Determine if we have multiple files for trend analysis
    has_multiple_files = len(uploaded_files) > 1

    # Use the most recent file for single-file analysis
    sorted_files = sorted(uploaded_files, key=lambda f: extract_date_from_filename(f.name), reverse=True)
    latest_file = sorted_files[0]

    # Load and process data for the latest file
    with st.spinner("Loading and processing data..."):
        try:
            df_full, df = load_data(latest_file)
            if has_multiple_files:
                for f in uploaded_files:
                    f.seek(0)
                trend_data = load_multiple_files(uploaded_files, group_by)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

    # Display metrics
    st.markdown("---")
    fig_pie, fig_bar, total_failures, unique_owners, unique_accounts, top_owners, owner_stats = create_executive_charts(df, group_by)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Failures", f"{total_failures:,}")
    group_label = "Unique Zones" if group_by == 'Zones' else "Unique Accounts"
    col2.metric(group_label, f"{unique_owners}")
    col3.metric("Total Accounts", f"{unique_accounts}")
    col4.metric("Reports Loaded", f"{len(uploaded_files)}")

    # Tabs for different views
    if has_multiple_files:
        tab1, tab2, tab3, tab4 = st.tabs(["Trend Analysis", "Executive Dashboard", "Security Drill-Down", "Download Reports"])
    else:
        tab1, tab2, tab3 = st.tabs(["Executive Dashboard", "Security Drill-Down", "Download Reports"])

    if has_multiple_files:
        with tab1:
            st.markdown("### Failure Trend Analysis")
            st.markdown(f"Analyzing **{len(uploaded_files)} reports** to track failures over time.")

            fig_trend, fig_area, summary_df = create_trend_charts(trend_data)

            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)

                st.markdown("---")
                st.markdown("### Progress Summary")
                st.markdown("**Goal:** See failure counts decrease over time (negative change = improvement)")

                def highlight_trend(val):
                    if val == '↓':
                        return 'color: green; font-weight: bold'
                    elif val == '↑':
                        return 'color: red; font-weight: bold'
                    return ''

                st.dataframe(
                    summary_df.style.applymap(highlight_trend, subset=['Trend']),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("---")
                st.markdown("### Cumulative View")
                st.plotly_chart(fig_area, use_container_width=True)

        exec_tab = tab2
        security_tab = tab3
        download_tab = tab4
    else:
        exec_tab = tab1
        security_tab = tab2
        download_tab = tab3

    with exec_tab:
        st.markdown("### Executive Summary: Who Should We Engage First?")
        st.markdown(f"*Showing data from: {latest_file.name}*")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("### Top 5 Contributors - What Controls to Fix First")

        st.markdown("""
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <span><span style="background:#e74c3c; padding: 2px 10px; border-radius: 4px; color: white;">High</span></span>
            <span><span style="background:#f39c12; padding: 2px 10px; border-radius: 4px; color: white;">Medium</span></span>
            <span><span style="background:#3498db; padding: 2px 10px; border-radius: 4px; color: white;">Low</span></span>
            <span><span style="background:#95a5a6; padding: 2px 10px; border-radius: 4px; color: white;">Info</span></span>
        </div>
        """, unsafe_allow_html=True)

        person_charts = create_person_charts(df, top_owners, group_by)

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
