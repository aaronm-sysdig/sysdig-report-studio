"""
Posture & compliance data handling for Sysdig Report Studio.

Parses posture report CSVs and aggregates failure data for charting.
Structured so the CSV loader can later be replaced with direct API calls
without changing the aggregation or chart logic.
"""
import gzip
import zipfile

import pandas as pd


def load_posture_csv(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a posture report CSV/GZ/ZIP and return full + fail-only DataFrames.

    Expected CSV columns:
        Result, Control ID, Control Name, Control Severity,
        Zones, Account Id, Account Name, Resource Name, Resource ID

    Args:
        uploaded_file: Streamlit UploadedFile or file-like object

    Returns:
        Tuple of (full_df, fail_only_df)
    """
    filename = getattr(uploaded_file, 'name', '')

    if filename.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            csv_files = [f for f in z.namelist()
                         if f.endswith('.csv') or f.endswith('.csv.gz')]
            if not csv_files:
                raise ValueError("No CSV files found in the zip archive")
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

    df_fail = df[df['Result'] == 'Fail'].copy()
    return df, df_fail


def aggregate_posture_data(df_fail: pd.DataFrame, group_by: str = 'Zones') -> dict:
    """Aggregate posture failure data for charting.

    Args:
        df_fail: DataFrame filtered to failing controls only
        group_by: Column to group by ('Zones' or 'Account Id')

    Returns:
        Dict of DataFrames keyed by chart type:
            - owner_stats: Owner, Total Failures, Percentage
            - posture_failures: same as owner_stats (for donut chart)
            - posture_top_contributors: same as owner_stats (for bar chart)
            - posture_severity_by_owner: Owner, Control Severity, Count
            - posture_heatmap: Owner, Control Name, Count
    """
    total_failures = len(df_fail)

    # Owner stats - aggregated failure counts
    owner_stats = df_fail.groupby(group_by).agg(
        total_failures=('Control ID', 'count'),
        unique_controls=('Control Name', 'nunique')
    ).reset_index()
    owner_stats.columns = ['Owner', 'Total Failures', 'Unique Controls']
    if total_failures > 0:
        owner_stats['Percentage'] = (
            owner_stats['Total Failures'] / total_failures * 100
        ).round(1)
    else:
        owner_stats['Percentage'] = 0.0
    owner_stats = owner_stats.sort_values('Total Failures', ascending=False)

    # Severity breakdown per owner (top 10 owners)
    top_10_owners = owner_stats.head(10)['Owner'].tolist()
    severity_data = df_fail[
        df_fail[group_by].isin(top_10_owners)
    ].groupby(
        [group_by, 'Control Severity']
    ).size().reset_index(name='Count')
    severity_data.columns = ['Owner', 'Control Severity', 'Count']

    # Heatmap data: top 20 owners x top 15 controls
    top_20_owners = owner_stats.head(20)['Owner'].tolist()
    top_15_controls = df_fail['Control Name'].value_counts().head(15).index.tolist()

    heatmap_subset = df_fail[
        (df_fail[group_by].isin(top_20_owners)) &
        (df_fail['Control Name'].isin(top_15_controls))
    ]
    heatmap_data = heatmap_subset.groupby(
        [group_by, 'Control Name']
    ).size().reset_index(name='Count')
    heatmap_data.columns = ['Owner', 'Control Name', 'Count']

    return {
        "owner_stats": owner_stats,
        "posture_failures": owner_stats[['Owner', 'Total Failures', 'Percentage']].copy(),
        "posture_top_contributors": owner_stats[['Owner', 'Total Failures', 'Percentage']].copy(),
        "posture_severity_by_owner": severity_data,
        "posture_heatmap": heatmap_data,
    }
