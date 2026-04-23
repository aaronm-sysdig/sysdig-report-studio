"""
Posture Delta Computation for Migration Tracking.

Compares two posture snapshots and computes what changed:
new failures, resolved failures, persistent failures, and breakdowns.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def compute_delta(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """
    Compare snapshot A (before) to snapshot B (after).

    Returns a dict with:
        - summary: net change, counts by category
        - new_failures: controls failing in B but not A
        - resolved: controls failing in A but not B
        - persistent: controls failing in both
        - by_severity: delta broken down by severity
        - by_zone: delta broken down by zone
        - by_account: delta broken down by account
    """
    fail_a = df_a[df_a['Result'] == 'Fail'].copy()
    fail_b = df_b[df_b['Result'] == 'Fail'].copy()

    # Create a composite key for matching: Control ID + Resource ID + Account Id
    key_cols = ['Control ID', 'Resource ID', 'Account Id']
    for df in (fail_a, fail_b):
        df['_key'] = df[key_cols].astype(str).agg('|'.join, axis=1)

    keys_a = set(fail_a['_key'])
    keys_b = set(fail_b['_key'])

    new_keys = keys_b - keys_a
    resolved_keys = keys_a - keys_b
    persistent_keys = keys_a & keys_b

    new_failures = fail_b[fail_b['_key'].isin(new_keys)].drop(columns='_key')
    resolved = fail_a[fail_a['_key'].isin(resolved_keys)].drop(columns='_key')
    persistent = fail_b[fail_b['_key'].isin(persistent_keys)].drop(columns='_key')

    # Severity breakdown
    sev_a = fail_a['Control Severity'].value_counts().to_dict()
    sev_b = fail_b['Control Severity'].value_counts().to_dict()
    all_sevs = sorted(set(list(sev_a.keys()) + list(sev_b.keys())),
                      key=lambda s: {'High': 0, 'Medium': 1, 'Low': 2, 'Info': 3}.get(s, 4))
    by_severity = {s: {'before': sev_a.get(s, 0), 'after': sev_b.get(s, 0),
                        'delta': sev_b.get(s, 0) - sev_a.get(s, 0)} for s in all_sevs}

    # Zone breakdown
    zone_a = fail_a.groupby('Zones').size()
    zone_b = fail_b.groupby('Zones').size()
    all_zones = sorted(set(zone_a.index) | set(zone_b.index))
    by_zone = {z: {'before': int(zone_a.get(z, 0)), 'after': int(zone_b.get(z, 0)),
                    'delta': int(zone_b.get(z, 0)) - int(zone_a.get(z, 0))} for z in all_zones}

    # Account breakdown
    acct_a = fail_a.groupby('Account Id').size()
    acct_b = fail_b.groupby('Account Id').size()
    all_accts = sorted(set(acct_a.index) | set(acct_b.index), key=str)
    by_account = {a: {'before': int(acct_a.get(a, 0)), 'after': int(acct_b.get(a, 0)),
                       'delta': int(acct_b.get(a, 0)) - int(acct_a.get(a, 0))} for a in all_accts}

    return {
        'summary': {
            'before_total': len(fail_a),
            'after_total': len(fail_b),
            'net_change': len(fail_b) - len(fail_a),
            'new_failures': len(new_failures),
            'resolved': len(resolved),
            'persistent': len(persistent),
        },
        'new_failures': new_failures,
        'resolved': resolved,
        'persistent': persistent,
        'by_severity': by_severity,
        'by_zone': by_zone,
        'by_account': by_account,
    }


def chart_severity_delta(by_severity: dict) -> go.Figure:
    """Grouped bar chart showing before/after failure counts by severity."""
    sevs = list(by_severity.keys())
    before = [by_severity[s]['before'] for s in sevs]
    after = [by_severity[s]['after'] for s in sevs]
    deltas = [by_severity[s]['delta'] for s in sevs]

    sev_colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#3498db', 'Info': '#95a5a6'}

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Before', x=sevs, y=before,
                         marker_color=[sev_colors.get(s, '#999') for s in sevs],
                         opacity=0.4, text=before, textposition='outside'))
    fig.add_trace(go.Bar(name='After', x=sevs, y=after,
                         marker_color=[sev_colors.get(s, '#999') for s in sevs],
                         text=after, textposition='outside'))

    # Add delta annotations
    for i, (s, d) in enumerate(zip(sevs, deltas)):
        color = '#27ae60' if d < 0 else '#e74c3c' if d > 0 else '#999'
        sign = '+' if d > 0 else ''
        fig.add_annotation(x=s, y=max(before[i], after[i]),
                           text=f"<b>{sign}{d}</b>", showarrow=False,
                           yshift=25, font=dict(color=color, size=14))

    fig.update_layout(
        barmode='group',
        title=dict(text='<b>Failure Delta by Severity</b>', x=0.5),
        height=400,
        yaxis=dict(title='Failure Count', rangemode='tozero'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='#fafafa',
    )
    return fig


def chart_zone_delta(by_zone: dict, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart showing delta per zone, sorted by magnitude."""
    items = sorted(by_zone.items(), key=lambda x: abs(x[1]['delta']), reverse=True)[:top_n]
    items.reverse()  # bottom-to-top for horizontal bar

    zones = [z for z, _ in items]
    deltas = [v['delta'] for _, v in items]
    colors = ['#27ae60' if d < 0 else '#e74c3c' if d > 0 else '#95a5a6' for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas, y=zones, orientation='h',
        marker_color=colors,
        text=[f"{'+' if d > 0 else ''}{d}" for d in deltas],
        textposition='outside',
    ))
    fig.update_layout(
        title=dict(text='<b>Failure Delta by Zone</b>', x=0.5),
        height=max(300, len(items) * 30 + 100),
        xaxis=dict(title='Change in Failures', zeroline=True,
                   zerolinecolor='#333', zerolinewidth=2),
        yaxis=dict(type='category'),
        plot_bgcolor='#fafafa',
    )
    return fig


def chart_trend(snapshots: list[dict]) -> go.Figure:
    """Line chart of total failures across all snapshots over time."""
    labels = [s['label'] for s in snapshots]
    totals = [s['total_failures'] for s in snapshots]
    highs = [s['high_failures'] for s in snapshots]
    mediums = [s['medium_failures'] for s in snapshots]
    lows = [s['low_failures'] for s in snapshots]
    dates = [s['created_at'][:10] for s in snapshots]

    x_labels = [f"{l}\n({d})" for l, d in zip(labels, dates)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=totals, name='Total',
                             mode='lines+markers+text', text=totals,
                             textposition='top center',
                             line=dict(color='#2c3e50', width=3),
                             marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=x_labels, y=highs, name='High',
                             mode='lines+markers',
                             line=dict(color='#e74c3c', width=2, dash='dot'),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=x_labels, y=mediums, name='Medium',
                             mode='lines+markers',
                             line=dict(color='#f39c12', width=2, dash='dot'),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=x_labels, y=lows, name='Low',
                             mode='lines+markers',
                             line=dict(color='#3498db', width=2, dash='dot'),
                             marker=dict(size=7)))

    fig.update_layout(
        title=dict(text='<b>Posture Failure Trend Across Migration</b>', x=0.5),
        height=450,
        xaxis=dict(title='Snapshot'),
        yaxis=dict(title='Failure Count', rangemode='tozero'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='#fafafa',
        hovermode='x unified',
    )
    return fig
