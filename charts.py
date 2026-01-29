"""
Chart rendering bits for the report studio.

This module handles all the Plotly chart creation. It's shared between the live
preview and PDF generation so everything looks consistent. If you change how a
chart looks here, it'll update everywhere.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Severity color mapping - consistent across all charts
SEVERITY_COLORS = {
    "Critical": "#d32f2f", "High": "#ffa000", "Medium": "#388e3c",
    "Low": "#1976d2", "Negligible": "#757575", "Informational": "#00bcd4"
}

# Non-severity color palette
DEFAULT_COLORS = ["#7b1fa2", "#00796b", "#c2185b", "#512da8", "#0097a7",
                  "#689f38", "#5d4037", "#455a64", "#f57c00", "#303f9f"]

# Traffic Light configuration
TRAFFIC_LIGHT_CONFIG = {
    "height": 200,
    "label_font_size": 24,
    "value_font_size": 56,
    "label_y_position": 0.80,
    "value_y_position": 0.35,
    "box_padding": 0.01,
}


def get_color_map_for_data(df: pd.DataFrame, cat_col: str) -> dict:
    """Returns severity colors if data contains severity values, else empty dict."""
    if cat_col not in df.columns:
        return {}
    data_values = set(df[cat_col].unique())
    if data_values & set(SEVERITY_COLORS.keys()):
        return SEVERITY_COLORS
    return {}


def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Detect category and value columns from DataFrame."""
    columns = df.columns.tolist()
    cat_col = None
    val_col = None

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if val_col is None:
                val_col = col
        else:
            if cat_col is None:
                cat_col = col

    # Fallback
    if cat_col is None and len(columns) > 0:
        cat_col = columns[0]
    if val_col is None and len(columns) > 1:
        val_col = columns[1]

    return cat_col, val_col


def create_chart_figure(
    df: pd.DataFrame,
    chart_type: str,
    title: str,
    for_pdf: bool = False,
    show_legend: bool = True,
    chart_height: int | None = None
) -> go.Figure | None:
    """
    Create a Plotly figure for the given chart type.

    Args:
        df: Data to chart
        chart_type: Type of chart (history, traffic_lights, vertical_bar, horizontal_bar, pie_chart)
        title: Chart title
        for_pdf: If True, use light theme suitable for PDF output
        show_legend: If True, show the legend (applies to bar charts)
        chart_height: Optional height in pixels for the chart

    Returns:
        Plotly Figure or None for unsupported types (like table)
    """
    if df.empty:
        return None

    columns = df.columns.tolist()
    template = "plotly_white" if for_pdf else "plotly_dark"

    cat_col, val_col = detect_columns(df)
    color_map = get_color_map_for_data(df, cat_col) if cat_col else {}

    if chart_type == "history":
        if not all(col in columns for col in ["timestamp", "value", "vuln_severity"]):
            return None
        color_map = get_color_map_for_data(df, "vuln_severity")
        fig = px.line(df, x="timestamp", y="value", color="vuln_severity",
                      color_discrete_map=color_map, template=template, markers=True,
                      title=title if not for_pdf else None)
        if for_pdf:
            fig.update_layout(margin=dict(l=40, r=40, t=20, b=40),
                              legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        return fig

    elif chart_type == "traffic_lights":
        cfg = TRAFFIC_LIGHT_CONFIG
        num_items = len(df)

        # Handle single-column data (values only, no labels)
        single_column = len(df.columns) == 1
        if single_column:
            val_col = df.columns[0]
            cat_col = None

        fig = go.Figure()
        padding = cfg["box_padding"]
        box_width = (1.0 - padding * (num_items + 1)) / num_items

        for i in range(num_items):
            row_data = df.iloc[i]
            category = "" if single_column else str(row_data[cat_col])
            value = int(row_data[val_col])

            if category in SEVERITY_COLORS:
                bg_color = SEVERITY_COLORS[category]
            else:
                bg_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            x0 = padding + i * (box_width + padding)
            x1 = x0 + box_width
            x_center = (x0 + x1) / 2

            # Rounded rectangle
            r = 0.02
            path = f"M {x0+r},0 L {x1-r},0 Q {x1},0 {x1},{r} L {x1},{1-r} Q {x1},1 {x1-r},1 L {x0+r},1 Q {x0},1 {x0},{1-r} L {x0},{r} Q {x0},0 {x0+r},0 Z"
            fig.add_shape(type="path", path=path, xref="paper", yref="paper",
                          fillcolor=bg_color, layer="below", line={"width": 0})

            # Label (only if we have categories)
            if category:
                fig.add_annotation(
                    x=x_center, y=cfg["label_y_position"],
                    xref="paper", yref="paper",
                    text=f"<b>{category}</b>",
                    showarrow=False,
                    font={"size": cfg["label_font_size"], "color": "white", "family": "Arial Black"},
                    xanchor="center", yanchor="middle"
                )

            # Value - center vertically if no label
            value_y = 0.5 if single_column else cfg["value_y_position"]
            fig.add_annotation(
                x=x_center, y=value_y,
                xref="paper", yref="paper",
                text=f"{value:,}",
                showarrow=False,
                font={"size": cfg["value_font_size"], "color": "white", "family": "Arial"},
                xanchor="center", yanchor="middle"
            )

        fig.update_layout(
            paper_bgcolor="white" if for_pdf else "rgba(0,0,0,0)",
            plot_bgcolor="white" if for_pdf else "rgba(0,0,0,0)",
            height=cfg["height"],
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=False,
            xaxis={"visible": False, "range": [0, 1]},
            yaxis={"visible": False, "range": [0, 1]}
        )
        return fig

    elif chart_type == "vertical_bar":
        fig = px.bar(df, x=cat_col, y=val_col, color=cat_col,
                     color_discrete_map=color_map, template=template,
                     title=title if not for_pdf else None)
        layout_opts = {"showlegend": show_legend}
        if chart_height:
            layout_opts["height"] = chart_height
        if for_pdf:
            layout_opts["showlegend"] = False
            layout_opts["margin"] = dict(l=40, r=40, t=20, b=40)
        fig.update_layout(**layout_opts)
        return fig

    elif chart_type == "horizontal_bar":
        fig = px.bar(df, x=val_col, y=cat_col, orientation='h', color=cat_col,
                     color_discrete_map=color_map, template=template,
                     title=title if not for_pdf else None)
        layout_opts = {"showlegend": show_legend}
        if chart_height:
            layout_opts["height"] = chart_height
        if for_pdf:
            layout_opts["showlegend"] = False
            layout_opts["margin"] = dict(l=40, r=40, t=20, b=40)
        fig.update_layout(**layout_opts)
        return fig

    elif chart_type == "pie_chart":
        total = df[val_col].sum()
        labels = []
        for _, row in df.iterrows():
            cat = row[cat_col]
            val = int(row[val_col])
            pct = (val / total * 100) if total > 0 else 0
            labels.append(f"{cat} ({val:,}) - {pct:.1f}%")

        fig = px.pie(df, names=labels, values=val_col,
                     color=cat_col, color_discrete_map=color_map, template=template,
                     title=title if not for_pdf else None)
        fig.update_traces(textinfo='label', textposition='outside')
        if for_pdf:
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        return fig

    elif chart_type == "table":
        return None  # Tables are handled separately

    return None


def chart_to_image_bytes(
    df: pd.DataFrame,
    chart_type: str,
    title: str,
    width: int = 1200,
    chart_height: int | None = None
) -> bytes | None:
    """
    Create a chart and return it as PNG bytes for PDF embedding.

    Args:
        df: Data to chart
        chart_type: Type of chart
        title: Chart title
        width: Image width in pixels
        chart_height: Optional explicit height in pixels

    Returns:
        PNG image bytes or None
    """
    fig = create_chart_figure(df, chart_type, title, for_pdf=True, chart_height=chart_height)
    if fig is None:
        return None

    # Determine height based on chart type (or use provided height)
    if chart_height:
        height = chart_height
    elif chart_type == "traffic_lights":
        height = TRAFFIC_LIGHT_CONFIG["height"]
    elif chart_type == "history":
        height = int(width * 0.45)
    else:
        height = int(width * 0.5)

    return fig.to_image(format="png", width=width, height=height, scale=2)


def get_chart_dimensions(chart_type: str, width: int = 1200) -> tuple[int, int]:
    """Get the dimensions for a chart type."""
    if chart_type == "traffic_lights":
        return width, TRAFFIC_LIGHT_CONFIG["height"]
    elif chart_type in ("history", "trend_summary"):
        return width, int(width * 0.45)
    else:
        return width, int(width * 0.5)


def calculate_trend_insights(df: pd.DataFrame) -> list[dict]:
    """
    Calculate trend insights from vulnerability history data.

    Compares the earliest and latest values for each severity level.

    Args:
        df: DataFrame with columns: timestamp, value, vuln_severity

    Returns:
        List of dicts with: severity, start_value, end_value, change, pct_change, trend
    """
    if df.empty or not all(col in df.columns for col in ["timestamp", "value", "vuln_severity"]):
        return []

    # Convert timestamp to datetime for proper sorting
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    insights = []
    severity_order = ["Critical", "High", "Medium", "Low", "Negligible", "Informational"]

    for severity in severity_order:
        sev_data = df[df['vuln_severity'] == severity].sort_values('timestamp')

        if len(sev_data) < 2:
            continue

        start_value = int(sev_data.iloc[0]['value'])
        end_value = int(sev_data.iloc[-1]['value'])
        change = end_value - start_value

        if start_value > 0:
            pct_change = (change / start_value) * 100
        else:
            pct_change = 100.0 if end_value > 0 else 0.0

        # Determine trend: for vulnerabilities, DOWN is good (green), UP is bad (red)
        if change < 0:
            trend = "improved"  # Fewer vulns = good
        elif change > 0:
            trend = "worsened"  # More vulns = bad
        else:
            trend = "unchanged"

        insights.append({
            "severity": severity,
            "start_value": start_value,
            "end_value": end_value,
            "change": change,
            "pct_change": pct_change,
            "trend": trend
        })

    return insights


def format_trend_change(change: int, pct_change: float, trend: str) -> tuple[str, str]:
    """
    Format the change value with arrow indicator.

    Returns:
        Tuple of (formatted_text, trend_type) for styling
    """
    sign = "+" if change > 0 else ""
    pct_str = f"{pct_change:+.1f}%"

    if trend == "improved":
        arrow = "↓"
        return f"{sign}{change:,} ({pct_str}) {arrow}", "improved"
    elif trend == "worsened":
        arrow = "↑"
        return f"{sign}{change:,} ({pct_str}) {arrow}", "worsened"
    else:
        return f"{change:,} (0%)", "unchanged"


def create_insights_dataframe(insights: list[dict]) -> pd.DataFrame:
    """Create a formatted DataFrame from insights for display."""
    if not insights:
        return pd.DataFrame()

    rows = []
    for item in insights:
        change_text, _ = format_trend_change(
            item['change'], item['pct_change'], item['trend']
        )
        rows.append({
            "Severity": item['severity'],
            "31 Days Ago": f"{item['start_value']:,}",
            "Today": f"{item['end_value']:,}",
            "Change": change_text
        })

    return pd.DataFrame(rows)
