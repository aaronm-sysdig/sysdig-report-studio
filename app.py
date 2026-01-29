"""
Sysdig Report Studio - Main Application - Written by aaron.miles@sysdig.com

A Streamlit app for building executive reports from Sysdig SYSQL/analytics data.
Design your report with various chart types, preview it live, then generate PDFs
on demand or on a schedule. Nothing too fancy, just gets the job done.

Does have a schduler built in if you want to leave it running, but you can also
manually generate a PDF manually if you wish.
"""
import streamlit as st
import pandas as pd
import yaml
import json
import os
import requests
import urllib.parse
from datetime import datetime

# Our own modules
import database as db
from scheduler import start_scheduler, stop_scheduler, get_scheduler, run_template_now
from charts import create_chart_figure, calculate_trend_insights, create_insights_dataframe
from config import SYSDIG_REGIONS, get_sysdig_host

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Sysdig Report Studio")
LOGO_URL = "logo1.png"

# Default values - change these for your environment
DEFAULT_CUSTOMER_NAME = "Acme Corp"
DEFAULT_API_TOKEN = ""  # Leave empty for security, or set for local dev

# ==========================================================
# Widget Type Configuration - easy to swap icons later
# icon: emoji string OR path to image file (if ends with .png/.svg)
# ==========================================================
WIDGET_TYPES = [
    {"name": "Vulnerability History", "icon": "📈", "key": "history"},
    {"name": "Vulnerability Trend Summary", "icon": "📉", "key": "trend_summary"},
    {"name": "Traffic Lights", "icon": "🚦", "key": "traffic_lights"},
    {"name": "Vertical Bar", "icon": "📊", "key": "vertical_bar"},
    {"name": "Horizontal Bar", "icon": "▤", "key": "horizontal_bar"},
    {"name": "Pie Chart", "icon": "🥧", "key": "pie_chart"},
    {"name": "Table", "icon": "📋", "key": "table"},
    {"name": "Horizontal Divider", "icon": "➖", "key": "divider"},
]


# ==========================================================
# Styling tweaks to tighten up the UI and try and remove some whitespace
# Not easy so it still has more than I want it to
# ==========================================================
st.markdown(f"""
<style>
    /* Reduce top padding in main content area */
    .block-container {{ padding-top: 1rem !important; }}
    /* Reduce spacing in sidebar */
    [data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
    [data-testid="stSidebar"] hr {{ margin: 0.5rem 0; }}
    /* Tighter headings */
    h1, h2, h3 {{ margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }}
    /* Reduce expander padding */
    .streamlit-expanderHeader {{ padding: 0.5rem 0 !important; }}
    .streamlit-expanderContent {{ padding-top: 0.5rem !important; }}
    /* Tighter containers */
    [data-testid="stVerticalBlock"] > div {{ gap: 0.5rem; }}
    /* Reduce caption margins */
    .stCaption {{ margin-bottom: 0.25rem !important; }}
    /* Tighter tab content */
    .stTabs [data-baseweb="tab-panel"] {{ padding-top: 0.5rem !important; }}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# Used to fetch API data from Sysdig
# ==========================================================
def fetch_zones(region: str, api_token: str) -> tuple[list[dict] | None, str | None]:
    """Fetch available zones from the Sysdig API."""
    host = get_sysdig_host(region)
    url = f"https://{host}/platform/v1/zones"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        if 'data' in result:
            # Return list of zones with id and name
            zones = [{"id": z["id"], "name": z["name"]} for z in result['data']]
            return zones, None
        else:
            return None, "Unexpected response format"

    except requests.exceptions.HTTPError as e:
        return None, f"API Error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except Exception:
        return None, "Failed to fetch zones"


def fetch_sysql_data(region: str, api_token: str, query: str) -> tuple[list[dict] | None, str | None]:
    """Execute a SysQL query against the Sysdig API."""
    host = get_sysdig_host(region)
    base_url = f"https://{host}/api/sysql/v2/query"
    encoded_query = urllib.parse.quote(query)
    url = f"{base_url}?q={encoded_query}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        if 'items' in result:
            items = result['items']
            if 'entities' in result and items:
                allowed_fields = list(result['entities'].keys())
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


def fetch_vulnerability_history(region: str, api_token: str, days: int, zone_id: int = None) -> tuple[list[dict] | None, str | None]:
    """Fetch vulnerability history data from the Sysdig analytics API."""
    host = get_sysdig_host(region)
    url = f"https://{host}/api/platform/analytics/v1/data/query"

    # Work out the date range - end is today's midnight, start is N days before
    now = datetime.now()
    end_of_today = now.replace(hour=23, minute=59, second=59, microsecond=0)
    start_date = (end_of_today - pd.Timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

    end_epoch = int(end_of_today.timestamp())
    start_epoch = int(start_date.timestamp())

    # Get system timezone name (needs IANA format like "Australia/Melbourne")
    try:
        local_tz = datetime.now().astimezone().tzinfo
        # Try to get the proper IANA key if available
        if hasattr(local_tz, 'key'):
            tz_name = local_tz.key
        else:
            tz_name = "Etc/UTC"
    except Exception:
        tz_name = "Etc/UTC"

    # Build zone IDs list - empty means all zones
    zone_ids = [zone_id] if zone_id else []

    payload = {
        "id": "011",
        "zoneIds": zone_ids,
        "scope": [
            {"rightOperand": {"name": "StartTime", "value": str(start_epoch)}},
            {"rightOperand": {"name": "EndTime", "value": str(end_epoch)}}
        ],
        "timezone": tz_name
    }

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)  # 5 min for large date ranges
        response.raise_for_status()
        result = response.json()

        if 'data' in result:
            return result['data'], None
        else:
            return None, "Unexpected response format (no 'data' key)"

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


def format_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and format ISO datetime strings in DataFrame columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                val = str(sample.iloc[0])
                if len(val) >= 19 and val[4] == '-' and val[7] == '-' and 'T' in val:
                    try:
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%b %-d, %Y, %-I:%M %p')
                    except Exception:
                        pass
    return df


def normalize_api_data(data) -> list[dict]:
    """Normalize API response data to a list of dicts format."""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return pd.DataFrame(data).to_dict('records')
    else:
        return []


def format_display_date(iso_string: str) -> str:
    """Format an ISO date string to the system's local timezone."""
    if not iso_string:
        return ""
    try:
        from datetime import timezone

        # Parse the ISO string
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))

        # If naive (no timezone), assume it's UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to local system timezone
        local_dt = dt.astimezone(None)

        return local_dt.strftime('%d/%b/%Y %H:%M')
    except Exception:
        # Fallback if parsing fails
        return iso_string[:16] if len(iso_string) >= 16 else iso_string


# ==========================================================
# 📊 CHART RENDERING
# ==========================================================
def render_chart(df: pd.DataFrame, chart_type: str, chart_key: str,
                 show_legend: bool = True, chart_height: int | None = None):
    """Render a chart in Streamlit using the shared chart creation module."""
    # Handle divider specially - it doesn't need data
    if chart_type == "divider":
        st.divider()
        return

    if df.empty:
        st.warning("No data to display")
        return

    if chart_type == "table":
        formatted_df = format_datetime_columns(df)
        st.dataframe(formatted_df, hide_index=True, use_container_width=True)
        st.caption(f"*Displaying {len(df)} records*")
        return

    if chart_type == "trend_summary":
        # Render the history line chart
        fig = create_chart_figure(df, "history", "", for_pdf=False)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_chart")

        # Calculate and render insights table
        insights = calculate_trend_insights(df)
        if insights:
            st.markdown("#### Trend Analysis")
            insights_df = create_insights_dataframe(insights)

            # Style the dataframe with colored arrows
            def style_change(val):
                if "↓" in str(val):
                    return "color: #2e7d32; font-weight: bold"  # Green
                elif "↑" in str(val):
                    return "color: #c62828; font-weight: bold"  # Red
                return ""

            styled_df = insights_df.style.applymap(
                style_change, subset=["Change"]
            )
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
        return

    # Pass empty title since we show it via st.subheader() above
    fig = create_chart_figure(df, chart_type, "", for_pdf=False,
                              show_legend=show_legend, chart_height=chart_height)

    if fig is None:
        st.error(f"Could not create chart for type: {chart_type}")
        return

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


# ==========================================================
# 2. SIDEBAR
# ==========================================================
with st.sidebar:
    st.image(LOGO_URL, width=300)
    st.header("Global Config")
    cust_name = st.text_input(
        "Customer Name",
        value=st.session_state.get('editing_customer_name', DEFAULT_CUSTOMER_NAME)
    )
    region = st.selectbox("Sysdig Region", list(SYSDIG_REGIONS.keys()))
    api_token = st.text_input("API Token", type="password", value=DEFAULT_API_TOKEN)

    st.divider()
    st.header("Element Designer")

    # Initialize selected widget type in session state
    if 'selected_widget_type' not in st.session_state:
        st.session_state['selected_widget_type'] = WIDGET_TYPES[0]['name']

    # Icon grid for widget type selection (4 columns, 2 rows)
    st.caption("Widget Type")

    # Row 1
    cols1 = st.columns(4)
    for col_idx, widget in enumerate(WIDGET_TYPES[:4]):
        with cols1[col_idx]:
            is_selected = st.session_state['selected_widget_type'] == widget['name']
            btn_type = "primary" if is_selected else "secondary"
            if st.button(widget['icon'], key=f"widget_btn_{widget['key']}",
                         help=widget['name'], use_container_width=True, type=btn_type):
                st.session_state['selected_widget_type'] = widget['name']
                st.session_state.pop('preview_data', None)
                st.session_state.pop('fetch_error', None)
                st.rerun()

    # Row 2
    cols2 = st.columns(4)
    for col_idx, widget in enumerate(WIDGET_TYPES[4:8]):
        with cols2[col_idx]:
            is_selected = st.session_state['selected_widget_type'] == widget['name']
            btn_type = "primary" if is_selected else "secondary"
            if st.button(widget['icon'], key=f"widget_btn_{widget['key']}",
                         help=widget['name'], use_container_width=True, type=btn_type):
                st.session_state['selected_widget_type'] = widget['name']
                st.session_state.pop('preview_data', None)
                st.session_state.pop('fetch_error', None)
                st.rerun()

    v_type = st.session_state['selected_widget_type']
    st.caption(f"*Selected: {v_type}*")

    # Horizontal divider doesn't need title/description
    if v_type == "Horizontal Divider":
        b_title = ""
        b_description = ""
    else:
        b_title = st.text_input("Element Title", value=f"Analysis of {v_type}")
        b_description = st.text_area("Description (optional)", value="", height=68)

    # Zone selector for vulnerability history widgets
    selected_zone_id = None
    if v_type in ["Vulnerability History", "Vulnerability Trend Summary"]:
        # Fetch zones if we have an API token and haven't already
        if api_token:
            cache_key = f"zones_{region}"
            if cache_key not in st.session_state:
                zones, error = fetch_zones(region, api_token)
                if zones:
                    st.session_state[cache_key] = zones
                else:
                    st.session_state[cache_key] = []

            zones = st.session_state.get(cache_key, [])
            if zones:
                zone_options = {z["name"]: z["id"] for z in zones}
                zone_names = ["All Zones"] + list(zone_options.keys())
                selected_zone_name = st.selectbox("Zone", zone_names)
                if selected_zone_name != "All Zones":
                    selected_zone_id = zone_options[selected_zone_name]

    if v_type == "Horizontal Divider":
        st.caption("*A horizontal line to separate sections and reset half-width alignment*")
        config_params = {"type": "divider"}
        sysql_query = None
        # Direct add button for divider (no fetch needed)
        if st.button("➕ Add Divider to Report", use_container_width=True, type="primary"):
            st.session_state['report_blocks'].append({
                "config": config_params.copy(),
                "snapshot": [],
                "width": "full"  # Dividers are always full width
            })
            st.toast("Divider added to report!")
        fetch_clicked = False
    elif v_type == "Vulnerability History":
        history_days = st.number_input("Days of History", min_value=1, max_value=31, value=7)
        config_params = {"type": "history", "id": "011", "days": history_days, "markers": True, "zone_id": selected_zone_id}
        sysql_query = None
        fetch_clicked = st.button("🚀 Fetch & Preview", use_container_width=True)
    elif v_type == "Vulnerability Trend Summary":
        # Hardcoded to 31 days for trend analysis
        st.caption("*Analyses 31 days of vulnerability data with change insights*")
        config_params = {"type": "trend_summary", "id": "011", "days": 31, "zone_id": selected_zone_id}
        sysql_query = None
        fetch_clicked = st.button("🚀 Fetch & Preview", use_container_width=True)
    else:
        default_q = "MATCH KubeWorkload AS k AFFECTED_BY Vulnerability AS v RETURN v.severity AS Severity, count(v) AS Vulnerabilities ORDER BY Severity DESC LIMIT 4;"
        sysql_query = st.text_area("SysQL Query", value=default_q, height=240)
        config_params = {"type": v_type.lower().replace(" ", "_"), "query": sysql_query}
        fetch_clicked = st.button("🚀 Fetch & Preview", use_container_width=True)

    st.divider()
    if st.button("🗑️ Clear Template", use_container_width=True):
        st.session_state['report_blocks'] = []
        st.session_state.pop('editing_template_id', None)
        st.session_state.pop('editing_template_name', None)
        st.session_state.pop('editing_customer_name', None)
        st.toast("Template Cleared!")

# ==========================================================
# 3. DATA FETCHING
# ==========================================================
if fetch_clicked:
    st.session_state.pop('preview_data', None)
    st.session_state.pop('fetch_error', None)

    if v_type == "Vulnerability History":
        if not api_token:
            st.session_state['fetch_error'] = "API Token is required"
        else:
            raw_data, error = fetch_vulnerability_history(region, api_token, history_days, selected_zone_id)
            if error:
                st.session_state['fetch_error'] = error
            elif raw_data:
                st.session_state['preview_data'] = raw_data
            else:
                st.session_state['fetch_error'] = "No data returned from query"
    elif v_type == "Vulnerability Trend Summary":
        if not api_token:
            st.session_state['fetch_error'] = "API Token is required"
        else:
            raw_data, error = fetch_vulnerability_history(region, api_token, 31, selected_zone_id)
            if error:
                st.session_state['fetch_error'] = error
            elif raw_data:
                st.session_state['preview_data'] = raw_data
            else:
                st.session_state['fetch_error'] = "No data returned from query"
    else:
        if not api_token:
            st.session_state['fetch_error'] = "API Token is required"
        elif not sysql_query:
            st.session_state['fetch_error'] = "SysQL Query is required"
        else:
            raw_data, error = fetch_sysql_data(region, api_token, sysql_query)
            if error:
                st.session_state['fetch_error'] = error
            elif raw_data:
                st.session_state['preview_data'] = normalize_api_data(raw_data)
            else:
                st.session_state['fetch_error'] = "No data returned from query"

    st.session_state['active_config'] = {**config_params, "title": b_title, "description": b_description}

# ==========================================================
# 4. MAIN STAGE
# ==========================================================
st.markdown('<h1 style="margin:0;">Sysdig Report Studio</h1>', unsafe_allow_html=True)

if 'report_blocks' not in st.session_state:
    st.session_state['report_blocks'] = []

tab_design, tab_preview, tab_reports = st.tabs([
    "🎨 Element Designer", "📋 Report Preview", "📁 Reports"
])

with tab_design:
    if 'fetch_error' in st.session_state:
        st.error(st.session_state['fetch_error'])

    if 'preview_data' in st.session_state:
        preview_df = pd.DataFrame(st.session_state['preview_data'])
        chart_type = st.session_state['active_config']['type']

        st.subheader(b_title)
        if b_description:
            st.caption(f"*{b_description}*")

        # Get current options for live preview (use defaults before options are set)
        preview_show_legend = st.session_state.get('element_show_legend', True)
        preview_height = st.session_state.get('element_height', None)
        render_chart(preview_df, chart_type, f"designer_{chart_type}",
                     show_legend=preview_show_legend, chart_height=preview_height)

        opt_col, _ = st.columns([1, 2])
        with opt_col:
            with st.expander("Element Options", expanded=True):
                element_width = st.radio(
                    "Width",
                    ["Full Width", "Half Width"],
                    key="element_width_select",
                    horizontal=True
                )

                # Show legend option for bar charts
                chart_type = st.session_state['active_config']['type']
                show_legend = True
                element_height = None
                if chart_type in ["vertical_bar", "horizontal_bar"]:
                    show_legend = st.checkbox("Show Legend", value=True, key="element_show_legend")
                    element_height = st.slider(
                        "Chart Height",
                        min_value=200,
                        max_value=800,
                        value=400,
                        step=50,
                        key="element_height",
                        help="Height of the chart in pixels"
                    )

            st.write("")
            if st.button("➕ Add to Report", type="primary"):
                config_to_save = st.session_state['active_config'].copy()
                config_to_save['title'] = b_title
                config_to_save['show_legend'] = show_legend
                if element_height:
                    config_to_save['chart_height'] = element_height
                st.session_state['report_blocks'].append({
                    "config": config_to_save,
                    "snapshot": st.session_state['preview_data'],
                    "width": "half" if element_width == "Half Width" else "full"
                })
                st.toast("Added to Report Preview!")
    elif v_type != "Horizontal Divider":
        st.info("Enter a SysQL query and click 'Fetch & Preview' to see your data.")

# ==========================================================
# 5. REPORT PREVIEW
# ==========================================================


def move_block(from_idx: int, to_idx: int):
    move_block_blocks = st.session_state['report_blocks']
    if 0 <= from_idx < len(move_block_blocks) and 0 <= to_idx < len(move_block_blocks):
        move_block_block = move_block_blocks.pop(from_idx)
        move_block_blocks.insert(to_idx, move_block_block)


def delete_block(idx: int):
    if 0 <= idx < len(st.session_state['report_blocks']):
        st.session_state['report_blocks'].pop(idx)


def _render_block_with_controls(idx: int, block: dict):
    conf = block['config']
    block_df = pd.DataFrame(block['snapshot'])
    width = block.get('width', 'full')
    width_icon = "◧" if width == 'half' else "▢"

    # Dividers get simplified rendering
    if conf['type'] == 'divider':
        div_col1, div_col2 = st.columns([6, 1])
        with div_col1:
            st.divider()
        with div_col2:
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("⬆", key=f"blk_up_{idx}", help="Move up", disabled=(idx == 0)):
                    move_block(idx, idx - 1)
                    st.rerun()
            with btn_cols[1]:
                if st.button("⬇", key=f"blk_down_{idx}", help="Move down",
                             disabled=(idx >= len(st.session_state['report_blocks']) - 1)):
                    move_block(idx, idx + 1)
                    st.rerun()
            with btn_cols[2]:
                if st.button("✕", key=f"blk_del_{idx}", help="Delete"):
                    delete_block(idx)
                    st.rerun()
        return

    with st.container(border=True):
        header_col, controls_col = st.columns([4, 1])
        with header_col:
            st.subheader(conf['title'])
            if conf.get('description'):
                st.caption(f"*{conf['description']}*")
        with controls_col:
            btn_cols = st.columns(4)
            with btn_cols[0]:
                if st.button("⬆", key=f"blk_up_{idx}", help="Move up", disabled=(idx == 0)):
                    move_block(idx, idx - 1)
                    st.rerun()
            with btn_cols[1]:
                if st.button("⬇", key=f"blk_down_{idx}", help="Move down",
                             disabled=(idx >= len(st.session_state['report_blocks']) - 1)):
                    move_block(idx, idx + 1)
                    st.rerun()
            with btn_cols[2]:
                new_width = "full" if width == "half" else "half"
                if st.button(width_icon, key=f"blk_width_{idx}", help=f"Toggle to {new_width} width"):
                    st.session_state['report_blocks'][idx]['width'] = new_width
                    st.rerun()
            with btn_cols[3]:
                if st.button("✕", key=f"blk_del_{idx}", help="Delete"):
                    delete_block(idx)
                    st.rerun()

        render_chart(block_df, conf['type'], f"report_{conf['type']}_{idx}",
                     show_legend=conf.get('show_legend', True),
                     chart_height=conf.get('chart_height'))


with tab_preview:
    if not st.session_state['report_blocks']:
        st.info("No elements added yet. Design an element in the first tab and click 'Add to Report'.")
    else:
        # Show editing status
        editing_id = st.session_state.get('editing_template_id')
        if editing_id:
            template = db.get_template(editing_id)
            if template:
                st.info(f"✏️ Editing: **{template['name']}**")
                if template.get('schedule_enabled'):
                    st.warning("⚠️ This report has an active schedule. Saving will update the scheduled report.")
        else:
            st.caption("Creating a new report")

        # Save Report section
        with st.expander("💾 Save Report", expanded=True):
            save_col1, save_col2 = st.columns([3, 1])
            with save_col1:
                report_name = st.text_input(
                    "Report Name",
                    value=st.session_state.get('editing_template_name', f"{cust_name} Report"),
                    key="save_report_name"
                )
            with save_col2:
                orientation = st.selectbox(
                    "Orientation",
                    ["portrait", "landscape"],
                    index=0,
                    key="save_orientation"
                )

            if st.button("💾 Save Report", type="primary", use_container_width=True):
                # Build config
                yaml_blocks = []
                for block in st.session_state['report_blocks']:
                    block_config = block['config'].copy()
                    block_config['width'] = block.get('width', 'full')
                    yaml_blocks.append(block_config)

                config = {
                    "global": {"customer": cust_name, "region": region, "orientation": orientation, "logo_path": LOGO_URL},
                    "template_blocks": yaml_blocks
                }
                config_yaml = yaml.dump(config, sort_keys=False)

                if editing_id:
                    # Update existing
                    db.update_template(editing_id, name=report_name, config_yaml=config_yaml)
                    st.success(f"Report '{report_name}' updated!")
                else:
                    # Create new
                    new_id = db.create_template(name=report_name, config_yaml=config_yaml)
                    st.session_state['editing_template_id'] = new_id
                    st.session_state['editing_template_name'] = report_name
                    st.success(f"Report '{report_name}' created!")

        st.divider()

        # Render blocks
        blocks = st.session_state['report_blocks']
        idx = 0
        while idx < len(blocks):
            block = blocks[idx]
            width = block.get('width', 'full')

            if width == 'half' and idx + 1 < len(blocks) and blocks[idx + 1].get('width', 'full') == 'half':
                col1, col2 = st.columns(2)
                with col1:
                    _render_block_with_controls(idx, block)
                with col2:
                    _render_block_with_controls(idx + 1, blocks[idx + 1])
                idx += 2
            else:
                _render_block_with_controls(idx, block)
                idx += 1

# ==========================================================
# 6. REPORTS TAB (Report-centric hub)
# ==========================================================
with tab_reports:
    st.subheader("Saved Reports")

    templates = db.get_all_templates()

    if not templates:
        st.info("No reports saved yet. Design a report in the Report Preview tab and save it.")
    else:
        for tmpl in templates:
            with st.container(border=True):
                col_info, col_schedule, col_actions = st.columns([2, 2, 1])

                with col_info:
                    st.markdown(f"### {tmpl['name']}")
                    st.caption(f"Created: {format_display_date(tmpl['created_at'])}")

                    # Show last generated report
                    latest = db.get_latest_report(tmpl['id'])
                    if latest:
                        status_icon = "✅" if latest['status'] == 'success' else "❌"
                        st.caption(f"Last generated: {status_icon} {format_display_date(latest['generated_at'])}")

                with col_schedule:
                    if tmpl.get('schedule_enabled'):
                        freq = tmpl.get('schedule_frequency', '').capitalize()
                        hour = tmpl.get('schedule_hour', 0)
                        tz = tmpl.get('schedule_timezone', 'UTC')

                        if freq == 'Weekly' and tmpl.get('schedule_day_of_week') is not None:
                            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                            freq += f" ({days[tmpl['schedule_day_of_week']]})"
                        elif freq == 'Monthly' and tmpl.get('schedule_day_of_month'):
                            freq += f" (Day {tmpl['schedule_day_of_month']})"

                        st.success(f"⏰ {freq} at {hour:02d}:00 {tz}")

                        if tmpl.get('schedule_last_run'):
                            st.caption(f"Last run: {format_display_date(tmpl['schedule_last_run'])}")
                    else:
                        st.caption("📅 No schedule configured")

                with col_actions:
                    # Edit button
                    if st.button("✏️ Edit", key=f"tmpl_edit_{tmpl['id']}", use_container_width=True):
                        if not api_token:
                            st.error("API token required to load report data")
                        else:
                            with st.spinner("Loading report data..."):
                                # Load template into preview
                                config = yaml.safe_load(tmpl['config_yaml'])
                                tmpl_region = config.get('global', {}).get('region', region)
                                blocks = []
                                for block_cfg in config.get('template_blocks', []):
                                    # Fetch data for this block
                                    if block_cfg.get('type') == 'history':
                                        if os.path.exists('vuln-history.json'):
                                            with open('vuln-history.json') as f:
                                                snapshot = json.load(f).get('data', [])
                                        else:
                                            snapshot = []
                                    else:
                                        query = block_cfg.get('query', '')
                                        if query:
                                            data, _ = fetch_sysql_data(tmpl_region, api_token, query)
                                            snapshot = normalize_api_data(data) if data else []
                                        else:
                                            snapshot = []
                                    blocks.append({
                                        "config": block_cfg,
                                        "snapshot": snapshot,
                                        "width": block_cfg.get('width', 'full')
                                    })
                                st.session_state['report_blocks'] = blocks
                                st.session_state['editing_template_id'] = tmpl['id']
                                st.session_state['editing_template_name'] = tmpl['name']
                                st.session_state['editing_customer_name'] = config.get('global', {}).get('customer', DEFAULT_CUSTOMER_NAME)
                            st.toast(f"Loaded '{tmpl['name']}' for editing. Go to Report Preview tab.")
                            st.rerun()

                    # Generate Now button
                    if st.button("▶️ Generate", key=f"tmpl_run_{tmpl['id']}", use_container_width=True):
                        if not api_token:
                            st.error("API token required")
                        else:
                            with st.spinner("Generating..."):
                                success, error = run_template_now(tmpl['id'], api_token)
                                if success:
                                    st.success("Generated!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed: {error}")

                    # Delete button
                    if st.button("🗑️ Delete", key=f"tmpl_del_{tmpl['id']}", use_container_width=True):
                        db.delete_template(tmpl['id'])
                        st.rerun()

                # Schedule configuration expander
                with st.expander("📅 Schedule Settings"):
                    sched_col1, sched_col2 = st.columns(2)

                    with sched_col1:
                        sched_enabled = st.checkbox(
                            "Enable Schedule",
                            value=bool(tmpl.get('schedule_enabled')),
                            key=f"sched_enabled_{tmpl['id']}"
                        )
                        sched_freq = st.selectbox(
                            "Frequency",
                            ["daily", "weekly", "monthly"],
                            index=["daily", "weekly", "monthly"].index(tmpl.get('schedule_frequency') or 'daily'),
                            key=f"sched_freq_{tmpl['id']}"
                        )
                        sched_hour = st.selectbox(
                            "Hour",
                            list(range(24)),
                            index=tmpl.get('schedule_hour') or 0,
                            format_func=lambda x: f"{x:02d}:00",
                            key=f"sched_hour_{tmpl['id']}"
                        )

                    with sched_col2:
                        sched_dow = None
                        sched_dom = None

                        if sched_freq == "weekly":
                            sched_dow = st.selectbox(
                                "Day of Week",
                                list(range(7)),
                                index=tmpl.get('schedule_day_of_week') or 0,
                                format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                                key=f"sched_dow_{tmpl['id']}"
                            )
                        elif sched_freq == "monthly":
                            sched_dom = st.selectbox(
                                "Day of Month",
                                list(range(1, 29)),
                                index=(tmpl.get('schedule_day_of_month') or 1) - 1,
                                key=f"sched_dom_{tmpl['id']}"
                            )

                        sched_tz = st.text_input(
                            "Timezone",
                            value=tmpl.get('schedule_timezone') or 'Australia/Sydney',
                            key=f"sched_tz_{tmpl['id']}"
                        )
                        sched_retain = st.number_input(
                            "Retain Count",
                            min_value=1, max_value=100,
                            value=tmpl.get('schedule_retain_count') or 10,
                            key=f"sched_retain_{tmpl['id']}"
                        )

                    if st.button("💾 Save Schedule", key=f"save_sched_{tmpl['id']}"):
                        db.update_template_schedule(
                            tmpl['id'],
                            enabled=sched_enabled,
                            frequency=sched_freq,
                            hour=sched_hour,
                            day_of_week=sched_dow,
                            day_of_month=sched_dom,
                            timezone=sched_tz,
                            retain_count=sched_retain
                        )
                        st.success("Schedule saved!")
                        st.rerun()

                # Generated reports expander
                with st.expander("📥 Generated Reports"):
                    reports = db.get_reports_for_template(tmpl['id'])
                    if not reports:
                        st.caption("No reports generated yet")
                    else:
                        for report in reports[:5]:  # Show last 5
                            rep_col1, rep_col2 = st.columns([4, 1])
                            with rep_col1:
                                status_icon = "✅" if report['status'] == 'success' else "❌"
                                st.caption(f"{status_icon} {report['filename']} - {format_display_date(report['generated_at'])}")
                            with rep_col2:
                                if report['status'] == 'success' and os.path.exists(report['file_path']):
                                    with open(report['file_path'], 'rb') as f:
                                        st.download_button(
                                            "📥",
                                            data=f.read(),
                                            file_name=report['filename'],
                                            mime="application/pdf",
                                            key=f"dl_{report['id']}"
                                        )

    # Admin section
    st.divider()
    with st.expander("🔧 Admin Tools"):
        # Scheduler controls
        st.subheader("Scheduler")
        scheduler = get_scheduler()
        sched_col1, sched_col2 = st.columns([3, 1])
        with sched_col1:
            if scheduler and scheduler.is_running():
                st.success("Scheduler is running", icon="✅")
            else:
                st.warning("Scheduler is stopped", icon="⏸️")
        with sched_col2:
            if scheduler and scheduler.is_running():
                if st.button("⏹️ Stop", use_container_width=True, key="stop_sched"):
                    stop_scheduler()
                    st.rerun()
            else:
                if api_token:
                    if st.button("▶️ Start", use_container_width=True, key="start_sched"):
                        start_scheduler(api_token=api_token, check_interval_minutes=60)
                        st.rerun()
                else:
                    st.caption("Need API token")

        st.divider()

        # Database info
        st.subheader("Database")
        st.caption(f"Path: {db.get_database_path()}")
        st.caption(f"Reports: {db.get_reports_directory()}")

        col_admin1, col_admin2 = st.columns(2)
        with col_admin1:
            if os.path.exists(db.get_database_path()):
                with open(db.get_database_path(), 'rb') as f:
                    st.download_button("📥 Download DB", data=f.read(),
                                       file_name="sysdig_reports.db", mime="application/octet-stream")

        with col_admin2:
            uploaded_db = st.file_uploader("Upload DB", type=['db'], key="db_upload")
            if uploaded_db:
                with open(db.get_database_path(), 'wb') as f:
                    f.write(uploaded_db.read())
                st.success("Database uploaded!")
                st.rerun()
