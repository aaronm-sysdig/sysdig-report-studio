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
from charts import create_chart_figure
from config import SYSDIG_REGIONS, get_sysdig_host

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Sysdig Report Studio")
LOGO_URL = "logo1.png"

# Default values - change these for your environment
DEFAULT_CUSTOMER_NAME = "Acme Corp"
DEFAULT_API_TOKEN = ""  # Leave empty for security, or set for local dev


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


def fetch_vulnerability_history(region: str, api_token: str, days: int) -> tuple[list[dict] | None, str | None]:
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

    payload = {
        "id": "011",
        "zoneIds": [],
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
        response = requests.post(url, json=payload, headers=headers, timeout=30)
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
def render_chart(df: pd.DataFrame, chart_type: str, chart_key: str):
    """Render a chart in Streamlit using the shared chart creation module."""
    if df.empty:
        st.warning("No data to display")
        return

    if chart_type == "table":
        formatted_df = format_datetime_columns(df)
        st.dataframe(formatted_df, hide_index=True, use_container_width=True)
        return

    # Pass empty title since we show it via st.subheader() above
    fig = create_chart_figure(df, chart_type, "", for_pdf=False)

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
    v_type = st.selectbox("Chart Type",
                          ["Vulnerability History", "Traffic Lights", "Vertical Bar", "Horizontal Bar", "Pie Chart",
                           "Table"])

    # Clear preview when chart type changes
    if 'last_chart_type' not in st.session_state:
        st.session_state['last_chart_type'] = v_type
    elif st.session_state['last_chart_type'] != v_type:
        st.session_state['last_chart_type'] = v_type
        st.session_state.pop('preview_data', None)
        st.session_state.pop('fetch_error', None)

    b_title = st.text_input("Element Title", value=f"Analysis of {v_type}")
    b_description = st.text_area("Description (optional)", value="", height=68)

    if v_type == "Vulnerability History":
        history_days = st.number_input("Days of History", min_value=1, max_value=31, value=7)
        config_params = {"type": "history", "id": "011", "days": history_days, "markers": True}
        sysql_query = None
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
            raw_data, error = fetch_vulnerability_history(region, api_token, history_days)
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
        render_chart(preview_df, chart_type, f"designer_{chart_type}")

        opt_col, _ = st.columns([1, 2])
        with opt_col:
            with st.expander("Element Options", expanded=True):
                element_width = st.radio(
                    "Width",
                    ["Full Width", "Half Width"],
                    key="element_width_select",
                    horizontal=True
                )

            st.write("")
            if st.button("➕ Add to Report", type="primary"):
                config_to_save = st.session_state['active_config'].copy()
                config_to_save['title'] = b_title
                st.session_state['report_blocks'].append({
                    "config": config_to_save,
                    "snapshot": st.session_state['preview_data'],
                    "width": "half" if element_width == "Half Width" else "full"
                })
                st.toast("Added to Report Preview!")
    else:
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

        render_chart(block_df, conf['type'], f"report_{conf['type']}_{idx}")


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
