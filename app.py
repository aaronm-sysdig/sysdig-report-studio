"""
Sysdig Report Studio — Main entry point.

Navigation and global config only. All page logic lives in dedicated modules.
"""
import streamlit as st

from config import SYSDIG_REGIONS
import posture_analytics
import report_studio
import bullish_runtime_vulns.page as bullish_page

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sysdig Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — copied verbatim from original app.py ────────────────────────────────
st.markdown("""
<style>
    /* Main content area - keep space for toolbar */
    .block-container { padding-top: 2rem !important; }
    /* Remove sidebar header and add padding for logo */
    [data-testid="stSidebarHeader"] { display: none !important; }
    [data-testid="stSidebarContent"] { padding-top: 2rem !important; }
    [data-testid="stSidebar"] hr { margin: 0.5rem 0; }
    /* Tighter headings */
    h1 { padding-top: 0 !important; padding-bottom: 0.5rem !important; margin-top: 0 !important; margin-bottom: 0.5rem !important; }
    h2, h3 { margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
    /* Tighter expanders and containers */
    .streamlit-expanderHeader { padding: 0.5rem 0 !important; }
    .streamlit-expanderContent { padding-top: 0.5rem !important; }
    [data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    .stCaption { margin-bottom: 0.25rem !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# Initialise customer name default on first run
if 'global_cust_name' not in st.session_state:
    st.session_state['global_cust_name'] = 'Acme Corp'

# Apply any pending customer name update (set by report_studio when loading a report for editing)
if '_pending_cust_name' in st.session_state:
    st.session_state['global_cust_name'] = st.session_state.pop('_pending_cust_name')

# ── Global config sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo1.png", width=300)
    st.divider()

    st.header("Global Config")
    region = st.selectbox("Sysdig Region", list(SYSDIG_REGIONS.keys()), key="global_region")
    api_token = st.text_input("API Token", type="password", key="global_api_token")
    cust_name = st.text_input("Customer Name", key="global_cust_name")

    st.divider()

    tool = st.radio(
        "Select Tool",
        options=[
            "📋  Posture Analytics",
            "📄  Report Studio",
            "🔥  Runtime Vulnerabilities",
        ],
        index=0,
    )

    st.divider()

# ── Route to selected tool ────────────────────────────────────────────────────
if tool == "📋  Posture Analytics":
    posture_analytics.render_page()
elif tool == "🔥  Runtime Vulnerabilities":
    bullish_page.render_page(api_token, region)
else:
    report_studio.render_sidebar(api_token, region)
    report_studio.render_page(api_token, region, cust_name)
