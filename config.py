"""
Shared configuration for Sysdig Report Studio.

Region mappings and API helpers used across all modules.
"""
from pathlib import Path
import streamlit as st

# Region to hostname mapping
SYSDIG_REGIONS = {
    "APJ": "app.au1.sysdig.com",
    "US East": "secure.sysdig.com",
    "EU": "eu1.app.sysdig.com",
    "EU North": "app.eu2.sysdig.com",
    "US West": "us2.app.sysdig.com",
    "India": "app.in1.sysdig.com",
    "US West (GCP)": "app.us4.sysdig.com",
    "ME Central": "app.me2.sysdig.com"
}

# Directory for storing registry vulnerability snapshots
VULN_DATA_DIR = Path.home() / "sysdig-vuln-data"


def get_sysdig_host(region: str) -> str:
    """Get the Sysdig API hostname for a given region."""
    return SYSDIG_REGIONS.get(region, f"app.{region}.sysdig.com")


def get_api_config() -> tuple[str, str]:
    """
    Return (api_token, base_url) from global session state.

    All modules call this instead of reading env vars or managing their own
    sidebar auth inputs. Returns empty strings if not yet configured.
    """
    token = st.session_state.get("global_api_token", "")
    region = st.session_state.get("global_region", "APJ")
    host = get_sysdig_host(region)
    base_url = f"https://{host}"
    return token, base_url
