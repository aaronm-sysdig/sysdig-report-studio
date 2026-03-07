"""
Shared configuration for Sysdig Report Studio.

Region mappings and API helpers used across all modules.
"""
from __future__ import annotations
from pathlib import Path
import streamlit as st

# Region to hostname mapping
SYSDIG_REGIONS = {
    "US East (North Virginia)":    "secure.sysdig.com",
    "US West (Oregon, AWS)":       "app.us2.sysdig.com",
    "US-3":                        "app.us3.sysdig.com",
    "US West (Dallas, GCP)":       "app.us4.sysdig.com",
    "EU Central (Frankfurt)":      "eu1.app.sysdig.com",
    "EU North (Stockholm)":        "app.eu2.sysdig.com",
    "Asia Pacific (Sydney)":       "app.au1.sysdig.com",
    "Middle East (Dammam, GCP)":   "app.me2.sysdig.com",
    "Asia Pacific South (Mumbai)": "app.in1.sysdig.com",
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
    region = st.session_state.get("global_region", "Asia Pacific (Sydney)")
    host = get_sysdig_host(region)
    base_url = f"https://{host}"
    return token, base_url
