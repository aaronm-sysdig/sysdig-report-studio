"""
Shared configuration for Sysdig Report Studio.

Region mappings and other settings that need to be consistent across modules.
"""

# Region to hostname mapping - add new regions here as needed
# Some regions use app.<region>.sysdig.com, others use <region>.app.sysdig.com
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


def get_sysdig_host(region: str) -> str:
    """Get the Sysdig API hostname for a given region."""
    return SYSDIG_REGIONS.get(region, f"app.{region}.sysdig.com")
