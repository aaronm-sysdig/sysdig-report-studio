"""
Shared configuration for Sysdig Report Studio.

Region mappings and other settings that need to be consistent across modules.
"""

# Region to hostname mapping - add new regions here as needed
# Some regions use app.<region>.sysdig.com, others use <region>.app.sysdig.com
SYSDIG_REGIONS = {
    "au1": "app.au1.sysdig.com",
    "us1": "secure.sysdig.com"
}


def get_sysdig_host(region: str) -> str:
    """Get the Sysdig API hostname for a given region."""
    return SYSDIG_REGIONS.get(region, f"app.{region}.sysdig.com")
