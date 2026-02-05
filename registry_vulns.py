"""
Registry vulnerability API client for Sysdig Report Studio.

Fetches container image vulnerability data from the Sysdig registry scanner
and normalizes it for charting. Adapted from Prakash's registry scanner module
to work with our multi-region config.
"""
import requests
import pandas as pd

from config import get_sysdig_host


def fetch_registry_results(
    region: str,
    api_token: str,
    limit: int = 100
) -> tuple[list[dict], str | None]:
    """Fetch all registry vulnerability results with cursor-based pagination.

    Args:
        region: Sysdig region key (e.g., "AU", "US East")
        api_token: Sysdig API bearer token
        limit: Page size for pagination

    Returns:
        Tuple of (results list, error message or None)
    """
    host = get_sysdig_host(region)
    url = f"https://{host}/secure/vulnerability/v1/registry-results"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    all_results = []
    cursor = None

    try:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(url, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            body = resp.json()

            data = body.get("data", [])
            all_results.extend(data)

            page_info = body.get("page", {})
            cursor = page_info.get("next")
            if not cursor:
                break

        return all_results, None

    except requests.exceptions.HTTPError as e:
        return [], f"API Error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return [], f"Request failed: {str(e)}"
    except Exception as e:
        return [], f"Unexpected error: {str(e)}"


def normalize_registry_data(results: list[dict]) -> pd.DataFrame:
    """Flatten API results into a DataFrame for charting.

    Returns DataFrame with columns:
        display_name, repository, tag, vendor, critical, high, medium, low,
        negligible, total_vulns, total_fixable, exploit_count, policy_status, in_use
    """
    rows = []
    for r in results:
        # Handle actual API fields (lowercase severity keys)
        vuln_sev = r.get("vulnTotalBySeverity",
                         r.get("vulnsBySev",
                                r.get("vulnTotalBySev", {})))
        fix_sev = r.get("fixableVulnsBySeverity",
                        r.get("fixableVulnsBySev", {}))

        crit = vuln_sev.get("critical", vuln_sev.get("Critical", 0))
        high = vuln_sev.get("high", vuln_sev.get("High", 0))
        med = vuln_sev.get("medium", vuln_sev.get("Medium", 0))
        low = vuln_sev.get("low", vuln_sev.get("Low", 0))
        neg = vuln_sev.get("negligible", vuln_sev.get("Negligible", 0))
        total_vulns = crit + high + med + low + neg

        fix_crit = fix_sev.get("critical", fix_sev.get("Critical", 0))
        fix_high = fix_sev.get("high", fix_sev.get("High", 0))
        fix_med = fix_sev.get("medium", fix_sev.get("Medium", 0))
        fix_low = fix_sev.get("low", fix_sev.get("Low", 0))
        fix_neg = fix_sev.get("negligible", fix_sev.get("Negligible", 0))
        total_fixable = fix_crit + fix_high + fix_med + fix_low + fix_neg

        pull_string = r.get("pullString", r.get("imagePullString", ""))
        parsed_repo = pull_string
        parsed_tag = ""
        if ":" in pull_string:
            parts = pull_string.rsplit(":", 1)
            parsed_repo = parts[0]
            parsed_tag = parts[1]

        name_part = parsed_repo.split("/")[-1] if "/" in parsed_repo else parsed_repo
        display_name = f"{name_part}:{parsed_tag}" if parsed_tag else name_part

        rows.append({
            "display_name": display_name,
            "repository": parsed_repo,
            "tag": parsed_tag or r.get("tag", ""),
            "vendor": r.get("vendor", ""),
            "critical": crit,
            "high": high,
            "medium": med,
            "low": low,
            "negligible": neg,
            "total_vulns": total_vulns,
            "total_fixable": total_fixable,
            "exploit_count": r.get("exploitCount", r.get("exploitableCount", 0)),
            "policy_status": r.get("policyStatus", r.get("policyEvaluation", "")),
            "in_use": r.get("inUse", False),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("total_vulns", ascending=False).reset_index(drop=True)
    return df
