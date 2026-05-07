"""Load legacy v1 Sysdig CSV exports (gzipped, different column names)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb


def _dbg(msg: str) -> None:
    print(f"[legacy-loader] {msg}", file=sys.stderr, flush=True)


# v1 column → v2 column mapping
_V1_TO_V2 = {
    "Vulnerability ID": "vulnerability_name",
    "Severity": "vulnerability_severity",
    "Package name": "package_name",
    "Package version": "package_version",
    "Package type": "package_type",
    "Package path": "package_path",
    "Image": "image_name",
    "OS Name": "os_name",
    "CVSS version": "cvss_version",
    "CVSS score": "cvss_score",
    "Vuln Publish date": "disclosure_date",
    "Vuln Fix date": "fix_available_date",
    "Fix version": "fix_version",
    "Public Exploit": "public_exploit",
    "K8S cluster name": "kubernetes_cluster_name",
    "K8S namespace name": "kubernetes_namespace_name",
    "K8S workload type": "kubernetes_workload_type",
    "K8S workload name": "kubernetes_workload_name",
    "K8S container name": "kubernetes_container_name",
    "Image ID": "image_id",
    "In use": "package_in_use",
    "Risk accepted": "risk_accepted",
    "CISA KEV Publish date": "cisa_kev_publish_date",
    "CISA KEV Due date": "cisa_kev_due_date",
    "CISA KEV Known Ransomware": "cisa_kev_known_ransomware",
}

# Columns we need but v1 doesn't have — synthesize
_SYNTHETIC_COLS = {
    'fix_available': "CASE WHEN \"fix_version\" IS NOT NULL AND \"fix_version\" != '' THEN true ELSE false END",
    "agent_tags": "''",
    "container_labels": "''",
    "namespace_labels": "''",
}

_BOOL_COLS = {
    "public_exploit", "package_in_use", "risk_accepted",
    "cisa_kev_known_ransomware", "fix_available",
}

_DATE_COLS = {
    "disclosure_date", "fix_available_date",
    "cisa_kev_publish_date", "cisa_kev_due_date",
}


def load_legacy_csv(conn: duckdb.DuckDBPyConnection, csv_path: Path,
                    severities: list[str] | None = None) -> tuple[str, int]:
    """Load a legacy v1 CSV (gz or plain) into the v2 staging table.

    - Maps v1 column names to v2 names
    - Filters by severity if specified (default: Critical, High)
    - Synthesizes missing columns (fix_available, labels)
    - Streams from gzip, no intermediate file

    Returns (table_name, row_count).
    """
    csv_path = csv_path.resolve()
    temp_table = "_ingest_staging"
    t0 = time.monotonic()
    _ms = lambda: int((time.monotonic() - t0) * 1000)

    sev_filter = ""
    if severities:
        sev_list = ", ".join(f"'{s}'" for s in severities)
        sev_filter = f'WHERE "Severity" IN ({sev_list})'

    _dbg(f"Loading legacy CSV ({csv_path.name}), severities={severities or 'all'}")

    # Build SELECT: map v1→v2 columns + synthetic columns
    select_parts = []
    for v1_col, v2_col in _V1_TO_V2.items():
        select_parts.append(f'"{v1_col}" AS "{v2_col}"')

    for v2_col, expr in _SYNTHETIC_COLS.items():
        select_parts.append(f'{expr} AS "{v2_col}"')

    select_cols = ",\n        ".join(select_parts)

    # Apply type casts
    cast_parts = []
    all_v2_cols = list(_V1_TO_V2.values()) + list(_SYNTHETIC_COLS.keys())
    for c in all_v2_cols:
        if c in _BOOL_COLS:
            cast_parts.append(
                f'CASE WHEN "{c}" IS NULL THEN FALSE '
                f'ELSE "{c}"::BOOLEAN END AS "{c}"'
            )
        elif c in _DATE_COLS:
            cast_parts.append(f'TRY_CAST("{c}" AS TIMESTAMPTZ) AS "{c}"')
        else:
            cast_parts.append(f'"{c}"')

    cast_cols = ",\n        ".join(cast_parts)

    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")

    sql = f"""
        CREATE TEMPORARY TABLE {temp_table} AS
        SELECT {cast_cols}
        FROM (
            SELECT {select_cols}
            FROM read_csv_auto('{csv_path}')
            {sev_filter}
        )
    """

    _dbg(f"Creating staging table... {_ms()}ms")
    conn.execute(sql)

    count = conn.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
    _dbg(f"Loaded {count:,} rows in {_ms()}ms")

    return temp_table, count
