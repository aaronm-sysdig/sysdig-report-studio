"""CSV schema validation. Reject on column mismatch before any DB work."""
from __future__ import annotations

from pathlib import Path
import csv


EXPECTED_COLUMNS = [
    "Vulnerability Name",
    "Vulnerability Severity",
    "Package Name",
    "Package Version",
    "Package Type",
    "Package Path",
    "Image Name",
    "OS Name",
    "CVSS Version",
    "CVSS Score",
    "CVSS Vector",
    "Disclosure Date",
    "Fix Available Date",
    "Fix Version",
    "Public Exploit",
    "Kubernetes Cluster Name",
    "Kubernetes Namespace Name",
    "Kubernetes Workload Type",
    "Kubernetes Workload Name",
    "Kubernetes Container Name",
    "Image ID",
    "Package In Use",
    "Risk Accepted",
    "CISA KEV Publish Date",
    "CISA KEV Due Date",
    "CISA KEV Known Ransomware",
    "Fix Available",
    "Agent Tags",
    "Container Labels",
    "Namespace Labels",
]


class CSVSchemaError(ValueError):
    """Raised when a CSV's column set doesn't match the expected Sysdig export."""


def validate_csv_columns(path: Path) -> None:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise CSVSchemaError(f"CSV is empty: {path}")

    header_set = set(header)
    expected_set = set(EXPECTED_COLUMNS)
    missing = expected_set - header_set
    extra = header_set - expected_set

    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"missing columns: {sorted(missing)}")
        if extra:
            msg_parts.append(f"unexpected columns: {sorted(extra)}")
        raise CSVSchemaError(f"CSV schema mismatch in {path}: {'; '.join(msg_parts)}")
