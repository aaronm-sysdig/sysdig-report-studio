"""Synthetic Sysdig CSV fixture builder for scenario-based tests.

Usage:
    b = ScenarioBuilder()
    b.add_finding(vulnerability_name="CVE-2026-99999", image_id="sha256:deadbeef")
    b.write_csv(Path("fixtures/day1.csv"))
    b.clear()
    b.add_finding(vulnerability_name="CVE-2026-99999", fix_available="false")
    b.write_csv(Path("fixtures/day2.csv"))
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sas.ingest.csv_validator import EXPECTED_COLUMNS

# Map snake_case field names → CSV header strings (must match EXPECTED_COLUMNS exactly)
_FIELD_TO_HEADER: dict[str, str] = {
    "vulnerability_name": "Vulnerability Name",
    "vulnerability_severity": "Vulnerability Severity",
    "package_name": "Package Name",
    "package_version": "Package Version",
    "package_type": "Package Type",
    "package_path": "Package Path",
    "image_name": "Image Name",
    "os_name": "OS Name",
    "cvss_version": "CVSS Version",
    "cvss_score": "CVSS Score",
    "cvss_vector": "CVSS Vector",
    "disclosure_date": "Disclosure Date",
    "fix_available_date": "Fix Available Date",
    "fix_version": "Fix Version",
    "public_exploit": "Public Exploit",
    "kubernetes_cluster_name": "Kubernetes Cluster Name",
    "kubernetes_namespace_name": "Kubernetes Namespace Name",
    "kubernetes_workload_type": "Kubernetes Workload Type",
    "kubernetes_workload_name": "Kubernetes Workload Name",
    "kubernetes_container_name": "Kubernetes Container Name",
    "image_id": "Image ID",
    "package_in_use": "Package In Use",
    "risk_accepted": "Risk Accepted",
    "cisa_kev_publish_date": "CISA KEV Publish Date",
    "cisa_kev_due_date": "CISA KEV Due Date",
    "cisa_kev_known_ransomware": "CISA KEV Known Ransomware",
    "fix_available": "Fix Available",
    "agent_tags": "Agent Tags",
    "container_labels": "Container Labels",
    "namespace_labels": "Namespace Labels",
}

# Sanity-check at import time: every EXPECTED_COLUMNS header must be present in the map
_mapped_headers = set(_FIELD_TO_HEADER.values())
_missing = set(EXPECTED_COLUMNS) - _mapped_headers
if _missing:
    raise RuntimeError(f"_builder.py: unmapped EXPECTED_COLUMNS: {sorted(_missing)}")


@dataclass
class Finding:
    vulnerability_name: str = "CVE-2026-00001"
    vulnerability_severity: str = "High"
    package_name: str = "libfoo"
    package_version: str = "1.0.0"
    package_type: str = "Golang"
    package_path: str = "/usr/local/bin/foo"
    image_name: str = "registry.example.com/myapp:v1"
    os_name: str = "alpine 3.20"
    cvss_version: str = "3.1"
    cvss_score: float = 7.5
    cvss_vector: str = "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N"
    disclosure_date: str = "2026-01-01T00:00:00Z"
    fix_available_date: str = "2026-01-02T00:00:00Z"
    fix_version: str = "1.0.1"
    public_exploit: str = "false"
    kubernetes_cluster_name: str = "sysdn02"
    kubernetes_namespace_name: str = "default"
    kubernetes_workload_type: str = "Deployment"
    kubernetes_workload_name: str = "myapp"
    kubernetes_container_name: str = "main"
    image_id: str = "sha256:abc123def456"
    package_in_use: str = "true"
    risk_accepted: str = "false"
    cisa_kev_publish_date: str = ""
    cisa_kev_due_date: str = ""
    cisa_kev_known_ransomware: str = ""
    fix_available: str = "true"
    agent_tags: str = "{}"
    container_labels: str = "{}"
    namespace_labels: str = "{}"


class ScenarioBuilder:
    """Accumulates Finding rows and writes them as a Sysdig-format CSV."""

    def __init__(self) -> None:
        self._rows: list[Finding] = []

    def add_finding(self, **overrides: Any) -> "ScenarioBuilder":
        """Append a Finding row, applying any keyword overrides to the defaults."""
        self._rows.append(Finding(**overrides))
        return self

    def clear(self) -> "ScenarioBuilder":
        """Reset the row buffer."""
        self._rows = []
        return self

    def write_csv(self, path: Path) -> None:
        """Write accumulated rows to *path* using Sysdig export column order."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            for finding in self._rows:
                raw = asdict(finding)
                row = {_FIELD_TO_HEADER[k]: str(v) for k, v in raw.items()}
                writer.writerow(row)
