"""Tests for the ScenarioBuilder CSV fixture helper."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.scenarios._builder import ScenarioBuilder
from sas.ingest.csv_validator import validate_csv_columns


def test_builder_produces_default_row(tmp_path: Path) -> None:
    """A finding added with no overrides should write one row of default values."""
    out = tmp_path / "out.csv"
    b = ScenarioBuilder()
    b.add_finding()
    b.write_csv(out)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["Vulnerability Name"] == "CVE-2026-00001"
    assert row["Vulnerability Severity"] == "High"
    assert row["Package Name"] == "libfoo"
    assert row["Image ID"] == "sha256:abc123def456"
    assert row["Kubernetes Cluster Name"] == "sysdn02"
    assert row["CVSS Score"] == "7.5"
    assert row["Fix Available"] == "true"


def test_builder_applies_overrides(tmp_path: Path) -> None:
    """Keyword overrides should replace default values in the output row."""
    out = tmp_path / "out.csv"
    b = ScenarioBuilder()
    b.add_finding(vulnerability_name="CVE-TEST", package_name="libbar", cvss_score=9.8)
    b.write_csv(out)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["Vulnerability Name"] == "CVE-TEST"
    assert row["Package Name"] == "libbar"
    assert row["CVSS Score"] == "9.8"
    # Other defaults unchanged
    assert row["Vulnerability Severity"] == "High"


def test_output_csv_passes_our_validator(tmp_path: Path) -> None:
    """CSV written by ScenarioBuilder must pass validate_csv_columns without error."""
    out = tmp_path / "valid.csv"
    b = ScenarioBuilder()
    b.add_finding()
    b.write_csv(out)

    # Should not raise CSVSchemaError
    validate_csv_columns(out)
