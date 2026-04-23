import pytest
from sas.ingest.csv_validator import (
    validate_csv_columns,
    CSVSchemaError,
    EXPECTED_COLUMNS,
)


def test_expected_columns_has_30_entries():
    assert len(EXPECTED_COLUMNS) == 30


def test_valid_csv_passes(fixtures_dir):
    validate_csv_columns(fixtures_dir / "minimal_valid.csv")


def test_malformed_csv_raises_with_missing_columns(fixtures_dir):
    with pytest.raises(CSVSchemaError) as exc:
        validate_csv_columns(fixtures_dir / "malformed.csv")
    assert "missing columns" in str(exc.value).lower()


def test_real_sample_csv_passes():
    from pathlib import Path
    repo_root = Path(__file__).parent.parent
    sample = repo_root / "phoenix-vuln-findings-2026_04_23.csv"
    if sample.exists():
        validate_csv_columns(sample)
