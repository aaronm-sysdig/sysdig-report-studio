import pandas as pd
from sas.ingest.csv_loader import load_csv


def test_load_csv_returns_dataframe_with_normalized_columns(fixtures_dir):
    df = load_csv(fixtures_dir / "minimal_valid.csv")
    assert isinstance(df, pd.DataFrame)
    assert "vulnerability_name" in df.columns
    assert "image_id" in df.columns
    assert "package_in_use" in df.columns
    assert len(df) == 1


def test_load_csv_parses_booleans(fixtures_dir):
    df = load_csv(fixtures_dir / "minimal_valid.csv")
    row = df.iloc[0]
    assert row["package_in_use"] is True or row["package_in_use"] == True  # noqa
    assert row["risk_accepted"] is False or row["risk_accepted"] == False  # noqa
    assert row["public_exploit"] is False or row["public_exploit"] == False  # noqa
    assert row["fix_available"] is True or row["fix_available"] == True  # noqa


def test_load_csv_parses_dates(fixtures_dir):
    df = load_csv(fixtures_dir / "minimal_valid.csv")
    row = df.iloc[0]
    assert pd.notna(row["disclosure_date"])
    assert pd.notna(row["fix_available_date"])


def test_load_csv_handles_empty_date_strings(fixtures_dir):
    df = load_csv(fixtures_dir / "minimal_valid.csv")
    row = df.iloc[0]
    # CISA KEV fields are empty strings in the fixture → should become NaT
    assert pd.isna(row["cisa_kev_publish_date"])
