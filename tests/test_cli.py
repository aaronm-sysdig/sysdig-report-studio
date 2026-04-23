import subprocess
import sys
from pathlib import Path
import pytest


def test_cli_help_shows_ingest_command(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "sas.ingest", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cli_ingests_minimal_csv_into_fresh_db(tmp_path, fixtures_dir, monkeypatch):
    monkeypatch.setenv("SAS_DATA_DIR", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "sas.ingest",
         str(fixtures_dir / "minimal_valid.csv")],
        capture_output=True, text=True,
        env={**__import__("os").environ, "SAS_DATA_DIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    # DB file should exist
    assert (tmp_path / "sas.duckdb").exists()


def test_cli_second_run_reports_idempotent(tmp_path, fixtures_dir, monkeypatch):
    import os
    env = {**os.environ, "SAS_DATA_DIR": str(tmp_path)}
    cmd = [sys.executable, "-m", "sas.ingest", str(fixtures_dir / "minimal_valid.csv")]
    subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r2.returncode == 0
    assert "already ingested" in r2.stdout.lower() or "idempotent" in r2.stdout.lower() \
        or "skipping" in r2.stdout.lower()
