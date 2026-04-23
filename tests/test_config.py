import os
from pathlib import Path
import pytest
from sas.ingest.config import Config, get_config


def test_default_config_has_home_based_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SAS_DATA_DIR", raising=False)
    cfg = get_config()
    assert cfg.data_dir == tmp_path / "sysdig-vuln-data"
    assert cfg.duckdb_path == tmp_path / "sysdig-vuln-data" / "sas.duckdb"


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SAS_DATA_DIR", str(tmp_path))
    cfg = get_config()
    assert cfg.data_dir == tmp_path
    assert cfg.duckdb_path == tmp_path / "sas.duckdb"


def test_ownership_mapping_path_derived_from_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SAS_DATA_DIR", str(tmp_path))
    cfg = get_config()
    assert cfg.ownership_mapping_path == tmp_path / "ownership.csv"


def test_ensure_data_dir_creates_missing_directory(monkeypatch, tmp_path):
    target = tmp_path / "new-dir"
    monkeypatch.setenv("SAS_DATA_DIR", str(target))
    cfg = get_config()
    cfg.ensure_data_dir()
    assert target.exists() and target.is_dir()
