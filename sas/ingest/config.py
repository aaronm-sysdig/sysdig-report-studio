"""SAS runtime configuration. Env-var driven; sensible defaults for local dev."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "sas.duckdb"

    @property
    def ownership_mapping_path(self) -> Path:
        return self.data_dir / "ownership.csv"

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Read config from env. SAS_DATA_DIR overrides default ~/sysdig-vuln-data."""
    env_dir = os.environ.get("SAS_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    else:
        data_dir = Path.home() / "sysdig-vuln-data"
    return Config(data_dir=data_dir)
