# SAS Phase 1 — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python ingestion pipeline for Sysdig Analytics Studio — a CLI that reads a Sysdig vulnerability CSV export, upserts entities into DuckDB, diffs findings against prior state, computes reason codes, and rebuilds rollup aggregates. End state: `python -m sas.ingest <csv>` ingests the real `phoenix-vuln-findings-2026_04_23.csv` sample cleanly and idempotently.

**Architecture:** Python 3.14 package at `sas/ingest/`. Single DuckDB file at `~/sysdig-vuln-data/sas.duckdb`. Pure functional core (schema, diff logic, reason codes) with a thin CLI shell. pandas + pyarrow for CSV parsing, DuckDB for storage and SQL. TDD throughout — every unit of logic is test-first against SQLite-or-DuckDB in-memory fixtures.

**Tech Stack:** Python 3.14 (already in `.venv/`), DuckDB (new dep), pandas (already installed), pyarrow (new dep), pytest (already installed), PyYAML (already installed for ownership config).

**Design references:**
- Spec: [`docs/superpowers/specs/2026-04-23-sas-design.md`](../specs/2026-04-23-sas-design.md) — this plan implements §3 (architecture), §4 (data model), §5 (time semantics), §6 (ingestion pipeline).
- Sample CSV: [`phoenix-vuln-findings-2026_04_23.csv`](../../../phoenix-vuln-findings-2026_04_23.csv) (1000 rows, 30 columns — real shape).

**Collaboration note:** Tasks 1–8 can be dispatched to Sonnet 4.6 workers. Tasks 9 (reason code logic) and 12 (end-to-end integration test) benefit from Opus review between subagent runs.

---

## File Structure

Every file in this phase. One responsibility per file. Files that change together live together.

```
sas/
├── __init__.py
├── ingest/
│   ├── __init__.py
│   ├── __main__.py            # python -m sas.ingest entrypoint
│   ├── cli.py                 # Argparse, top-level orchestration
│   ├── config.py              # Paths, env vars, config loader
│   ├── schema.py              # DuckDB DDL — all CREATE TABLE statements + migrations
│   ├── csv_loader.py          # CSV → staging DataFrame via pyarrow
│   ├── csv_validator.py       # Expected column set, reject on mismatch
│   ├── snapshot.py            # snapshot_id computation, idempotency check
│   ├── entity_upsert.py       # image/cve/package/cluster/namespace/workload/repository upserts
│   ├── ownership.py           # Resolver chain (Label, MappingFile, Fallback)
│   ├── runtime_snapshot.py    # workload_runs_image_daily writes
│   ├── finding_diff.py        # The core diff: new/reseen/reopened/disappeared
│   ├── reason_code.py         # Pure logic: compute reason_code given prior state
│   ├── rollup.py              # Rebuild daily_metrics_* tables
│   └── logger.py              # ingest_log append, timing helpers
│
tests/
├── __init__.py
├── conftest.py                # Shared fixtures — in-memory DuckDB, sample data
├── fixtures/
│   ├── day1.csv               # Small hand-crafted 10-row CSV for deterministic tests
│   ├── day2_added_finding.csv # day1 + 1 new finding
│   ├── day2_resolved.csv      # day1 - 1 finding (becomes CLOSED)
│   ├── day2_reopened.csv      # day1 with previously-closed finding back
│   ├── ownership_sample.csv   # Small ownership mapping file
│   └── malformed.csv          # Missing columns, for validator test
├── test_csv_validator.py
├── test_csv_loader.py
├── test_snapshot.py
├── test_schema.py
├── test_entity_upsert.py
├── test_ownership.py
├── test_runtime_snapshot.py
├── test_finding_diff.py
├── test_reason_code.py
├── test_rollup.py
├── test_cli.py                # Argparse + orchestration smoke
└── test_integration.py        # End-to-end: ingest real sample CSV, assert invariants
```

**Why this split:** Each file is independently testable. Schema DDL sits alone so we can version it and write a migration test. `reason_code.py` is pure logic (no DuckDB) — trivially unit-testable. `finding_diff.py` is the ugliest part and isolated from everything else.

**What `__init__.py` files contain:** empty, except `sas/ingest/__init__.py` which exports the public functions (`ingest(csv_path)`, `reattribute()`).

---

## Task 1: Project scaffolding + dependencies

**Files:**
- Create: `sas/__init__.py` (empty)
- Create: `sas/ingest/__init__.py` (empty for now)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Modify: `requirements.txt`
- Create: `pytest.ini` at repo root

- [ ] **Step 1: Add new dependencies to requirements.txt**

The existing `requirements.txt` already has pandas and pyyaml. Add DuckDB and pyarrow.

File contents to append (don't overwrite):

```
duckdb>=1.0.0
pyarrow>=15.0.0
```

- [ ] **Step 2: Install dependencies into the existing venv**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: `Successfully installed duckdb-... pyarrow-...`

- [ ] **Step 3: Create pytest config**

File: `pytest.ini` (at repo root)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

- [ ] **Step 4: Create empty package files**

Files (all empty):
- `sas/__init__.py`
- `sas/ingest/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Create conftest.py with a DuckDB-in-memory fixture**

File: `tests/conftest.py`

```python
import duckdb
import pytest
from pathlib import Path


@pytest.fixture
def db():
    """Fresh in-memory DuckDB for each test."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def fixtures_dir():
    """Path to tests/fixtures/ directory."""
    return Path(__file__).parent / "fixtures"
```

- [ ] **Step 6: Create fixtures directory**

Run: `mkdir -p tests/fixtures`

- [ ] **Step 7: Verify pytest discovers the empty test tree**

Run: `.venv/bin/pytest`
Expected: `no tests ran in X.XXs` (zero tests, zero failures).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pytest.ini sas/ tests/
git commit -m "feat(sas): phase 1 scaffolding + duckdb dependency"
```

---

## Task 2: DuckDB schema

**Files:**
- Create: `sas/ingest/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_schema.py`

```python
import duckdb
import pytest
from sas.ingest.schema import create_schema, EXPECTED_TABLES


def test_create_schema_creates_all_tables(db):
    create_schema(db)
    rows = db.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    actual = [r[0] for r in rows]
    assert actual == sorted(EXPECTED_TABLES)


def test_create_schema_is_idempotent(db):
    create_schema(db)
    create_schema(db)  # should not raise
    rows = db.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchone()
    assert rows[0] == len(EXPECTED_TABLES)


def test_finding_state_has_expected_columns(db):
    create_schema(db)
    rows = db.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'finding_state' ORDER BY column_name"
    ).fetchall()
    cols = {r[0] for r in rows}
    required = {
        "finding_id", "image_id", "cve_id", "package_name",
        "package_version", "package_path", "severity", "cvss_score",
        "in_use", "fix_available", "fix_version", "risk_accepted",
        "public_exploit", "first_seen", "last_seen", "state",
        "reason_code", "closed_at", "reopened_at", "reopen_count",
        "days_open", "is_regression",
    }
    assert required.issubset(cols), f"missing: {required - cols}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sas.ingest.schema'`

- [ ] **Step 3: Write the schema module**

File: `sas/ingest/schema.py`

```python
"""DuckDB schema for Sysdig Analytics Studio. All DDL lives here.

Matches §4 of the design spec.
"""

EXPECTED_TABLES = [
    "image",
    "repository",
    "cve",
    "package",
    "cluster",
    "namespace",
    "workload",
    "team",
    "owner",
    "image_in_repository",
    "workload_runs_image_daily",
    "namespace_in_cluster",
    "workload_in_namespace",
    "workload_owned_by",
    "finding_state",
    "daily_metrics_by_image",
    "daily_metrics_by_workload",
    "daily_metrics_by_team",
    "daily_metrics_by_repository",
    "daily_metrics_by_cluster_severity",
    "ingest_log",
    "snapshot",
]


_DDL = [
    # --- Entities ---
    """
    CREATE TABLE IF NOT EXISTS image (
        image_id VARCHAR PRIMARY KEY,
        os_name VARCHAR,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        current_repository VARCHAR,
        current_tag VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS repository (
        repository VARCHAR PRIMARY KEY,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cve (
        cve_id VARCHAR PRIMARY KEY,
        disclosure_date TIMESTAMP,
        fix_available_date TIMESTAMP,
        cvss_version VARCHAR,
        initial_severity VARCHAR,
        cisa_kev_publish_date TIMESTAMP,
        cisa_kev_due_date TIMESTAMP,
        cisa_kev_known_ransomware BOOLEAN,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS package (
        package_name VARCHAR,
        package_type VARCHAR,
        PRIMARY KEY (package_name, package_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cluster (
        cluster_name VARCHAR PRIMARY KEY,
        distribution VARCHAR,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS namespace (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        PRIMARY KEY (cluster_name, namespace_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        PRIMARY KEY (cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team (
        team_id VARCHAR PRIMARY KEY,
        display_name VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS owner (
        owner_id VARCHAR PRIMARY KEY,
        display_name VARCHAR
    )
    """,
    # --- Relationships ---
    """
    CREATE TABLE IF NOT EXISTS image_in_repository (
        image_id VARCHAR,
        repository VARCHAR,
        tag VARCHAR,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        PRIMARY KEY (image_id, repository, tag)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload_runs_image_daily (
        date DATE,
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        container_name VARCHAR,
        image_id VARCHAR,
        replica_count INTEGER,
        PRIMARY KEY (date, cluster_name, namespace_name, workload_type, workload_name, container_name, image_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS namespace_in_cluster (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        PRIMARY KEY (cluster_name, namespace_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload_in_namespace (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        PRIMARY KEY (cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload_owned_by (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        team_id VARCHAR,
        owner_id VARCHAR,
        resolved_by_strategy VARCHAR,
        resolved_from VARCHAR,
        PRIMARY KEY (cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    # --- State log ---
    """
    CREATE TABLE IF NOT EXISTS finding_state (
        finding_id BIGINT PRIMARY KEY,
        image_id VARCHAR,
        cve_id VARCHAR,
        package_name VARCHAR,
        package_version VARCHAR,
        package_path VARCHAR,
        severity VARCHAR,
        cvss_score DOUBLE,
        in_use BOOLEAN,
        fix_available BOOLEAN,
        fix_version VARCHAR,
        risk_accepted BOOLEAN,
        public_exploit BOOLEAN,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        state VARCHAR,
        reason_code VARCHAR,
        closed_at TIMESTAMP,
        reopened_at TIMESTAMP,
        reopen_count INTEGER DEFAULT 0,
        days_open INTEGER,
        is_regression BOOLEAN DEFAULT FALSE
    )
    """,
    # Natural-key lookup index on finding_state
    """
    CREATE INDEX IF NOT EXISTS idx_finding_state_natural_key
    ON finding_state (image_id, cve_id, package_name, package_version, package_path)
    """,
    # --- Rollups ---
    """
    CREATE TABLE IF NOT EXISTS daily_metrics_by_image (
        date DATE,
        image_id VARCHAR,
        count_open_critical INTEGER,
        count_open_high INTEGER,
        count_open_medium INTEGER,
        count_open_low INTEGER,
        count_open INTEGER,
        count_new INTEGER,
        count_fixed_patched INTEGER,
        count_fixed_retired INTEGER,
        count_fixed_accepted INTEGER,
        count_fixed_other INTEGER,
        count_regressed INTEGER,
        mttr_sum INTEGER,
        mttr_count INTEGER,
        PRIMARY KEY (date, image_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_metrics_by_workload (
        date DATE,
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        count_open_critical INTEGER,
        count_open_high INTEGER,
        count_open_medium INTEGER,
        count_open_low INTEGER,
        count_open INTEGER,
        count_new INTEGER,
        count_fixed_patched INTEGER,
        count_fixed_retired INTEGER,
        count_fixed_accepted INTEGER,
        count_fixed_other INTEGER,
        count_regressed INTEGER,
        mttr_sum INTEGER,
        mttr_count INTEGER,
        replica_count INTEGER,
        PRIMARY KEY (date, cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_metrics_by_team (
        date DATE,
        team_id VARCHAR,
        count_open_critical INTEGER,
        count_open_high INTEGER,
        count_open_medium INTEGER,
        count_open_low INTEGER,
        count_open INTEGER,
        count_new INTEGER,
        count_fixed_patched INTEGER,
        count_fixed_retired INTEGER,
        count_fixed_accepted INTEGER,
        count_fixed_other INTEGER,
        count_regressed INTEGER,
        mttr_sum INTEGER,
        mttr_count INTEGER,
        PRIMARY KEY (date, team_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_metrics_by_repository (
        date DATE,
        repository VARCHAR,
        count_open_critical INTEGER,
        count_open_high INTEGER,
        count_open_medium INTEGER,
        count_open_low INTEGER,
        count_open INTEGER,
        count_new INTEGER,
        count_fixed_patched INTEGER,
        count_fixed_retired INTEGER,
        count_fixed_accepted INTEGER,
        count_fixed_other INTEGER,
        count_regressed INTEGER,
        PRIMARY KEY (date, repository)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_metrics_by_cluster_severity (
        date DATE,
        cluster_name VARCHAR,
        severity VARCHAR,
        count_open INTEGER,
        PRIMARY KEY (date, cluster_name, severity)
    )
    """,
    # --- Operational ---
    """
    CREATE TABLE IF NOT EXISTS snapshot (
        snapshot_id VARCHAR PRIMARY KEY,
        snapshot_at TIMESTAMP,
        source_filename VARCHAR,
        row_count INTEGER,
        ingested_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_log (
        snapshot_id VARCHAR,
        stage VARCHAR,
        rows_affected INTEGER,
        duration_ms INTEGER,
        logged_at TIMESTAMP,
        PRIMARY KEY (snapshot_id, stage)
    )
    """,
]


def create_schema(conn) -> None:
    """Create all SAS tables. Idempotent — safe to call on an existing DB."""
    for stmt in _DDL:
        conn.execute(stmt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_schema.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/schema.py tests/test_schema.py
git commit -m "feat(sas): duckdb schema for entities, state log, rollups"
```

---

## Task 3: Config loader

**Files:**
- Create: `sas/ingest/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_config.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sas.ingest.config'`

- [ ] **Step 3: Write the config module**

File: `sas/ingest/config.py`

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/config.py tests/test_config.py
git commit -m "feat(sas): config loader with SAS_DATA_DIR env override"
```

---

## Task 4: CSV validator

**Files:**
- Create: `sas/ingest/csv_validator.py`
- Create: `tests/fixtures/malformed.csv`
- Create: `tests/fixtures/minimal_valid.csv`
- Create: `tests/test_csv_validator.py`

- [ ] **Step 1: Create fixture files**

File: `tests/fixtures/minimal_valid.csv`

```
Vulnerability Name,Vulnerability Severity,Package Name,Package Version,Package Type,Package Path,Image Name,OS Name,CVSS Version,CVSS Score,CVSS Vector,Disclosure Date,Fix Available Date,Fix Version,Public Exploit,Kubernetes Cluster Name,Kubernetes Namespace Name,Kubernetes Workload Type,Kubernetes Workload Name,Kubernetes Container Name,Image ID,Package In Use,Risk Accepted,CISA KEV Publish Date,CISA KEV Due Date,CISA KEV Known Ransomware,Fix Available,Agent Tags,Container Labels,Namespace Labels
CVE-2026-00001,Critical,libfoo,1.0,OS,/usr/lib/libfoo,registry/foo:1.0,alpine 3.20,3.0,9.1,AV:N,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z,1.1,false,cluster-a,ns-a,Deployment,foo-app,foo,sha256:abc123,true,false,,,,true,{},{},{}
```

File: `tests/fixtures/malformed.csv`

```
wrong,header,set
1,2,3
```

- [ ] **Step 2: Write the failing test**

File: `tests/test_csv_validator.py`

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_csv_validator.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 4: Write the validator**

File: `sas/ingest/csv_validator.py`

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_csv_validator.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add sas/ingest/csv_validator.py tests/fixtures/minimal_valid.csv tests/fixtures/malformed.csv tests/test_csv_validator.py
git commit -m "feat(sas): csv schema validator with 30-column check"
```

---

## Task 5: CSV loader (pyarrow → normalized DataFrame)

**Files:**
- Create: `sas/ingest/csv_loader.py`
- Create: `tests/test_csv_loader.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_csv_loader.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_csv_loader.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the loader**

File: `sas/ingest/csv_loader.py`

```python
"""CSV → pandas DataFrame with column normalization and type coercion.

Column name normalization: spaces and non-alphanumerics → underscores, lowercase.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.csv as pv


_BOOL_COLS = [
    "public_exploit",
    "package_in_use",
    "risk_accepted",
    "cisa_kev_known_ransomware",
    "fix_available",
]

_DATE_COLS = [
    "disclosure_date",
    "fix_available_date",
    "cisa_kev_publish_date",
    "cisa_kev_due_date",
]


def _normalize(col: str) -> str:
    s = col.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_csv(path: Path) -> pd.DataFrame:
    """Read a Sysdig vuln CSV into a DataFrame with normalized columns and types."""
    # pyarrow is much faster than pandas for large CSVs; convert to pandas after.
    table = pv.read_csv(str(path))
    df = table.to_pandas()
    df.columns = [_normalize(c) for c in df.columns]

    for c in _BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].map(_coerce_bool)

    for c in _DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    return df


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    # Handle pandas NA / NaN / None
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "t")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_csv_loader.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/csv_loader.py tests/test_csv_loader.py
git commit -m "feat(sas): csv loader with column normalization and type coercion"
```

---

## Task 6: Snapshot identity + idempotency

**Files:**
- Create: `sas/ingest/snapshot.py`
- Create: `tests/test_snapshot.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_snapshot.py`

```python
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.snapshot import (
    compute_snapshot_id,
    extract_snapshot_at,
    is_already_ingested,
    record_snapshot,
)


def test_compute_snapshot_id_deterministic(fixtures_dir):
    a = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=1)
    b = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=1)
    assert a == b
    assert len(a) >= 16  # looks like a hash


def test_compute_snapshot_id_differs_on_row_count(fixtures_dir):
    a = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=1)
    b = compute_snapshot_id(fixtures_dir / "minimal_valid.csv", row_count=2)
    assert a != b


def test_extract_snapshot_at_from_filename():
    # phoenix-vuln-findings-2026_04_23.csv → 2026-04-23 12:00:00 UTC
    ts = extract_snapshot_at(Path("phoenix-vuln-findings-2026_04_23.csv"))
    assert ts.year == 2026 and ts.month == 4 and ts.day == 23
    assert ts.hour == 12 and ts.minute == 0


def test_extract_snapshot_at_falls_back_to_now_on_unparseable(monkeypatch):
    fixed = datetime(2026, 5, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "sas.ingest.snapshot._now_utc", lambda: fixed
    )
    ts = extract_snapshot_at(Path("no-date-in-name.csv"))
    assert ts == fixed


def test_idempotency_flow(db):
    create_schema(db)
    sid = "test-snap-001"
    assert is_already_ingested(db, sid) is False
    record_snapshot(
        db,
        snapshot_id=sid,
        snapshot_at=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
        source_filename="foo.csv",
        row_count=42,
    )
    assert is_already_ingested(db, sid) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_snapshot.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the snapshot module**

File: `sas/ingest/snapshot.py`

```python
"""Snapshot identity and idempotency. A snapshot is one CSV ingest event."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path


_FILENAME_DATE_RE = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def compute_snapshot_id(path: Path, row_count: int) -> str:
    """Deterministic ID from filename + row count. Same CSV = same ID."""
    h = hashlib.sha256()
    h.update(path.name.encode("utf-8"))
    h.update(b"|")
    h.update(str(row_count).encode("utf-8"))
    return h.hexdigest()[:32]


def extract_snapshot_at(path: Path) -> datetime:
    """Parse a YYYY_MM_DD from the filename; fall back to now() if absent.

    Convention (per spec): reports are pulled at 12:00 UTC, so anchor to 12:00.
    """
    m = _FILENAME_DATE_RE.search(path.name)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, 12, 0, tzinfo=timezone.utc)
    return _now_utc()


def is_already_ingested(conn, snapshot_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM snapshot WHERE snapshot_id = ?", [snapshot_id]
    ).fetchone()
    return row is not None


def record_snapshot(
    conn,
    *,
    snapshot_id: str,
    snapshot_at: datetime,
    source_filename: str,
    row_count: int,
) -> None:
    conn.execute(
        "INSERT INTO snapshot (snapshot_id, snapshot_at, source_filename, row_count, ingested_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [snapshot_id, snapshot_at, source_filename, row_count, _now_utc()],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_snapshot.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/snapshot.py tests/test_snapshot.py
git commit -m "feat(sas): snapshot identity and idempotency gate"
```

---

## Task 7: Entity upsert

**Files:**
- Create: `sas/ingest/entity_upsert.py`
- Create: `tests/test_entity_upsert.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_entity_upsert.py`

```python
from datetime import datetime, timezone
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.entity_upsert import upsert_entities


@pytest.fixture
def sample_row_df():
    return pd.DataFrame([{
        "vulnerability_name": "CVE-2026-00001",
        "vulnerability_severity": "Critical",
        "package_name": "libfoo",
        "package_version": "1.0",
        "package_type": "OS",
        "package_path": "/usr/lib/libfoo",
        "image_name": "registry/foo:1.0",
        "os_name": "alpine 3.20",
        "cvss_version": "3.0",
        "cvss_score": 9.1,
        "cvss_vector": "AV:N",
        "disclosure_date": pd.Timestamp("2026-01-01T00:00:00Z"),
        "fix_available_date": pd.Timestamp("2026-01-02T00:00:00Z"),
        "fix_version": "1.1",
        "public_exploit": False,
        "kubernetes_cluster_name": "cluster-a",
        "kubernetes_namespace_name": "ns-a",
        "kubernetes_workload_type": "Deployment",
        "kubernetes_workload_name": "foo-app",
        "kubernetes_container_name": "foo",
        "image_id": "sha256:abc123",
        "package_in_use": True,
        "risk_accepted": False,
        "cisa_kev_publish_date": pd.NaT,
        "cisa_kev_due_date": pd.NaT,
        "cisa_kev_known_ransomware": False,
        "fix_available": True,
        "agent_tags": "{}",
        "container_labels": "{}",
        "namespace_labels": "{}",
    }])


def test_upsert_creates_all_entity_rows(db, sample_row_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    upsert_entities(db, sample_row_df, snapshot_at)

    assert db.execute("SELECT count(*) FROM image").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM cve").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM package").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM cluster").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM namespace").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM workload").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM repository").fetchone()[0] == 1


def test_upsert_is_idempotent(db, sample_row_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    upsert_entities(db, sample_row_df, snapshot_at)
    upsert_entities(db, sample_row_df, snapshot_at)
    assert db.execute("SELECT count(*) FROM image").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM cve").fetchone()[0] == 1


def test_upsert_updates_last_seen_on_second_pass(db, sample_row_df):
    create_schema(db)
    t1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
    upsert_entities(db, sample_row_df, t1)
    upsert_entities(db, sample_row_df, t2)
    row = db.execute("SELECT first_seen, last_seen FROM image").fetchone()
    assert row[0] == t1
    assert row[1] == t2


def test_upsert_extracts_repository_and_tag_from_image_name(db, sample_row_df):
    create_schema(db)
    upsert_entities(db, sample_row_df, datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc))
    row = db.execute(
        "SELECT repository, tag FROM image_in_repository"
    ).fetchone()
    assert row[0] == "registry/foo"
    assert row[1] == "1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_entity_upsert.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the entity upsert module**

File: `sas/ingest/entity_upsert.py`

```python
"""Upsert entity tables and relationships from a normalized row DataFrame.

Handles: image, cve, package, cluster, namespace, workload, repository,
image_in_repository, namespace_in_cluster, workload_in_namespace.

All upserts use (first_seen, last_seen) semantics: first_seen set on INSERT,
last_seen updated on every seen.
"""
from __future__ import annotations

from datetime import datetime
from typing import Tuple

import pandas as pd


def _split_image_name(image_name: str) -> Tuple[str, str]:
    """Split 'registry/foo:1.0' → ('registry/foo', '1.0'). If no tag, default 'latest'."""
    if not image_name:
        return "", "latest"
    if "@" in image_name:
        # registry/foo@sha256:... → treat sha as tag-like
        repo, digest = image_name.rsplit("@", 1)
        return repo, digest
    if ":" in image_name and "/" not in image_name.rsplit(":", 1)[-1]:
        repo, tag = image_name.rsplit(":", 1)
        return repo, tag
    return image_name, "latest"


def upsert_entities(conn, df: pd.DataFrame, snapshot_at: datetime) -> None:
    """Upsert all entity rows + edges derived from the CSV frame."""
    # Prepare per-entity frames by deduplicating on keys.

    # image — key image_id
    img = df[["image_id", "os_name", "image_name"]].drop_duplicates("image_id").copy()
    img[["repository", "tag"]] = img["image_name"].apply(
        lambda s: pd.Series(_split_image_name(s))
    )

    for _, r in img.iterrows():
        conn.execute(
            """
            INSERT INTO image (image_id, os_name, first_seen, last_seen,
                               current_repository, current_tag)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (image_id) DO UPDATE SET
                last_seen = EXCLUDED.last_seen,
                current_repository = EXCLUDED.current_repository,
                current_tag = EXCLUDED.current_tag,
                os_name = COALESCE(image.os_name, EXCLUDED.os_name)
            """,
            [r["image_id"], r["os_name"], snapshot_at, snapshot_at,
             r["repository"], r["tag"]],
        )

    # repository — key repository
    repos = img[["repository"]].drop_duplicates()
    for _, r in repos.iterrows():
        conn.execute(
            """
            INSERT INTO repository (repository, first_seen, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT (repository) DO UPDATE SET last_seen = EXCLUDED.last_seen
            """,
            [r["repository"], snapshot_at, snapshot_at],
        )

    # image_in_repository edge
    for _, r in img.iterrows():
        conn.execute(
            """
            INSERT INTO image_in_repository (image_id, repository, tag, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (image_id, repository, tag) DO UPDATE SET last_seen = EXCLUDED.last_seen
            """,
            [r["image_id"], r["repository"], r["tag"], snapshot_at, snapshot_at],
        )

    # cve — key vulnerability_name
    cve_cols = [
        "vulnerability_name", "disclosure_date", "fix_available_date",
        "cvss_version", "vulnerability_severity",
        "cisa_kev_publish_date", "cisa_kev_due_date", "cisa_kev_known_ransomware",
    ]
    cves = df[cve_cols].drop_duplicates("vulnerability_name")
    for _, r in cves.iterrows():
        conn.execute(
            """
            INSERT INTO cve (cve_id, disclosure_date, fix_available_date,
                             cvss_version, initial_severity,
                             cisa_kev_publish_date, cisa_kev_due_date,
                             cisa_kev_known_ransomware, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cve_id) DO UPDATE SET
                last_seen = EXCLUDED.last_seen,
                cisa_kev_publish_date = COALESCE(cve.cisa_kev_publish_date, EXCLUDED.cisa_kev_publish_date),
                cisa_kev_due_date = COALESCE(cve.cisa_kev_due_date, EXCLUDED.cisa_kev_due_date),
                cisa_kev_known_ransomware = EXCLUDED.cisa_kev_known_ransomware
            """,
            [
                r["vulnerability_name"],
                _py_dt(r["disclosure_date"]), _py_dt(r["fix_available_date"]),
                r["cvss_version"], r["vulnerability_severity"],
                _py_dt(r["cisa_kev_publish_date"]), _py_dt(r["cisa_kev_due_date"]),
                bool(r["cisa_kev_known_ransomware"]),
                snapshot_at, snapshot_at,
            ],
        )

    # package — key (name, type)
    pkgs = df[["package_name", "package_type"]].drop_duplicates()
    for _, r in pkgs.iterrows():
        conn.execute(
            "INSERT INTO package (package_name, package_type) VALUES (?, ?) "
            "ON CONFLICT (package_name, package_type) DO NOTHING",
            [r["package_name"], r["package_type"]],
        )

    # cluster
    clusters = df[["kubernetes_cluster_name"]].drop_duplicates()
    for _, r in clusters.iterrows():
        conn.execute(
            """
            INSERT INTO cluster (cluster_name, first_seen, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT (cluster_name) DO UPDATE SET last_seen = EXCLUDED.last_seen
            """,
            [r["kubernetes_cluster_name"], snapshot_at, snapshot_at],
        )

    # namespace
    ns = df[["kubernetes_cluster_name", "kubernetes_namespace_name"]].drop_duplicates()
    for _, r in ns.iterrows():
        conn.execute(
            """
            INSERT INTO namespace (cluster_name, namespace_name, first_seen, last_seen)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (cluster_name, namespace_name) DO UPDATE SET last_seen = EXCLUDED.last_seen
            """,
            [r["kubernetes_cluster_name"], r["kubernetes_namespace_name"],
             snapshot_at, snapshot_at],
        )
        conn.execute(
            "INSERT INTO namespace_in_cluster (cluster_name, namespace_name) "
            "VALUES (?, ?) ON CONFLICT DO NOTHING",
            [r["kubernetes_cluster_name"], r["kubernetes_namespace_name"]],
        )

    # workload
    wl = df[[
        "kubernetes_cluster_name", "kubernetes_namespace_name",
        "kubernetes_workload_type", "kubernetes_workload_name",
    ]].drop_duplicates()
    for _, r in wl.iterrows():
        conn.execute(
            """
            INSERT INTO workload (cluster_name, namespace_name, workload_type, workload_name,
                                  first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET last_seen = EXCLUDED.last_seen
            """,
            [r["kubernetes_cluster_name"], r["kubernetes_namespace_name"],
             r["kubernetes_workload_type"], r["kubernetes_workload_name"],
             snapshot_at, snapshot_at],
        )
        conn.execute(
            """
            INSERT INTO workload_in_namespace (cluster_name, namespace_name,
                                               workload_type, workload_name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [r["kubernetes_cluster_name"], r["kubernetes_namespace_name"],
             r["kubernetes_workload_type"], r["kubernetes_workload_name"]],
        )


def _py_dt(v):
    """pandas Timestamp/NaT → python datetime/None for DuckDB binding."""
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_entity_upsert.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/entity_upsert.py tests/test_entity_upsert.py
git commit -m "feat(sas): entity upsert for image/cve/package/cluster/ns/workload"
```

---

## Task 8: Ownership resolver

**Files:**
- Create: `sas/ingest/ownership.py`
- Create: `tests/fixtures/ownership_sample.csv`
- Create: `tests/test_ownership.py`

- [ ] **Step 1: Create the ownership fixture**

File: `tests/fixtures/ownership_sample.csv`

```
cluster,namespace,workload_type,workload_name,team,owner
eks-corporate,platform-*,*,*,platform,aaron.miles
eks-corporate,checkout-*,*,*,checkout-team,
*,*,*,audit-service,security,
```

- [ ] **Step 2: Write the failing test**

File: `tests/test_ownership.py`

```python
import json
import pytest

from sas.ingest.ownership import (
    LabelStrategy,
    MappingFileStrategy,
    NamespaceFallback,
    ResolverChain,
    OwnershipResult,
)


def test_label_strategy_reads_team_from_namespace_labels():
    labels_json = json.dumps({
        "kubernetes.namespace.label.team": "checkout",
    })
    strat = LabelStrategy(label="team")
    r = strat.resolve(
        cluster="c", namespace="ns", workload_type="Deployment",
        workload_name="w", namespace_labels_json=labels_json,
        agent_tags_json="{}", container_labels_json="{}",
    )
    assert r == OwnershipResult(
        team_id="checkout", owner_id=None,
        resolved_by_strategy="label:team", resolved_from="namespace_labels:team",
    )


def test_label_strategy_returns_none_if_label_absent():
    strat = LabelStrategy(label="team")
    r = strat.resolve(
        cluster="c", namespace="ns", workload_type="Deployment",
        workload_name="w", namespace_labels_json="{}",
        agent_tags_json="{}", container_labels_json="{}",
    )
    assert r is None


def test_mapping_file_strategy_glob_match(fixtures_dir):
    strat = MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv")
    r = strat.resolve(
        cluster="eks-corporate", namespace="platform-a",
        workload_type="Deployment", workload_name="foo",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.team_id == "platform"
    assert r.owner_id == "aaron.miles"


def test_mapping_file_strategy_workload_name_wildcard(fixtures_dir):
    strat = MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv")
    r = strat.resolve(
        cluster="any-cluster", namespace="any-ns",
        workload_type="Deployment", workload_name="audit-service",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.team_id == "security"


def test_namespace_fallback_always_returns_namespace_as_team():
    strat = NamespaceFallback()
    r = strat.resolve(
        cluster="c", namespace="ns-foo",
        workload_type="Deployment", workload_name="w",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r == OwnershipResult(
        team_id="ns-foo", owner_id=None,
        resolved_by_strategy="namespace_fallback", resolved_from="namespace:ns-foo",
    )


def test_resolver_chain_first_non_none_wins(fixtures_dir):
    chain = ResolverChain([
        LabelStrategy(label="team"),
        MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv"),
        NamespaceFallback(),
    ])
    # No label, no mapping hit → fallback
    r = chain.resolve(
        cluster="unknown", namespace="random",
        workload_type="Deployment", workload_name="x",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.resolved_by_strategy == "namespace_fallback"
    assert r.team_id == "random"


def test_resolver_chain_label_wins_over_mapping(fixtures_dir):
    chain = ResolverChain([
        LabelStrategy(label="team"),
        MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv"),
        NamespaceFallback(),
    ])
    labels_json = json.dumps({"kubernetes.namespace.label.team": "override"})
    r = chain.resolve(
        cluster="eks-corporate", namespace="platform-a",
        workload_type="Deployment", workload_name="foo",
        namespace_labels_json=labels_json, agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.team_id == "override"
    assert r.resolved_by_strategy == "label:team"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ownership.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Write the ownership module**

File: `sas/ingest/ownership.py`

```python
"""Ownership resolver chain. Returns (team_id, owner_id) for a workload.

Strategies are evaluated in order; first non-None wins. Every result carries
resolved_by_strategy + resolved_from for auditability.
"""
from __future__ import annotations

import csv
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass(frozen=True)
class OwnershipResult:
    team_id: Optional[str]
    owner_id: Optional[str]
    resolved_by_strategy: str
    resolved_from: str


class Strategy(Protocol):
    def resolve(self, *, cluster: str, namespace: str, workload_type: str,
                workload_name: str, namespace_labels_json: str,
                agent_tags_json: str, container_labels_json: str
                ) -> Optional[OwnershipResult]: ...


_LABEL_PREFIXES = [
    "kubernetes.namespace.label.",
    "kube.label.",
    "",  # raw key, in case the label is stored ungilded
]


class LabelStrategy:
    """Look for a configured label in namespace_labels / agent_tags / container_labels."""

    def __init__(self, label: str):
        self.label = label

    def resolve(self, *, cluster, namespace, workload_type, workload_name,
                namespace_labels_json, agent_tags_json, container_labels_json):
        for source_name, blob in (
            ("namespace_labels", namespace_labels_json),
            ("agent_tags", agent_tags_json),
            ("container_labels", container_labels_json),
        ):
            try:
                d = json.loads(blob) if blob else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(d, dict):
                continue
            for prefix in _LABEL_PREFIXES:
                key = f"{prefix}{self.label}"
                if key in d and d[key]:
                    return OwnershipResult(
                        team_id=str(d[key]),
                        owner_id=None,
                        resolved_by_strategy=f"label:{self.label}",
                        resolved_from=f"{source_name}:{self.label}",
                    )
        return None


class MappingFileStrategy:
    """CSV with columns: cluster,namespace,workload_type,workload_name,team,owner.

    Values accept glob wildcards ('*'). First matching row wins. File is re-read
    on each resolve call — acceptable for ingest-time use (not per-query hot path).
    """

    def __init__(self, path: Path):
        self.path = path

    def resolve(self, *, cluster, namespace, workload_type, workload_name,
                namespace_labels_json, agent_tags_json, container_labels_json):
        if not self.path.exists():
            return None
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (fnmatch.fnmatchcase(cluster, row.get("cluster", "*") or "*")
                    and fnmatch.fnmatchcase(namespace, row.get("namespace", "*") or "*")
                    and fnmatch.fnmatchcase(workload_type, row.get("workload_type", "*") or "*")
                    and fnmatch.fnmatchcase(workload_name, row.get("workload_name", "*") or "*")):
                    team = row.get("team") or None
                    owner = row.get("owner") or None
                    return OwnershipResult(
                        team_id=team, owner_id=owner,
                        resolved_by_strategy="mapping_file",
                        resolved_from=f"{self.path.name}:row_match",
                    )
        return None


class NamespaceFallback:
    """Last-resort: team = namespace, no owner."""

    def resolve(self, *, cluster, namespace, workload_type, workload_name,
                namespace_labels_json, agent_tags_json, container_labels_json):
        return OwnershipResult(
            team_id=namespace,
            owner_id=None,
            resolved_by_strategy="namespace_fallback",
            resolved_from=f"namespace:{namespace}",
        )


class ResolverChain:
    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies

    def resolve(self, **kwargs) -> OwnershipResult:
        for strat in self.strategies:
            r = strat.resolve(**kwargs)
            if r is not None:
                return r
        # Belt-and-braces: if no fallback was provided, synthesize one.
        return NamespaceFallback().resolve(**kwargs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ownership.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add sas/ingest/ownership.py tests/fixtures/ownership_sample.csv tests/test_ownership.py
git commit -m "feat(sas): ownership resolver chain with label + mapping-file + fallback"
```

---

## Task 9: Reason code (pure logic, TDD-friendly)

**Files:**
- Create: `sas/ingest/reason_code.py`
- Create: `tests/test_reason_code.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_reason_code.py`

```python
from sas.ingest.reason_code import compute_reason_code, ReasonContext


def test_risk_accepted_flip_is_accepted():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=True,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "ACCEPTED"


def test_newer_digest_without_cve_is_patched():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=True,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "PATCHED"


def test_image_not_running_anywhere_is_retired():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=False,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "RETIRED"


def test_cve_missing_from_feed_is_feed_withdrawn():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=True,
    )
    assert compute_reason_code(ctx) == "FEED_WITHDRAWN"


def test_none_of_the_above_is_unknown():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "UNKNOWN"


def test_accepted_takes_precedence_over_patched():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=True,
        newer_digest_exists_without_cve=True,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "ACCEPTED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_reason_code.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the reason code module**

File: `sas/ingest/reason_code.py`

```python
"""Pure logic: given prior + current state, compute the reason_code for a CLOSED finding.

This is deliberately a pure function of a small named context so it can be
tested exhaustively without any DB dependency. The caller (finding_diff) is
responsible for assembling the context from DuckDB queries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ReasonCode = Literal["PATCHED", "RETIRED", "SCALED_TO_ZERO", "ACCEPTED",
                     "FEED_WITHDRAWN", "UNKNOWN"]


@dataclass(frozen=True)
class ReasonContext:
    risk_accepted_was: bool
    risk_accepted_is: bool
    newer_digest_exists_without_cve: bool
    image_still_runs_anywhere: bool
    cve_missing_from_feed: bool


def compute_reason_code(ctx: ReasonContext) -> ReasonCode:
    # Order matches spec §4.2.
    if not ctx.risk_accepted_was and ctx.risk_accepted_is:
        return "ACCEPTED"
    if ctx.newer_digest_exists_without_cve:
        return "PATCHED"
    if not ctx.image_still_runs_anywhere:
        return "RETIRED"
    if ctx.cve_missing_from_feed:
        return "FEED_WITHDRAWN"
    return "UNKNOWN"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_reason_code.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/reason_code.py tests/test_reason_code.py
git commit -m "feat(sas): reason_code pure logic with precedence rules"
```

---

## Task 10: Runtime snapshot writer

**Files:**
- Create: `sas/ingest/runtime_snapshot.py`
- Create: `tests/test_runtime_snapshot.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_runtime_snapshot.py`

```python
from datetime import datetime, timezone, date
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.runtime_snapshot import write_runtime_snapshot


@pytest.fixture
def two_container_df():
    return pd.DataFrame([
        {
            "kubernetes_cluster_name": "cluster-a",
            "kubernetes_namespace_name": "ns-a",
            "kubernetes_workload_type": "Deployment",
            "kubernetes_workload_name": "foo",
            "kubernetes_container_name": "foo-main",
            "image_id": "sha256:aaa",
        },
        {
            "kubernetes_cluster_name": "cluster-a",
            "kubernetes_namespace_name": "ns-a",
            "kubernetes_workload_type": "Deployment",
            "kubernetes_workload_name": "foo",
            "kubernetes_container_name": "foo-main",
            "image_id": "sha256:aaa",
        },  # duplicate — should collapse to replica_count=2
        {
            "kubernetes_cluster_name": "cluster-a",
            "kubernetes_namespace_name": "ns-a",
            "kubernetes_workload_type": "Deployment",
            "kubernetes_workload_name": "bar",
            "kubernetes_container_name": "bar-main",
            "image_id": "sha256:bbb",
        },
    ])


def test_runtime_snapshot_aggregates_replica_counts(db, two_container_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    write_runtime_snapshot(db, two_container_df, snapshot_at)

    rows = db.execute(
        "SELECT workload_name, replica_count FROM workload_runs_image_daily "
        "ORDER BY workload_name"
    ).fetchall()
    assert rows == [("bar", 1), ("foo", 2)]


def test_runtime_snapshot_is_idempotent_on_same_day(db, two_container_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    write_runtime_snapshot(db, two_container_df, snapshot_at)
    write_runtime_snapshot(db, two_container_df, snapshot_at)
    count = db.execute("SELECT count(*) FROM workload_runs_image_daily").fetchone()[0]
    assert count == 2  # same 2 unique rows, not 4


def test_runtime_snapshot_uses_date_of_snapshot_at(db, two_container_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    write_runtime_snapshot(db, two_container_df, snapshot_at)
    row = db.execute(
        "SELECT date FROM workload_runs_image_daily LIMIT 1"
    ).fetchone()
    assert row[0] == date(2026, 4, 23)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_runtime_snapshot.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the runtime snapshot module**

File: `sas/ingest/runtime_snapshot.py`

```python
"""Writes one row per (date, cluster, namespace, workload, container, image_id).

Replica count is derived by grouping duplicate rows in the CSV — each CSV row
corresponds to one running container instance.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


_GROUP_COLS = [
    "kubernetes_cluster_name",
    "kubernetes_namespace_name",
    "kubernetes_workload_type",
    "kubernetes_workload_name",
    "kubernetes_container_name",
    "image_id",
]


def write_runtime_snapshot(conn, df: pd.DataFrame, snapshot_at: datetime) -> None:
    agg = (
        df[_GROUP_COLS]
        .groupby(_GROUP_COLS, dropna=False)
        .size()
        .reset_index(name="replica_count")
    )
    snapshot_date = snapshot_at.date()

    for _, r in agg.iterrows():
        conn.execute(
            """
            INSERT INTO workload_runs_image_daily
              (date, cluster_name, namespace_name, workload_type, workload_name,
               container_name, image_id, replica_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (date, cluster_name, namespace_name, workload_type,
                         workload_name, container_name, image_id)
            DO UPDATE SET replica_count = EXCLUDED.replica_count
            """,
            [
                snapshot_date,
                r["kubernetes_cluster_name"], r["kubernetes_namespace_name"],
                r["kubernetes_workload_type"], r["kubernetes_workload_name"],
                r["kubernetes_container_name"], r["image_id"],
                int(r["replica_count"]),
            ],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_runtime_snapshot.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/runtime_snapshot.py tests/test_runtime_snapshot.py
git commit -m "feat(sas): runtime snapshot with replica aggregation"
```

---

## Task 11: Finding state diff (the core)

**Files:**
- Create: `sas/ingest/finding_diff.py`
- Create: `tests/test_finding_diff.py`

**Important:** this task is the heart of the whole phase. It's the largest file in this plan. The test set is deliberately exhaustive — every state transition (new / reseen / reopened / disappeared-with-each-reason-code) gets its own test.

- [ ] **Step 1: Write the failing test**

File: `tests/test_finding_diff.py`

```python
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.runtime_snapshot import write_runtime_snapshot
from sas.ingest.entity_upsert import upsert_entities
from sas.ingest.finding_diff import diff_and_apply_findings


def _row(image_id="sha256:aaa", cve="CVE-2026-00001", pkg="libfoo", ver="1.0",
         pkg_path="/lib/libfoo", severity="Critical", risk_accepted=False,
         fix_available=True, in_use=True, public_exploit=False):
    return {
        "vulnerability_name": cve,
        "vulnerability_severity": severity,
        "package_name": pkg,
        "package_version": ver,
        "package_type": "OS",
        "package_path": pkg_path,
        "image_name": "registry/foo:1.0",
        "os_name": "alpine 3.20",
        "cvss_version": "3.0",
        "cvss_score": 9.1,
        "cvss_vector": "AV:N",
        "disclosure_date": pd.Timestamp("2026-01-01T00:00:00Z"),
        "fix_available_date": pd.Timestamp("2026-01-02T00:00:00Z"),
        "fix_version": "1.1",
        "public_exploit": public_exploit,
        "kubernetes_cluster_name": "cluster-a",
        "kubernetes_namespace_name": "ns-a",
        "kubernetes_workload_type": "Deployment",
        "kubernetes_workload_name": "foo",
        "kubernetes_container_name": "foo-main",
        "image_id": image_id,
        "package_in_use": in_use,
        "risk_accepted": risk_accepted,
        "cisa_kev_publish_date": pd.NaT,
        "cisa_kev_due_date": pd.NaT,
        "cisa_kev_known_ransomware": False,
        "fix_available": fix_available,
        "agent_tags": "{}",
        "container_labels": "{}",
        "namespace_labels": "{}",
    }


def _prep(db, df, snapshot_at):
    upsert_entities(db, df, snapshot_at)
    write_runtime_snapshot(db, df, snapshot_at)


def test_new_finding_inserts_open_row(db):
    create_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([_row()])
    _prep(db, df, day1)
    diff_and_apply_findings(db, df, day1)
    rows = db.execute(
        "SELECT cve_id, state, first_seen, last_seen FROM finding_state"
    ).fetchall()
    assert rows == [("CVE-2026-00001", "OPEN", day1, day1)]


def test_reseen_finding_updates_last_seen_only(db):
    create_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    df = pd.DataFrame([_row()])
    _prep(db, df, day1); diff_and_apply_findings(db, df, day1)
    _prep(db, df, day2); diff_and_apply_findings(db, df, day2)
    count = db.execute("SELECT count(*) FROM finding_state").fetchone()[0]
    assert count == 1
    row = db.execute(
        "SELECT first_seen, last_seen, state FROM finding_state"
    ).fetchone()
    assert row[0] == day1
    assert row[1] == day2
    assert row[2] == "OPEN"


def test_disappeared_finding_closes_with_reason_retired(db):
    """Image disappears entirely → image_still_runs_anywhere=False → RETIRED."""
    create_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    df1 = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df1, day1); diff_and_apply_findings(db, df1, day1)
    # Day 2: completely different image in the CSV; sha256:aaa no longer seen
    df2 = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df2, day2); diff_and_apply_findings(db, df2, day2)

    closed_rows = db.execute(
        "SELECT state, reason_code FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchall()
    assert closed_rows == [("CLOSED", "RETIRED")]


def test_disappeared_finding_without_risk_flip_is_unknown_or_feed_withdrawn(db):
    """Sibling risk_accepted=True on a DIFFERENT CVE does NOT trigger ACCEPTED.

    The ACCEPTED path requires the same (image, cve, pkg) natural key to be
    flipped — not a different finding on the same image.
    """
    create_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    df1 = pd.DataFrame([_row(risk_accepted=False)])
    _prep(db, df1, day1); diff_and_apply_findings(db, df1, day1)
    # Day 2: same image running, CVE-2026-00001 gone, a different CVE present
    df2 = pd.DataFrame([_row(cve="CVE-2026-00002", risk_accepted=True)])
    _prep(db, df2, day2); diff_and_apply_findings(db, df2, day2)
    row = db.execute(
        "SELECT reason_code FROM finding_state WHERE cve_id = 'CVE-2026-00001'"
    ).fetchone()
    # image still runs; original CVE not in today's feed → FEED_WITHDRAWN
    assert row[0] in ("UNKNOWN", "FEED_WITHDRAWN")


def test_reopened_finding_creates_new_record_and_increments_reopen_count(db):
    create_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=2)
    df_with = pd.DataFrame([_row()])
    df_without = pd.DataFrame([_row(cve="CVE-2026-00002")])  # same image, different CVE

    _prep(db, df_with, day1); diff_and_apply_findings(db, df_with, day1)
    _prep(db, df_without, day2); diff_and_apply_findings(db, df_without, day2)
    _prep(db, df_with, day3); diff_and_apply_findings(db, df_with, day3)

    rows = db.execute(
        "SELECT state, reopen_count, is_regression FROM finding_state "
        "WHERE cve_id = 'CVE-2026-00001' ORDER BY first_seen"
    ).fetchall()
    # Two rows for the same natural key: the closed original, and the reopened record.
    assert len(rows) == 2
    closed, reopened = rows
    assert closed[0] == "CLOSED"
    assert reopened[0] == "OPEN"
    assert reopened[1] == 1
    assert reopened[2] is True


def test_days_open_is_computed(db):
    create_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day5 = day1 + timedelta(days=4)
    df = pd.DataFrame([_row()])
    _prep(db, df, day1); diff_and_apply_findings(db, df, day1)
    _prep(db, df, day5); diff_and_apply_findings(db, df, day5)
    row = db.execute("SELECT days_open FROM finding_state").fetchone()
    assert row[0] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_finding_diff.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the finding_diff module**

File: `sas/ingest/finding_diff.py`

```python
"""The heart of the ingest: diff today's findings against the current OPEN state.

Four transition types:
  - NEW         → INSERT OPEN row with first_seen = snapshot_at
  - RESEEN      → UPDATE last_seen, drift columns (severity, cvss, etc.)
  - REOPENED    → INSERT a new row with reopened_at, reopen_count=prev+1
  - DISAPPEARED → UPDATE state=CLOSED, reason_code computed from ReasonContext

Reason-code detection uses the graph snapshot taken THIS ingest:
  - image_still_runs_anywhere: look in workload_runs_image_daily for today's date
  - newer_digest_exists_without_cve: check image_in_repository for any image in
    the same repository with a later first_seen that does NOT have this CVE
  - cve_missing_from_feed: was this CVE present in any row of today's CSV?
  - risk_accepted flip: read the prior OPEN row vs any incoming row referencing
    the same natural key
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Iterable

import pandas as pd

from sas.ingest.reason_code import ReasonContext, compute_reason_code


_DRIFT_COLUMNS = {
    "severity", "cvss_score", "fix_available", "fix_version",
    "risk_accepted", "public_exploit", "in_use",
}


def _natural_key(r) -> tuple:
    return (
        r["image_id"], r["vulnerability_name"], r["package_name"],
        r["package_version"], r["package_path"],
    )


def _row_to_fs_values(r, snapshot_at: datetime):
    return {
        "image_id": r["image_id"],
        "cve_id": r["vulnerability_name"],
        "package_name": r["package_name"],
        "package_version": r["package_version"],
        "package_path": r["package_path"],
        "severity": r["vulnerability_severity"],
        "cvss_score": float(r["cvss_score"]) if pd.notna(r["cvss_score"]) else None,
        "in_use": bool(r["package_in_use"]),
        "fix_available": bool(r["fix_available"]),
        "fix_version": r["fix_version"] if pd.notna(r["fix_version"]) else None,
        "risk_accepted": bool(r["risk_accepted"]),
        "public_exploit": bool(r["public_exploit"]),
    }


def diff_and_apply_findings(conn, df: pd.DataFrame, snapshot_at: datetime) -> dict:
    """Compare today's findings against current OPEN state and apply transitions.

    Returns a dict with counts: {"new": N, "reseen": N, "reopened": N, "closed": N}
    """
    counts = {"new": 0, "reseen": 0, "reopened": 0, "closed": 0}
    today = snapshot_at.date()
    today_cve_ids = set(df["vulnerability_name"].unique())

    # Today's natural keys
    today_keys = {_natural_key(r): r for _, r in df.iterrows()}

    # Current OPEN findings
    open_rows = conn.execute(
        """
        SELECT finding_id, image_id, cve_id, package_name, package_version,
               package_path, risk_accepted, first_seen, reopen_count
        FROM finding_state
        WHERE state = 'OPEN'
        """
    ).fetchall()
    open_by_key = {
        (r[1], r[2], r[3], r[4], r[5]): {
            "finding_id": r[0],
            "risk_accepted_was": bool(r[6]),
            "first_seen": r[7],
            "reopen_count": r[8] or 0,
        }
        for r in open_rows
    }

    # 1. NEW + RESEEN + REOPENED
    for key, r in today_keys.items():
        v = _row_to_fs_values(r, snapshot_at)
        if key in open_by_key:
            # RESEEN — update last_seen, drift columns, days_open
            prior = open_by_key[key]
            days_open = (today - prior["first_seen"].date()).days
            conn.execute(
                """
                UPDATE finding_state SET
                  last_seen = ?, severity = ?, cvss_score = ?, in_use = ?,
                  fix_available = ?, fix_version = ?, risk_accepted = ?,
                  public_exploit = ?, days_open = ?
                WHERE finding_id = ?
                """,
                [snapshot_at, v["severity"], v["cvss_score"], v["in_use"],
                 v["fix_available"], v["fix_version"], v["risk_accepted"],
                 v["public_exploit"], days_open, prior["finding_id"]],
            )
            counts["reseen"] += 1
        else:
            # NEW or REOPENED — check closed history
            closed_prior = conn.execute(
                """
                SELECT finding_id, reopen_count FROM finding_state
                WHERE image_id = ? AND cve_id = ? AND package_name = ?
                  AND package_version = ? AND package_path = ?
                  AND state = 'CLOSED'
                ORDER BY closed_at DESC LIMIT 1
                """,
                [key[0], key[1], key[2], key[3], key[4]],
            ).fetchone()
            if closed_prior is not None:
                new_reopen_count = (closed_prior[1] or 0) + 1
                _insert_finding_row(conn, v, snapshot_at, reopened_at=snapshot_at,
                                    reopen_count=new_reopen_count,
                                    is_regression=True)
                counts["reopened"] += 1
            else:
                _insert_finding_row(conn, v, snapshot_at, reopened_at=None,
                                    reopen_count=0, is_regression=False)
                counts["new"] += 1

    # 2. DISAPPEARED — OPEN rows whose natural key wasn't in today
    for key, prior in open_by_key.items():
        if key in today_keys:
            continue
        image_id, cve_id, _, _, _ = key
        ctx = _build_reason_context(
            conn, image_id=image_id, cve_id=cve_id,
            risk_accepted_was=prior["risk_accepted_was"],
            today=today, today_cve_ids=today_cve_ids, df=df,
        )
        reason = compute_reason_code(ctx)
        days_open = (today - prior["first_seen"].date()).days
        conn.execute(
            """
            UPDATE finding_state SET
              state = 'CLOSED', reason_code = ?, closed_at = ?, days_open = ?
            WHERE finding_id = ?
            """,
            [reason, snapshot_at, days_open, prior["finding_id"]],
        )
        counts["closed"] += 1

    return counts


def _insert_finding_row(conn, v, snapshot_at, *, reopened_at, reopen_count, is_regression):
    conn.execute(
        """
        INSERT INTO finding_state (
          finding_id, image_id, cve_id, package_name, package_version, package_path,
          severity, cvss_score, in_use, fix_available, fix_version, risk_accepted,
          public_exploit, first_seen, last_seen, state, reason_code, closed_at,
          reopened_at, reopen_count, days_open, is_regression
        ) VALUES (
          nextval('seq_finding_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          'OPEN', NULL, NULL, ?, ?, 0, ?
        )
        """,
        [v["image_id"], v["cve_id"], v["package_name"], v["package_version"],
         v["package_path"], v["severity"], v["cvss_score"], v["in_use"],
         v["fix_available"], v["fix_version"], v["risk_accepted"],
         v["public_exploit"], snapshot_at, snapshot_at, reopened_at, reopen_count,
         is_regression],
    )


def _build_reason_context(conn, *, image_id: str, cve_id: str,
                          risk_accepted_was: bool, today: date,
                          today_cve_ids: set, df: pd.DataFrame) -> ReasonContext:
    # image_still_runs_anywhere: any workload running this image_id today?
    row = conn.execute(
        "SELECT 1 FROM workload_runs_image_daily WHERE date = ? AND image_id = ? LIMIT 1",
        [today, image_id],
    ).fetchone()
    image_still_runs = row is not None

    # newer_digest_exists_without_cve: find same repo, a digest observed later, whose
    # findings on today do not include this CVE
    repo_row = conn.execute(
        "SELECT repository FROM image_in_repository WHERE image_id = ? LIMIT 1",
        [image_id],
    ).fetchone()
    newer_without_cve = False
    if repo_row is not None:
        repo = repo_row[0]
        newer_digests = conn.execute(
            """
            SELECT iir.image_id FROM image_in_repository iir
            JOIN image i ON i.image_id = iir.image_id
            WHERE iir.repository = ? AND i.first_seen > (
              SELECT first_seen FROM image WHERE image_id = ?
            )
            """,
            [repo, image_id],
        ).fetchall()
        newer_ids = {r[0] for r in newer_digests}
        if newer_ids:
            today_with_cve = set(
                df.loc[df["vulnerability_name"] == cve_id, "image_id"].unique()
            )
            if not (newer_ids & today_with_cve):
                newer_without_cve = True

    # cve_missing_from_feed: did today's CSV contain this CVE at all?
    cve_missing = cve_id not in today_cve_ids

    # risk_accepted flip: did any row for this image arrive today with risk_accepted=True?
    risk_is_now = bool(
        df[(df["image_id"] == image_id) & (df["risk_accepted"] == True)].shape[0] > 0  # noqa
    )

    return ReasonContext(
        risk_accepted_was=risk_accepted_was,
        risk_accepted_is=risk_is_now and not risk_accepted_was,
        newer_digest_exists_without_cve=newer_without_cve,
        image_still_runs_anywhere=image_still_runs,
        cve_missing_from_feed=cve_missing,
    )
```

- [ ] **Step 4: Create the sequence referenced by _insert_finding_row**

Modify: `sas/ingest/schema.py` — add one sequence to `_DDL` list at the end (before the commented sections). Add this entry:

```python
    "CREATE SEQUENCE IF NOT EXISTS seq_finding_id START 1",
```

Place it immediately before the `snapshot` table CREATE.

- [ ] **Step 5: Re-run schema tests to confirm still green**

Run: `.venv/bin/pytest tests/test_schema.py -v`
Expected: all PASS.

- [ ] **Step 6: Run finding_diff tests to verify they pass**

Run: `.venv/bin/pytest tests/test_finding_diff.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add sas/ingest/finding_diff.py sas/ingest/schema.py tests/test_finding_diff.py
git commit -m "feat(sas): finding state diff with new/reseen/reopened/closed transitions"
```

---

## Task 12: Rollup rebuilder

**Files:**
- Create: `sas/ingest/rollup.py`
- Create: `tests/test_rollup.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_rollup.py`

```python
from datetime import datetime, timezone, date
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.entity_upsert import upsert_entities
from sas.ingest.runtime_snapshot import write_runtime_snapshot
from sas.ingest.finding_diff import diff_and_apply_findings
from sas.ingest.rollup import rebuild_rollups_for_date


def _basic_row(cve, severity="Critical", image_id="sha256:aaa"):
    return {
        "vulnerability_name": cve, "vulnerability_severity": severity,
        "package_name": "libfoo", "package_version": "1.0", "package_type": "OS",
        "package_path": f"/lib/{cve}", "image_name": "registry/foo:1.0",
        "os_name": "alpine 3.20", "cvss_version": "3.0", "cvss_score": 9.1,
        "cvss_vector": "AV:N",
        "disclosure_date": pd.Timestamp("2026-01-01T00:00:00Z"),
        "fix_available_date": pd.Timestamp("2026-01-02T00:00:00Z"),
        "fix_version": "1.1", "public_exploit": False,
        "kubernetes_cluster_name": "cluster-a", "kubernetes_namespace_name": "ns-a",
        "kubernetes_workload_type": "Deployment", "kubernetes_workload_name": "foo",
        "kubernetes_container_name": "foo-main", "image_id": image_id,
        "package_in_use": True, "risk_accepted": False,
        "cisa_kev_publish_date": pd.NaT, "cisa_kev_due_date": pd.NaT,
        "cisa_kev_known_ransomware": False, "fix_available": True,
        "agent_tags": "{}", "container_labels": "{}", "namespace_labels": "{}",
    }


def test_rollup_by_image_counts_open_criticals(db):
    create_schema(db)
    snap = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        _basic_row("CVE-1", severity="Critical"),
        _basic_row("CVE-2", severity="Critical"),
        _basic_row("CVE-3", severity="High"),
    ])
    upsert_entities(db, df, snap)
    write_runtime_snapshot(db, df, snap)
    diff_and_apply_findings(db, df, snap)

    rebuild_rollups_for_date(db, snap.date())

    row = db.execute(
        "SELECT count_open_critical, count_open_high, count_open "
        "FROM daily_metrics_by_image WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row == (2, 1, 3)


def test_rollup_counts_new_on_the_day_they_appeared(db):
    create_schema(db)
    snap = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([_basic_row("CVE-1")])
    upsert_entities(db, df, snap); write_runtime_snapshot(db, df, snap)
    diff_and_apply_findings(db, df, snap)
    rebuild_rollups_for_date(db, snap.date())
    row = db.execute(
        "SELECT count_new FROM daily_metrics_by_image"
    ).fetchone()
    assert row[0] == 1


def test_rollup_is_idempotent(db):
    create_schema(db)
    snap = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([_basic_row("CVE-1")])
    upsert_entities(db, df, snap); write_runtime_snapshot(db, df, snap)
    diff_and_apply_findings(db, df, snap)
    rebuild_rollups_for_date(db, snap.date())
    rebuild_rollups_for_date(db, snap.date())
    count = db.execute(
        "SELECT count(*) FROM daily_metrics_by_image"
    ).fetchone()[0]
    assert count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_rollup.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the rollup module**

File: `sas/ingest/rollup.py`

```python
"""Rebuild the daily_metrics_* rollup tables for a given date.

Idempotent: DELETE then INSERT for the target date. Always safe to re-run.

Counts by reason code follow the spec's categorization:
  - count_fixed_patched: CLOSED with reason_code=PATCHED
  - count_fixed_retired: CLOSED with reason_code IN (RETIRED, SCALED_TO_ZERO)
  - count_fixed_accepted: CLOSED with reason_code=ACCEPTED
  - count_fixed_other:   CLOSED with reason_code IN (FEED_WITHDRAWN, UNKNOWN)
"""
from __future__ import annotations

from datetime import date


def rebuild_rollups_for_date(conn, target: date) -> None:
    _rebuild_by_image(conn, target)
    _rebuild_by_workload(conn, target)
    _rebuild_by_team(conn, target)
    _rebuild_by_repository(conn, target)
    _rebuild_by_cluster_severity(conn, target)


def _rebuild_by_image(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_image WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_image (
          date, image_id,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed, mttr_sum, mttr_count
        )
        SELECT
          ? AS date,
          image_id,
          SUM(CASE WHEN state='OPEN' AND severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND CAST(first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code IN ('FEED_WITHDRAWN','UNKNOWN') THEN 1 ELSE 0 END),
          SUM(CASE WHEN reopened_at IS NOT NULL
                     AND CAST(reopened_at AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     THEN days_open ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     THEN 1 ELSE 0 END)
        FROM finding_state
        GROUP BY image_id
        """,
        [target, target, target, target, target, target, target, target, target],
    )


def _rebuild_by_workload(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_workload WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_workload (
          date, cluster_name, namespace_name, workload_type, workload_name,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed, mttr_sum, mttr_count, replica_count
        )
        SELECT
          wri.date,
          wri.cluster_name, wri.namespace_name, wri.workload_type, wri.workload_name,
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND CAST(fs.first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('FEED_WITHDRAWN','UNKNOWN') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.reopened_at IS NOT NULL
                     AND CAST(fs.reopened_at AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN fs.days_open ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN 1 ELSE 0 END),
          SUM(wri.replica_count)
        FROM workload_runs_image_daily wri
        LEFT JOIN finding_state fs ON fs.image_id = wri.image_id
        WHERE wri.date = ?
        GROUP BY wri.date, wri.cluster_name, wri.namespace_name,
                 wri.workload_type, wri.workload_name
        """,
        [target, target, target, target, target, target, target, target, target],
    )


def _rebuild_by_team(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_team WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_team (
          date, team_id,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed, mttr_sum, mttr_count
        )
        SELECT
          ? AS date,
          wo.team_id,
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND CAST(fs.first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('FEED_WITHDRAWN','UNKNOWN') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.reopened_at IS NOT NULL
                     AND CAST(fs.reopened_at AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN fs.days_open ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN 1 ELSE 0 END)
        FROM workload_runs_image_daily wri
        JOIN workload_owned_by wo ON
             wo.cluster_name = wri.cluster_name
         AND wo.namespace_name = wri.namespace_name
         AND wo.workload_type = wri.workload_type
         AND wo.workload_name = wri.workload_name
        LEFT JOIN finding_state fs ON fs.image_id = wri.image_id
        WHERE wri.date = ?
        GROUP BY wo.team_id
        """,
        [target, target, target, target, target, target, target, target, target, target],
    )


def _rebuild_by_repository(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_repository WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_repository (
          date, repository,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed
        )
        SELECT
          ? AS date,
          iir.repository,
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND CAST(fs.first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('FEED_WITHDRAWN','UNKNOWN') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.reopened_at IS NOT NULL
                     AND CAST(fs.reopened_at AS DATE) = ? THEN 1 ELSE 0 END)
        FROM image_in_repository iir
        LEFT JOIN finding_state fs ON fs.image_id = iir.image_id
        GROUP BY iir.repository
        """,
        [target, target, target, target, target, target, target],
    )


def _rebuild_by_cluster_severity(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_cluster_severity WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_cluster_severity (date, cluster_name, severity, count_open)
        SELECT
          ? AS date, wri.cluster_name, fs.severity, COUNT(DISTINCT fs.finding_id)
        FROM workload_runs_image_daily wri
        JOIN finding_state fs ON fs.image_id = wri.image_id
        WHERE wri.date = ? AND fs.state = 'OPEN'
        GROUP BY wri.cluster_name, fs.severity
        """,
        [target, target],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_rollup.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/rollup.py tests/test_rollup.py
git commit -m "feat(sas): daily rollup rebuilder for image/workload/team/repo/cluster"
```

---

## Task 13: Ingest log + top-level orchestration

**Files:**
- Create: `sas/ingest/logger.py`
- Create: `sas/ingest/pipeline.py`  (renamed from what cli.py would delegate to)
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_pipeline.py`

```python
from datetime import datetime, timezone
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.pipeline import run_pipeline
from sas.ingest.ownership import ResolverChain, NamespaceFallback


def test_run_pipeline_end_to_end_on_minimal_csv(db, fixtures_dir):
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    result = run_pipeline(
        conn=db,
        csv_path=fixtures_dir / "minimal_valid.csv",
        resolver=resolver,
    )
    assert result["new"] >= 1
    assert result["snapshot_id"]
    # verify data landed
    assert db.execute("SELECT count(*) FROM finding_state").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM daily_metrics_by_image").fetchone()[0] >= 1
    assert db.execute("SELECT count(*) FROM snapshot").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM ingest_log").fetchone()[0] >= 1


def test_rerunning_same_csv_is_noop(db, fixtures_dir):
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    run_pipeline(conn=db, csv_path=fixtures_dir / "minimal_valid.csv", resolver=resolver)
    result2 = run_pipeline(conn=db, csv_path=fixtures_dir / "minimal_valid.csv", resolver=resolver)
    assert result2["already_ingested"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write the logger module**

File: `sas/ingest/logger.py`

```python
"""Append rows to ingest_log for a given snapshot_id."""
from __future__ import annotations

from datetime import datetime, timezone


def log_stage(conn, *, snapshot_id: str, stage: str, rows_affected: int,
              duration_ms: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_log (snapshot_id, stage, rows_affected, duration_ms, logged_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (snapshot_id, stage) DO UPDATE SET
          rows_affected = EXCLUDED.rows_affected,
          duration_ms = EXCLUDED.duration_ms,
          logged_at = EXCLUDED.logged_at
        """,
        [snapshot_id, stage, rows_affected, duration_ms,
         datetime.now(timezone.utc)],
    )
```

- [ ] **Step 4: Write the pipeline module**

File: `sas/ingest/pipeline.py`

```python
"""Top-level ingest orchestration. Composes all the steps per spec §6."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from sas.ingest.csv_validator import validate_csv_columns
from sas.ingest.csv_loader import load_csv
from sas.ingest.snapshot import (
    compute_snapshot_id, extract_snapshot_at,
    is_already_ingested, record_snapshot,
)
from sas.ingest.entity_upsert import upsert_entities
from sas.ingest.ownership import ResolverChain
from sas.ingest.runtime_snapshot import write_runtime_snapshot
from sas.ingest.finding_diff import diff_and_apply_findings
from sas.ingest.rollup import rebuild_rollups_for_date
from sas.ingest.logger import log_stage


def run_pipeline(*, conn, csv_path: Path, resolver: ResolverChain,
                 force: bool = False) -> dict:
    """Execute the full ingest pipeline for one CSV. Returns a summary dict."""
    csv_path = Path(csv_path)

    # 1. Validate
    t0 = time.monotonic()
    validate_csv_columns(csv_path)
    _ms = lambda t: int((time.monotonic() - t) * 1000)

    # 2. Load
    t = time.monotonic()
    df = load_csv(csv_path)
    load_ms = _ms(t)

    # 3. snapshot_id + idempotency
    snapshot_id = compute_snapshot_id(csv_path, row_count=len(df))
    snapshot_at = extract_snapshot_at(csv_path)
    if not force and is_already_ingested(conn, snapshot_id):
        return {"already_ingested": True, "snapshot_id": snapshot_id}

    # 4. Record the snapshot
    record_snapshot(conn, snapshot_id=snapshot_id, snapshot_at=snapshot_at,
                    source_filename=csv_path.name, row_count=len(df))
    log_stage(conn, snapshot_id=snapshot_id, stage="load",
              rows_affected=len(df), duration_ms=load_ms)

    # 5. Entities
    t = time.monotonic()
    upsert_entities(conn, df, snapshot_at)
    log_stage(conn, snapshot_id=snapshot_id, stage="entities",
              rows_affected=len(df), duration_ms=_ms(t))

    # 6. Ownership
    t = time.monotonic()
    _resolve_and_upsert_ownership(conn, df, resolver)
    log_stage(conn, snapshot_id=snapshot_id, stage="ownership",
              rows_affected=len(df), duration_ms=_ms(t))

    # 7. Runtime snapshot
    t = time.monotonic()
    write_runtime_snapshot(conn, df, snapshot_at)
    log_stage(conn, snapshot_id=snapshot_id, stage="runtime_snapshot",
              rows_affected=len(df), duration_ms=_ms(t))

    # 8. Finding diff
    t = time.monotonic()
    diff_counts = diff_and_apply_findings(conn, df, snapshot_at)
    log_stage(conn, snapshot_id=snapshot_id, stage="finding_diff",
              rows_affected=sum(diff_counts.values()), duration_ms=_ms(t))

    # 9. Rollups
    t = time.monotonic()
    rebuild_rollups_for_date(conn, snapshot_at.date())
    log_stage(conn, snapshot_id=snapshot_id, stage="rollups",
              rows_affected=0, duration_ms=_ms(t))

    total_ms = _ms(t0)
    log_stage(conn, snapshot_id=snapshot_id, stage="total",
              rows_affected=len(df), duration_ms=total_ms)

    return {
        "already_ingested": False,
        "snapshot_id": snapshot_id,
        "rows": len(df),
        "total_ms": total_ms,
        **diff_counts,
    }


def _resolve_and_upsert_ownership(conn, df: pd.DataFrame,
                                   resolver: ResolverChain) -> None:
    wl_df = df[[
        "kubernetes_cluster_name", "kubernetes_namespace_name",
        "kubernetes_workload_type", "kubernetes_workload_name",
        "namespace_labels", "agent_tags", "container_labels",
    ]].drop_duplicates(subset=[
        "kubernetes_cluster_name", "kubernetes_namespace_name",
        "kubernetes_workload_type", "kubernetes_workload_name",
    ])

    for _, r in wl_df.iterrows():
        result = resolver.resolve(
            cluster=r["kubernetes_cluster_name"],
            namespace=r["kubernetes_namespace_name"],
            workload_type=r["kubernetes_workload_type"],
            workload_name=r["kubernetes_workload_name"],
            namespace_labels_json=r["namespace_labels"],
            agent_tags_json=r["agent_tags"],
            container_labels_json=r["container_labels"],
        )
        if result.team_id:
            conn.execute(
                "INSERT INTO team (team_id, display_name) VALUES (?, ?) "
                "ON CONFLICT DO NOTHING",
                [result.team_id, result.team_id],
            )
        if result.owner_id:
            conn.execute(
                "INSERT INTO owner (owner_id, display_name) VALUES (?, ?) "
                "ON CONFLICT DO NOTHING",
                [result.owner_id, result.owner_id],
            )
        conn.execute(
            """
            INSERT INTO workload_owned_by
              (cluster_name, namespace_name, workload_type, workload_name,
               team_id, owner_id, resolved_by_strategy, resolved_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cluster_name, namespace_name, workload_type, workload_name)
            DO UPDATE SET
              team_id = EXCLUDED.team_id,
              owner_id = EXCLUDED.owner_id,
              resolved_by_strategy = EXCLUDED.resolved_by_strategy,
              resolved_from = EXCLUDED.resolved_from
            """,
            [r["kubernetes_cluster_name"], r["kubernetes_namespace_name"],
             r["kubernetes_workload_type"], r["kubernetes_workload_name"],
             result.team_id, result.owner_id,
             result.resolved_by_strategy, result.resolved_from],
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: all 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add sas/ingest/logger.py sas/ingest/pipeline.py tests/test_pipeline.py
git commit -m "feat(sas): end-to-end ingest pipeline with per-stage logging"
```

---

## Task 14: CLI entrypoint

**Files:**
- Create: `sas/ingest/cli.py`
- Create: `sas/ingest/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_cli.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL — CLI doesn't exist.

- [ ] **Step 3: Write the CLI**

File: `sas/ingest/cli.py`

```python
"""`python -m sas.ingest <csv>` — the public command."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from sas.ingest.config import get_config
from sas.ingest.schema import create_schema
from sas.ingest.ownership import (
    ResolverChain, LabelStrategy, MappingFileStrategy, NamespaceFallback,
)
from sas.ingest.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sas.ingest",
        description="Ingest a Sysdig vulnerability CSV into the SAS analytics store.",
    )
    parser.add_argument("csv", type=Path, help="Path to the Sysdig CSV export")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest even if snapshot_id already recorded")
    args = parser.parse_args(argv)

    cfg = get_config()
    cfg.ensure_data_dir()

    resolver = _build_default_resolver(cfg)

    conn = duckdb.connect(str(cfg.duckdb_path))
    try:
        create_schema(conn)
        result = run_pipeline(
            conn=conn, csv_path=args.csv,
            resolver=resolver, force=args.force,
        )
    finally:
        conn.close()

    if result.get("already_ingested"):
        print(f"already ingested: snapshot_id={result['snapshot_id']} — skipping")
        return 0

    print(
        f"ingested {result['rows']} rows in {result['total_ms']}ms "
        f"(new={result.get('new', 0)} reseen={result.get('reseen', 0)} "
        f"reopened={result.get('reopened', 0)} closed={result.get('closed', 0)})"
    )
    return 0


def _build_default_resolver(cfg) -> ResolverChain:
    strategies = [
        LabelStrategy(label="team"),
        LabelStrategy(label="cost-center"),
    ]
    if cfg.ownership_mapping_path.exists():
        strategies.append(MappingFileStrategy(path=cfg.ownership_mapping_path))
    strategies.append(NamespaceFallback())
    return ResolverChain(strategies)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write the __main__ module**

File: `sas/ingest/__main__.py`

```python
from sas.ingest.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add sas/ingest/cli.py sas/ingest/__main__.py tests/test_cli.py
git commit -m "feat(sas): cli entrypoint with idempotency message"
```

---

## Task 15: End-to-end integration test on the real sample

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_integration.py`

```python
"""End-to-end integration test using the real Phoenix sample CSV.

Asserts invariants, not exact values — because a real sample will drift over time.
"""
from pathlib import Path
import pytest
import duckdb

from sas.ingest.schema import create_schema
from sas.ingest.pipeline import run_pipeline
from sas.ingest.ownership import ResolverChain, NamespaceFallback


REPO_ROOT = Path(__file__).parent.parent
SAMPLE_CSV = REPO_ROOT / "phoenix-vuln-findings-2026_04_23.csv"


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample CSV not present")
def test_real_sample_ingests_cleanly(tmp_path):
    db_path = tmp_path / "sas.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        create_schema(conn)
        resolver = ResolverChain([NamespaceFallback()])
        result = run_pipeline(conn=conn, csv_path=SAMPLE_CSV, resolver=resolver)
    finally:
        conn.close()

    assert not result["already_ingested"]
    assert result["rows"] > 0
    assert result["new"] == result["rows"] or result["new"] > 0

    # Reopen and assert invariants
    conn = duckdb.connect(str(db_path))
    try:
        # Some data landed in each key table
        for table in ("image", "cve", "package", "cluster", "namespace",
                      "workload", "repository", "finding_state",
                      "workload_runs_image_daily", "daily_metrics_by_image"):
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            assert count > 0, f"{table} was empty after ingest"

        # Every finding_state row has a valid state
        bad = conn.execute(
            "SELECT count(*) FROM finding_state WHERE state NOT IN ('OPEN','CLOSED')"
        ).fetchone()[0]
        assert bad == 0

        # On first ingest, every finding should be OPEN (no prior state to close)
        closed = conn.execute(
            "SELECT count(*) FROM finding_state WHERE state = 'CLOSED'"
        ).fetchone()[0]
        assert closed == 0

        # replica_count is never negative
        neg = conn.execute(
            "SELECT count(*) FROM workload_runs_image_daily WHERE replica_count < 1"
        ).fetchone()[0]
        assert neg == 0
    finally:
        conn.close()


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample CSV not present")
def test_real_sample_reingestion_is_idempotent(tmp_path):
    db_path = tmp_path / "sas.duckdb"
    resolver = ResolverChain([NamespaceFallback()])

    conn = duckdb.connect(str(db_path))
    try:
        create_schema(conn)
        first = run_pipeline(conn=conn, csv_path=SAMPLE_CSV, resolver=resolver)
        first_findings = conn.execute("SELECT count(*) FROM finding_state").fetchone()[0]
    finally:
        conn.close()

    conn = duckdb.connect(str(db_path))
    try:
        second = run_pipeline(conn=conn, csv_path=SAMPLE_CSV, resolver=resolver)
        second_findings = conn.execute("SELECT count(*) FROM finding_state").fetchone()[0]
    finally:
        conn.close()

    assert second["already_ingested"] is True
    assert first_findings == second_findings
```

- [ ] **Step 2: Run the integration test**

Run: `.venv/bin/pytest tests/test_integration.py -v`
Expected: 2 PASS (or 2 SKIPPED if the sample CSV isn't present on the runner).

- [ ] **Step 3: Run the full test suite and verify all green**

Run: `.venv/bin/pytest -v`
Expected: every test passes. No unexpected skips.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(sas): integration test on real phoenix sample csv"
```

---

## Task 16: Gitignore + small housekeeping

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add SAS data paths to .gitignore**

Check current `.gitignore` content, then append (do not overwrite):

```
# SAS ingest artifacts
sas.duckdb
sas.duckdb.wal
*.duckdb
*.duckdb.wal
~/sysdig-vuln-data/
.pytest_cache/
__pycache__/
*.pyc
```

Only append entries that aren't already present.

- [ ] **Step 2: Verify the real sample CSV is still tracked (or explicitly ignored)**

Run: `git status`
Expected: `phoenix-vuln-findings-2026_04_23.csv` state unchanged (committed earlier; not affected by these additions).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore(sas): gitignore duckdb and pytest cache artifacts"
```

---

## Phase 1 complete

At this point:
- `python -m sas.ingest <csv-file>` is a working CLI.
- Re-running the same CSV is a no-op.
- The real sample CSV ingests cleanly.
- Every step is covered by tests. Full suite: `.venv/bin/pytest -v`.
- Every state transition has a reason code. Every rollup is rebuildable.

**Next phase:** the Query primitive, FastAPI, and OpenAPI spec — lives in a separate plan (`2026-XX-XX-sas-phase2-query-engine-api.md`).
