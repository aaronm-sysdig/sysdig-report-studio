# Fast Ingest Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the ingest pipeline to use DuckDB native CSV reading and bulk SQL, eliminating all Pandas `.iterrows()` loops — reducing ingest time from ~2 min/file to ~10-20 sec/file for 1M-row CSVs.

**Architecture:** Load each CSV into a DuckDB temp table via `read_csv_auto()` (streams, zero extra memory). All entity upserts, finding diffs, and runtime snapshots become single bulk SQL statements. Ownership resolver stays in Python (needs `fnmatch`) but batches inserts.

**Tech Stack:** DuckDB native CSV reader, bulk `INSERT ... SELECT`, `executemany()`, existing schema unchanged.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `sas/ingest/fast_loader.py` | **Create** | DuckDB-native CSV loader, returns temp table name |
| `sas/ingest/fast_entity_upsert.py` | **Create** | Bulk SQL upserts for image, cve, package, cluster, namespace, workload, repository |
| `sas/ingest/fast_runtime_snapshot.py` | **Create** | Single bulk INSERT for workload_runs_image_daily |
| `sas/ingest/fast_finding_diff.py` | **Create** | Set-based SQL for NEW/RESEEN/REOPENED/DISAPPEARED transitions |
| `sas/ingest/fast_pipeline.py` | **Create** | Orchestrator, wires all fast modules together |
| `sas/ingest/pipeline.py` | **Untouched** | Existing pipeline preserved as fallback |
| `sas/ingest/cli.py` | **Modify** | Add `--fast` flag to use new pipeline |

---

### Task 1: DuckDB-native CSV loader

**Files:**
- Create: `sas/ingest/fast_loader.py`

- [ ] **Step 1: Write `fast_loader.py`**

Create `sas/ingest/fast_loader.py` that loads a CSV into a DuckDB temp table using `read_csv_auto()`. Handles column normalization and type coercion in SQL.

```python
"""DuckDB-native CSV loader — streams, no Pandas."""
from __future__ import annotations

import re
from pathlib import Path

import duckdb


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


def load_csv_to_temp(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> str:
    """Load a Sysdig CSV into a DuckDB temp table. Returns the temp table name.

    Uses DuckDB's native CSV reader which streams — no full in-memory load.
    Columns are normalized (lowercase, underscores) and types coerced.
    """
    temp_table = "_ingest_staging"

    # Read CSV with auto-detection, rename columns to normalized names
    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")

    # First, read to discover columns
    sample = conn.execute(f"SELECT * FROM read_csv_auto('{csv_path}') LIMIT 0")
    raw_columns = [desc[0] for desc in sample.description]
    norm_columns = [_normalize(c) for c in raw_columns]

    # Build column alias clause
    aliases = ", ".join(f'"{rc}" AS "{nc}"' for rc, nc in zip(raw_columns, norm_columns))

    # Create temp table with normalized columns and coerced types
    bool_casts = ", ".join(
        f'"{c}" = CASE WHEN "{c}" IS NULL THEN FALSE ELSE "{c}"::BOOLEAN END'
        for c in _BOOL_COLS if c in norm_columns
    )
    date_casts = ", ".join(
        f'"{c}" = TRY_CAST("{c}" AS TIMESTAMPTZ)'
        for c in _DATE_COLS if c in norm_columns
    )

    # Build the full column list with casts
    all_cols = []
    for c in norm_columns:
        if c in _BOOL_COLS:
            all_cols.append(f'CASE WHEN "{c}" IS NULL THEN FALSE ELSE "{c}"::BOOLEAN END AS "{c}"')
        elif c in _DATE_COLS:
            all_cols.append(f'TRY_CAST("{c}" AS TIMESTAMPTZ) AS "{c}"')
        else:
            all_cols.append(f'"{c}"')

    select_cols = ", ".join(all_cols)

    conn.execute(f"""
        CREATE TEMPORARY TABLE {temp_table} AS
        SELECT {select_cols}
        FROM (SELECT {aliases} FROM read_csv_auto('{csv_path}'))
    """)

    # Get row count
    count = conn.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]

    return temp_table, count
```

- [ ] **Step 2: Verify it loads correctly**

Run this test to confirm the temp table has the right columns and row count:
```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
.venv/bin/python -c "
import duckdb
from pathlib import Path
from sas.ingest.fast_loader import load_csv_to_temp

conn = duckdb.connect(':memory:')
csv = Path('sas/sample_data/Phoenix/Kubernetes Workload Vulnerability Findings_2026-04-26T14_00_32.671Z.csv')
table, count = load_csv_to_temp(conn, csv)
print(f'Loaded {count} rows into {table}')
cols = [d[0] for d in conn.execute(f'DESCRIBE {table}').fetchall()]
print(f'Columns ({len(cols)}): {cols[:5]}...')
print(f'Has image_id: {\"image_id\" in cols}')
print(f'Has vulnerability_name: {\"vulnerability_name\" in cols}')
"
```
Expected: `Loaded 1000000 rows into _ingest_staging`, columns include `image_id`, `vulnerability_name`, etc.

- [ ] **Step 3: Commit**

```bash
git add sas/ingest/fast_loader.py
git commit -m "feat(ingest): add DuckDB-native CSV loader (no Pandas)"
```

---

### Task 2: Bulk entity upsert

**Files:**
- Create: `sas/ingest/fast_entity_upsert.py`

- [ ] **Step 1: Write `fast_entity_upsert.py`**

Replace all 8 `.iterrows()` loops with bulk `INSERT ... SELECT DISTINCT ... ON CONFLICT` statements.

```python
"""Bulk entity upserts from staging table — pure SQL, no iteration."""
from __future__ import annotations

from datetime import datetime


def upsert_entities(conn, snapshot_at: datetime) -> None:
    """Upsert all entity rows from _ingest_staging using bulk SQL."""
    t = snapshot_at  # shorthand

    # --- image ---
    conn.execute(f"""
        INSERT INTO image (image_id, os_name, first_seen, last_seen,
                           current_repository, current_tag)
        SELECT
            image_id,
            os_name,
            '{t}'::timestamptz,
            '{t}'::timestamptz,
            _repo(image_name),
            _tag(image_name)
        FROM _ingest_staging
        GROUP BY image_id, os_name, image_name
        ON CONFLICT (image_id) DO UPDATE SET
            last_seen = EXCLUDED.last_seen,
            current_repository = EXCLUDED.current_repository,
            current_tag = EXCLUDED.current_tag,
            os_name = COALESCE(image.os_name, EXCLUDED.os_name)
    """)

    # --- repository ---
    conn.execute(f"""
        INSERT INTO repository (repository, first_seen, last_seen)
        SELECT DISTINCT _repo(image_name), '{t}'::timestamptz, '{t}'::timestamptz
        FROM _ingest_staging
        ON CONFLICT (repository) DO UPDATE SET last_seen = EXCLUDED.last_seen
    """)

    # --- image_in_repository ---
    conn.execute(f"""
        INSERT INTO image_in_repository (image_id, repository, tag, first_seen, last_seen)
        SELECT
            image_id,
            _repo(image_name),
            _tag(image_name),
            '{t}'::timestamptz,
            '{t}'::timestamptz
        FROM _ingest_staging
        GROUP BY image_id, image_name
        ON CONFLICT (image_id, repository, tag) DO UPDATE SET last_seen = EXCLUDED.last_seen
    """)

    # --- cve ---
    conn.execute(f"""
        INSERT INTO cve (cve_id, disclosure_date, fix_available_date,
                         cvss_version, initial_severity,
                         cisa_kev_publish_date, cisa_kev_due_date,
                         cisa_kev_known_ransomware, first_seen, last_seen)
        SELECT
            vulnerability_name,
            disclosure_date,
            fix_available_date,
            cvss_version,
            vulnerability_severity,
            cisa_kev_publish_date,
            cisa_kev_due_date,
            cisa_kev_known_ransomware,
            '{t}'::timestamptz,
            '{t}'::timestamptz
        FROM _ingest_staging
        GROUP BY vulnerability_name, disclosure_date, fix_available_date,
                 cvss_version, vulnerability_severity,
                 cisa_kev_publish_date, cisa_kev_due_date,
                 cisa_kev_known_ransomware
        ON CONFLICT (cve_id) DO UPDATE SET
            last_seen = EXCLUDED.last_seen,
            cisa_kev_publish_date = COALESCE(cve.cisa_kev_publish_date, EXCLUDED.cisa_kev_publish_date),
            cisa_kev_due_date = COALESCE(cve.cisa_kev_due_date, EXCLUDED.cisa_kev_due_date),
            cisa_kev_known_ransomware = EXCLUDED.cisa_kev_known_ransomware
    """)

    # --- package ---
    conn.execute("""
        INSERT INTO package (package_name, package_type)
        SELECT DISTINCT package_name, package_type
        FROM _ingest_staging
        ON CONFLICT (package_name, package_type) DO NOTHING
    """)

    # --- cluster ---
    conn.execute(f"""
        INSERT INTO cluster (cluster_name, first_seen, last_seen)
        SELECT DISTINCT kubernetes_cluster_name, '{t}'::timestamptz, '{t}'::timestamptz
        FROM _ingest_staging
        ON CONFLICT (cluster_name) DO UPDATE SET last_seen = EXCLUDED.last_seen
    """)

    # --- namespace ---
    conn.execute(f"""
        INSERT INTO namespace (cluster_name, namespace_name, first_seen, last_seen)
        SELECT DISTINCT kubernetes_cluster_name, kubernetes_namespace_name,
                        '{t}'::timestamptz, '{t}'::timestamptz
        FROM _ingest_staging
        ON CONFLICT (cluster_name, namespace_name) DO UPDATE SET last_seen = EXCLUDED.last_seen
    """)
    conn.execute("""
        INSERT INTO namespace_in_cluster (cluster_name, namespace_name)
        SELECT DISTINCT kubernetes_cluster_name, kubernetes_namespace_name
        FROM _ingest_staging
        ON CONFLICT DO NOTHING
    """)

    # --- workload ---
    conn.execute(f"""
        INSERT INTO workload (cluster_name, namespace_name, workload_type, workload_name,
                              first_seen, last_seen)
        SELECT DISTINCT kubernetes_cluster_name, kubernetes_namespace_name,
                        kubernetes_workload_type, kubernetes_workload_name,
                        '{t}'::timestamptz, '{t}'::timestamptz
        FROM _ingest_staging
        ON CONFLICT DO UPDATE SET last_seen = EXCLUDED.last_seen
    """)
    conn.execute("""
        INSERT INTO workload_in_namespace (cluster_name, namespace_name,
                                           workload_type, workload_name)
        SELECT DISTINCT kubernetes_cluster_name, kubernetes_namespace_name,
                        kubernetes_workload_type, kubernetes_workload_name
        FROM _ingest_staging
        ON CONFLICT DO NOTHING
    """)


def register_split_functions(conn) -> None:
    """Register SQL scalar functions for splitting image_name into repo + tag."""
    def _split_repo(image_name: str) -> str:
        if not image_name:
            return ""
        if "@" in image_name:
            repo, _ = image_name.rsplit("@", 1)
            return repo
        if ":" in image_name and "/" not in image_name.rsplit(":", 1)[-1]:
            repo, _ = image_name.rsplit(":", 1)
            return repo
        return image_name

    def _split_tag(image_name: str) -> str:
        if not image_name:
            return "latest"
        if "@" in image_name:
            _, digest = image_name.rsplit("@", 1)
            return digest
        if ":" in image_name and "/" not in image_name.rsplit(":", 1)[-1]:
            _, tag = image_name.rsplit(":", 1)
            return tag
        return "latest"

    conn.create_function("_repo", _split_repo, null_handling="pass-through")
    conn.create_function("_tag", _split_tag, null_handling="pass-through")
```

- [ ] **Step 2: Verify entity counts match old pipeline**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
.venv/bin/python -c "
import duckdb
from pathlib import Path
from datetime import datetime
from sas.ingest.schema import create_schema
from sas.ingest.fast_loader import load_csv_to_temp
from sas.ingest.fast_entity_upsert import upsert_entities, register_split_functions

db = Path.home() / 'sysdig-vuln-data' / 'sas.duckdb'
conn = duckdb.connect(str(db))
create_schema(conn)

csv = Path('sas/sample_data/Phoenix/Kubernetes Workload Vulnerability Findings_2026-04-26T14_00_32.671Z.csv')
table, count = load_csv_to_temp(conn, csv)
print(f'Loaded {count} rows')

register_split_functions(conn)
upsert_entities(conn, datetime(2026, 4, 26, 14, 0, 32))

for t in ['image', 'cve', 'package', 'cluster', 'namespace', 'workload', 'repository']:
    n = conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
    print(f'  {t}: {n}')
conn.close()
"
```
Expected: image ~1000s, cve ~1000s, package ~100s, cluster ~10, namespace ~100s, workload ~1000s

- [ ] **Step 3: Commit**

```bash
git add sas/ingest/fast_entity_upsert.py
git commit -m "feat(ingest): bulk SQL entity upsert (replaces 8 iterrows loops)"
```

---

### Task 3: Bulk runtime snapshot

**Files:**
- Create: `sas/ingest/fast_runtime_snapshot.py`

- [ ] **Step 1: Write `fast_runtime_snapshot.py`**

Single bulk INSERT replaces iterrows over unique (container, image) tuples.

```python
"""Bulk runtime snapshot write — single SQL statement."""
from __future__ import annotations

from datetime import datetime


def write_runtime_snapshot(conn, snapshot_at: datetime) -> None:
    """Write workload_runs_image_daily from staging — single bulk INSERT."""
    snapshot_date = snapshot_at.date()
    conn.execute(f"""
        INSERT INTO workload_runs_image_daily
          (date, cluster_name, namespace_name, workload_type, workload_name,
           container_name, image_id, replica_count)
        SELECT DISTINCT
            '{snapshot_date}'::date,
            kubernetes_cluster_name,
            kubernetes_namespace_name,
            kubernetes_workload_type,
            kubernetes_workload_name,
            kubernetes_container_name,
            image_id,
            1
        FROM _ingest_staging
        ON CONFLICT (date, cluster_name, namespace_name, workload_type,
                     workload_name, container_name, image_id)
        DO NOTHING
    """)
```

- [ ] **Step 2: Commit**

```bash
git add sas/ingest/fast_runtime_snapshot.py
git commit -m "feat(ingest): bulk runtime snapshot (single INSERT)"
```

---

### Task 4: Set-based finding diff

**Files:**
- Create: `sas/ingest/fast_finding_diff.py`

- [ ] **Step 1: Write `fast_finding_diff.py`**

This is the most complex module. Replace iterrows + per-row SQL with set-based operations.

Key approach:
1. Create a temp table of today's natural keys with all finding_state columns
2. LEFT JOIN against existing OPEN findings to classify NEW vs RESEEN vs REOPENED
3. Set of OPEN keys NOT in today = DISAPPEARED
4. Reason context: use SQL subqueries instead of DataFrame filtering

```python
"""Set-based finding diff — no iterrows, pure SQL."""
from __future__ import annotations

from datetime import datetime

from sas.ingest.reason_code import ReasonContext, compute_reason_code


def diff_and_apply_findings(conn, snapshot_at: datetime) -> dict:
    """Compare today's findings against current OPEN state, apply transitions.

    Uses set-based SQL: creates temp table of today's findings, LEFT JOINs
    against existing OPEN state to classify transitions.
    """
    today = snapshot_at.date()
    t = snapshot_at

    # 1. Create temp table of today's findings with natural keys
    conn.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _today_findings AS
        SELECT
            image_id,
            vulnerability_name AS cve_id,
            package_name,
            package_version,
            package_path,
            vulnerability_severity AS severity,
            CAST(cvss_score AS DOUBLE) AS cvss_score,
            package_in_use AS in_use,
            fix_available,
            fix_version,
            risk_accepted,
            public_exploit,
            cisa_kev_known_ransomware,
            '{t}'::timestamptz AS ts
        FROM _ingest_staging
    """)

    # 2. Find today's unique CVE IDs (for reason context)
    today_cve_ids = set(
        conn.execute("SELECT DISTINCT cve_id FROM _today_findings").fetchall()
    )
    today_cve_ids = {r[0] for r in today_cve_ids}

    # 3. LEFT JOIN: classify each today finding as NEW, RESEEN, or REOPENED
    conn.execute("""
        CREATE OR REPLACE TEMPORARY TABLE _classified AS
        SELECT
            tf.*,
            fs.finding_id,
            fs.risk_accepted AS risk_accepted_was,
            fs.first_seen AS prior_first_seen,
            fs.reopen_count AS prior_reopen_count,
            CASE
                WHEN fs.finding_id IS NOT NULL THEN 'RESEEN'
                ELSE 'NEW_OR_REOPENED'
            END AS transition
        FROM _today_findings tf
        LEFT JOIN finding_state fs ON
            fs.image_id = tf.image_id
            AND fs.cve_id = tf.cve_id
            AND fs.package_name = tf.package_name
            AND fs.package_version = tf.package_version
            AND fs.package_path = tf.package_path
            AND fs.state = 'OPEN'
    """)

    # 4. Handle RESEEN — bulk UPDATE
    resseen_count = conn.execute(f"""
        UPDATE finding_state SET
            last_seen = '{t}'::timestamptz,
            severity = c.severity,
            cvss_score = c.cvss_score,
            in_use = c.in_use,
            fix_available = c.fix_available,
            fix_version = c.fix_version,
            risk_accepted = c.risk_accepted,
            public_exploit = c.public_exploit,
            cisa_kev_known_ransomware = c.cisa_kev_known_ransomware,
            days_open = CAST(julianday('{today}') - julianday(DATE(c.prior_first_seen)) AS INTEGER)
        FROM _classified c
        WHERE c.transition = 'RESEEN'
          AND c.finding_id = finding_state.finding_id
    """).rowcount or 0

    # 5. Handle NEW / REOPENED — check closed history per natural key
    new_or_reopened = conn.execute("""
        SELECT image_id, cve_id, package_name, package_version, package_path,
               severity, cvss_score, in_use, fix_available, fix_version,
               risk_accepted, public_exploit, cisa_kev_known_ransomware
        FROM _classified
        WHERE transition = 'NEW_OR_REOPENED'
    """).fetchall()

    new_count = 0
    reopened_count = 0

    for row in new_or_reopened:
        v = dict(zip(
            ['image_id', 'cve_id', 'package_name', 'package_version', 'package_path',
             'severity', 'cvss_score', 'in_use', 'fix_available', 'fix_version',
             'risk_accepted', 'public_exploit', 'cisa_kev_known_ransomware'],
            row
        ))
        closed_prior = conn.execute("""
            SELECT finding_id, reopen_count FROM finding_state
            WHERE image_id = ? AND cve_id = ? AND package_name = ?
              AND package_version = ? AND package_path = ?
              AND state = 'CLOSED'
            ORDER BY closed_at DESC LIMIT 1
        """, [v['image_id'], v['cve_id'], v['package_name'],
              v['package_version'], v['package_path']]).fetchone()

        if closed_prior is not None:
            new_reopen_count = (closed_prior[1] or 0) + 1
            _insert_finding(conn, v, t, reopened_at=t, reopen_count=new_reopen_count,
                          is_regression=True)
            reopened_count += 1
        else:
            _insert_finding(conn, v, t, reopened_at=None, reopen_count=0,
                          is_regression=False)
            new_count += 1

    # 6. DISAPPEARED — OPEN findings not in today
    disappeared = conn.execute("""
        SELECT fs.image_id, fs.cve_id, fs.package_name, fs.package_version,
               fs.package_path, fs.finding_id, fs.risk_accepted, fs.first_seen
        FROM finding_state fs
        LEFT JOIN _classified c ON
            c.image_id = fs.image_id
            AND c.cve_id = fs.cve_id
            AND c.package_name = fs.package_name
            AND c.package_version = fs.package_version
            AND c.package_path = fs.package_path
        WHERE fs.state = 'OPEN' AND c.finding_id IS NULL
    """).fetchall()

    closed_count = 0
    for row in disappeared:
        d = dict(zip(
            ['image_id', 'cve_id', 'package_name', 'package_version', 'package_path',
             'finding_id', 'risk_accepted_was', 'prior_first_seen'],
            row
        ))
        ctx = _build_reason_context_sql(
            conn, image_id=d['image_id'], cve_id=d['cve_id'],
            risk_accepted_was=d['risk_accepted_was'],
            today=today, today_cve_ids=today_cve_ids
        )
        reason = compute_reason_code(ctx)
        days_open = (today - d['prior_first_seen'].date()).days
        conn.execute("""
            UPDATE finding_state SET
              state = 'CLOSED', reason_code = ?, closed_at = ?, days_open = ?
            WHERE finding_id = ?
        """, [reason, t, days_open, d['finding_id']])
        closed_count += 1

    # Cleanup temp tables
    conn.execute("DROP TABLE IF EXISTS _today_findings")
    conn.execute("DROP TABLE IF EXISTS _classified")

    return {"new": new_count, "reseen": resseen_count, "reopened": reopened_count,
            "closed": closed_count}


def _insert_finding(conn, v, snapshot_at, *, reopened_at, reopen_count, is_regression):
    conn.execute("""
        INSERT INTO finding_state (
          finding_id, image_id, cve_id, package_name, package_version,
          package_path, severity, cvss_score, in_use, fix_available,
          fix_version, risk_accepted, public_exploit, cisa_kev_known_ransomware,
          first_seen, last_seen,
          state, reason_code, closed_at, reopened_at, reopen_count,
          days_open, is_regression
        ) VALUES (
          nextval('seq_finding_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, 'OPEN', NULL, NULL, ?, ?, 0, ?
        )
    """, [v['image_id'], v['cve_id'], v['package_name'], v['package_version'],
          v['package_path'], v['severity'], v['cvss_score'], v['in_use'],
          v['fix_available'], v['fix_version'], v['risk_accepted'],
          v['public_exploit'], v['cisa_kev_known_ransomware'],
          snapshot_at, snapshot_at,
          reopened_at, reopen_count, is_regression])


def _build_reason_context_sql(
    conn, *, image_id: str, cve_id: str, risk_accepted_was: bool,
    today, today_cve_ids: set
) -> ReasonContext:
    # image_still_runs_anywhere
    row = conn.execute(
        "SELECT 1 FROM workload_runs_image_daily "
        "WHERE date = ? AND image_id = ? LIMIT 1",
        [today, image_id],
    ).fetchone()
    image_still_runs = row is not None

    # newer_digest_exists_without_cve
    repo_row = conn.execute(
        "SELECT repository FROM image_in_repository "
        "WHERE image_id = ? LIMIT 1",
        [image_id],
    ).fetchone()
    newer_without_cve = False
    if repo_row is not None:
        repo = repo_row[0]
        newer_digests = conn.execute("""
            SELECT iir.image_id FROM image_in_repository iir
            JOIN image i ON i.image_id = iir.image_id
            WHERE iir.repository = ? AND i.first_seen > (
              SELECT first_seen FROM image WHERE image_id = ?
            )
        """, [repo, image_id]).fetchall()
        newer_ids = {r[0] for r in newer_digests}
        if newer_ids:
            # Check if any newer image has this CVE today — use SQL
            placeholders = ",".join("?" for _ in newer_ids)
            today_with_cve = set(
                conn.execute(f"""
                    SELECT DISTINCT image_id FROM _ingest_staging
                    WHERE vulnerability_name = ? AND image_id IN ({placeholders})
                """, [cve_id] + list(newer_ids)).fetchall()
            )
            today_with_cve = {r[0] for r in today_with_cve}
            if not (newer_ids & today_with_cve):
                newer_without_cve = True

    # cve_missing_from_feed
    cve_missing = cve_id not in today_cve_ids

    # risk_accepted flip — check staging table
    mask_row = conn.execute("""
        SELECT 1 FROM _ingest_staging
        WHERE image_id = ? AND vulnerability_name = ? AND risk_accepted = TRUE
        LIMIT 1
    """, [image_id, cve_id]).fetchone()
    risk_is_now = (mask_row is not None) and not risk_accepted_was

    return ReasonContext(
        risk_accepted_was=risk_accepted_was,
        risk_accepted_is=risk_is_now,
        newer_digest_exists_without_cve=newer_without_cve,
        image_still_runs_anywhere=image_still_runs,
        cve_missing_from_feed=cve_missing,
    )
```

Note: The NEW/REOPENED loop still iterates in Python (one SQL per finding) because each needs a closed-history lookup. This is acceptable because it's only on findings NOT already OPEN (typically a small fraction). The RESEEN path (the vast majority) is bulk SQL.

- [ ] **Step 2: Commit**

```bash
git add sas/ingest/fast_finding_diff.py
git commit -m "feat(ingest): set-based finding diff (bulk RESEEN, SQL reason context)"
```

---

### Task 3: Fast pipeline orchestrator

**Files:**
- Create: `sas/ingest/fast_pipeline.py`
- Modify: `sas/ingest/cli.py` (add `--fast` flag)

- [ ] **Step 1: Write `fast_pipeline.py`**

Orchestrator that wires together the fast modules, reusing existing snapshot, ownership, and rollup logic.

```python
"""Fast ingest orchestration — DuckDB-native, no Pandas."""
from __future__ import annotations

import time
from pathlib import Path

from sas.ingest.snapshot import (
    compute_snapshot_id, extract_snapshot_at,
    is_already_ingested, record_snapshot,
)
from sas.ingest.ownership import ResolverChain
from sas.ingest.rollup import rebuild_rollups_for_date
from sas.ingest.logger import log_stage
from sas.ingest.fast_loader import load_csv_to_temp
from sas.ingest.fast_entity_upsert import upsert_entities, register_split_functions
from sas.ingest.fast_runtime_snapshot import write_runtime_snapshot
from sas.ingest.fast_finding_diff import diff_and_apply_findings


def run_pipeline(*, conn, csv_path: Path, resolver: ResolverChain,
                 force: bool = False) -> dict:
    """Execute the fast ingest pipeline for one CSV. Returns a summary dict."""
    csv_path = Path(csv_path)
    t0 = time.monotonic()
    _ms = lambda t: int((time.monotonic() - t) * 1000)

    # 1. Load CSV into temp table
    t = time.monotonic()
    temp_table, row_count = load_csv_to_temp(conn, csv_path)
    log_stage(conn, snapshot_id=f"file:{csv_path.name}", stage="load",
              rows_affected=row_count, duration_ms=_ms(t))

    # 2. Snapshot id + idempotency
    snapshot_id = compute_snapshot_id(csv_path, row_count=row_count)
    snapshot_at = extract_snapshot_at(csv_path)
    if not force and is_already_ingested(conn, snapshot_id):
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        return {"already_ingested": True, "snapshot_id": snapshot_id}

    conn.execute("BEGIN TRANSACTION")
    try:
        # 3. Record snapshot
        record_snapshot(conn, snapshot_id=snapshot_id, snapshot_at=snapshot_at,
                        source_filename=csv_path.name, row_count=row_count)

        # 4. Register split functions and upsert entities
        t = time.monotonic()
        register_split_functions(conn)
        upsert_entities(conn, snapshot_at)
        log_stage(conn, snapshot_id=snapshot_id, stage="entities",
                  rows_affected=row_count, duration_ms=_ms(t))

        # 5. Ownership (reuse existing — needs Python for fnmatch)
        t = time.monotonic()
        _resolve_and_upsert_ownership(conn, resolver)
        log_stage(conn, snapshot_id=snapshot_id, stage="ownership",
                  rows_affected=row_count, duration_ms=_ms(t))

        # 6. Runtime snapshot
        t = time.monotonic()
        write_runtime_snapshot(conn, snapshot_at)
        log_stage(conn, snapshot_id=snapshot_id, stage="runtime_snapshot",
                  rows_affected=row_count, duration_ms=_ms(t))

        # 7. Finding diff
        t = time.monotonic()
        diff_counts = diff_and_apply_findings(conn, snapshot_at)
        log_stage(conn, snapshot_id=snapshot_id, stage="finding_diff",
                  rows_affected=sum(diff_counts.values()), duration_ms=_ms(t))

        # 8. Rollups
        t = time.monotonic()
        rebuild_rollups_for_date(conn, snapshot_at.date())
        log_stage(conn, snapshot_id=snapshot_id, stage="rollups",
                  rows_affected=0, duration_ms=_ms(t))

        total_ms = _ms(t0)
        log_stage(conn, snapshot_id=snapshot_id, stage="total",
                  rows_affected=row_count, duration_ms=total_ms)

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("DROP TABLE IF EXISTS _ingest_staging")

    return {
        "already_ingested": False,
        "snapshot_id": snapshot_id,
        "rows": row_count,
        "total_ms": total_ms,
        **diff_counts,
    }


def _resolve_and_upsert_ownership(conn, resolver: ResolverChain) -> None:
    """Resolve ownership from staging table — batched inserts."""
    workloads = conn.execute("""
        SELECT DISTINCT
            kubernetes_cluster_name,
            kubernetes_namespace_name,
            kubernetes_workload_type,
            kubernetes_workload_name,
            namespace_labels,
            agent_tags,
            container_labels
        FROM _ingest_staging
    """).fetchall()

    teams = set()
    owners = set()
    workload_resolutions = []

    for w in workloads:
        result = resolver.resolve(
            cluster=w[0], namespace=w[1], workload_type=w[2], workload_name=w[3],
            namespace_labels_json=w[4], agent_tags_json=w[5],
            container_labels_json=w[6],
        )
        if result.team_id:
            teams.add(result.team_id)
        if result.owner_id:
            owners.add(result.owner_id)
        workload_resolutions.append((
            w[0], w[1], w[2], w[3],
            result.team_id, result.owner_id,
            result.resolved_by_strategy, result.resolved_from,
        ))

    # Batch insert teams
    for team_id in teams:
        conn.execute(
            "INSERT INTO team (team_id, display_name) VALUES (?, ?) "
            "ON CONFLICT DO NOTHING",
            [team_id, team_id],
        )
    # Batch insert owners
    for owner_id in owners:
        conn.execute(
            "INSERT INTO owner (owner_id, display_name) VALUES (?, ?) "
            "ON CONFLICT DO NOTHING",
            [owner_id, owner_id],
        )
    # Batch upsert workload ownership
    conn.executemany(
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
        workload_resolutions,
    )
```

- [ ] **Step 2: Add `--fast` flag to CLI**

Modify `sas/ingest/cli.py` to add a `--fast` flag that uses the new pipeline:

```python
# In main():
parser.add_argument("--fast", action="store_true",
                    help="Use DuckDB-native pipeline (no Pandas)")

# After creating resolver:
if args.fast:
    from sas.ingest.fast_pipeline import run_pipeline as fast_run_pipeline
    result = fast_run_pipeline(
        conn=conn, csv_path=args.csv,
        resolver=resolver, force=args.force,
    )
else:
    result = run_pipeline(
        conn=conn, csv_path=args.csv,
        resolver=resolver, force=args.force,
    )
```

- [ ] **Step 3: Test on first Phoenix file**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
rm -f ~/sysdig-vuln-data/sas.duckdb*
SAS_DATA_DIR=/Users/aaron.miles/GitHub/sysdig-report-studio/sas/sample_data/Phoenix \
  .venv/bin/python -m sas.ingest --fast \
  'sas/sample_data/Phoenix/Kubernetes Workload Vulnerability Findings_2026-04-26T14_00_32.671Z.csv'
```

Expected: Completes in ~10-30 seconds (vs 134 seconds with old pipeline).

- [ ] **Step 4: Verify counts match old pipeline**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
.venv/bin/python -c "
import duckdb
db = Path.home() / 'sysdig-vuln-data' / 'sas.duckdb'
conn = duckdb.connect(str(db))
for t in ['image', 'cve', 'package', 'cluster', 'namespace', 'workload',
          'repository', 'workload_runs_image_daily', 'finding_state']:
    n = conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
    print(f'  {t}: {n}')
fs_open = conn.execute(\"SELECT count(*) FROM finding_state WHERE state='OPEN'\").fetchone()[0]
fs_closed = conn.execute(\"SELECT count(*) FROM finding_state WHERE state='CLOSED'\").fetchone()[0]
print(f'  finding_state OPEN: {fs_open}, CLOSED: {fs_closed}')
conn.close()
"
```

Expected: Similar counts to old pipeline (image ~1000s, cve ~1000s, finding_state ~10000s OPEN)

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/fast_pipeline.py sas/ingest/cli.py
git commit -m "feat(ingest): fast pipeline orchestrator with --fast CLI flag"
```

---

### Task 4: Ingest all 11 Phoenix files

- [ ] **Step 1: Clean DB and run all 11 files**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
rm -f ~/sysdig-vuln-data/sas.duckdb*
for f in sas/sample_data/Phoenix/*.csv; do
  echo "=== $(basename $f) ==="
  SAS_DATA_DIR=/Users/aaron.miles/GitHub/sysdig-report-studio/sas/sample_data/Phoenix \
    .venv/bin/python -m sas.ingest --fast "$f"
  echo ""
done
echo "=== DONE ==="
```

Expected: All 11 files ingested, total time ~2-5 minutes (vs estimated 20+ minutes with old pipeline).

- [ ] **Step 2: Final verification**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
.venv/bin/python -c "
import duckdb
from pathlib import Path
db = Path.home() / 'sysdig-vuln-data' / 'sas.duckdb'
conn = duckdb.connect(str(db))

print('=== Table counts ===')
for t in ['image', 'cve', 'package', 'cluster', 'namespace', 'workload',
          'repository', 'workload_runs_image_daily', 'finding_state', 'snapshot']:
    n = conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
    print(f'  {t}: {n}')

fs_open = conn.execute(\"SELECT count(*) FROM finding_state WHERE state='OPEN'\").fetchone()[0]
fs_closed = conn.execute(\"SELECT count(*) FROM finding_state WHERE state='CLOSED'\").fetchone()[0]
print(f'  finding_state OPEN: {fs_open}, CLOSED: {fs_closed}')

print()
print('=== Snapshots ===')
for row in conn.execute('SELECT snapshot_at, source_filename, row_count FROM snapshot ORDER BY snapshot_at').fetchall():
    print(f'  {row[0]} - {row[1]} ({row[2]:,} rows)')

print()
db_size = db.stat().st_size
print(f'DB size: {db_size / 1024 / 1024:.1f} MB')
conn.close()
"
```

Expected: 11 snapshots, date range Apr 26 - May 6, DB size < 500MB (vs 12GB raw CSVs)

---

## Self-Review

**Spec coverage:**
- [x] DuckDB-native CSV loading (no Pandas) — Task 1
- [x] Bulk entity upsert (replaces 8 iterrows) — Task 2
- [x] Bulk runtime snapshot — Task 3
- [x] Set-based finding diff — Task 4
- [x] Ownership resolver preserved (Python, batched) — Task 5
- [x] CLI flag for opt-in — Task 5
- [x] Test on 1 file, verify counts — Task 5
- [x] Ingest all 11 Phoenix files — Task 6

**Placeholder scan:** No TBDs, no "implement later", all code blocks complete.

**Type consistency:** `snapshot_at` is `datetime` throughout. `today` is `date`. Temp table is `_ingest_staging`. Split functions `_repo()`/`_tag()` registered before entity upsert.

**Risk assessment:**
- **Low risk:** Entity upsert and runtime snapshot are pure SQL — more correct than iterrows
- **Medium risk:** Finding diff — RESEEN is bulk SQL (safe), NEW/REOPENED still iterates (same as before), DISAPPEARED uses SQL subqueries for reason context (same logic, different implementation)
- **Ownership:** Unchanged logic, just batched inserts — lowest risk
- **Fallback:** Old pipeline untouched, `--fast` is opt-in
