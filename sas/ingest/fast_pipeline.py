"""Fast ingest orchestration — DuckDB-native, no Pandas."""
from __future__ import annotations

import sys
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


def _dbg(msg: str) -> None:
    print(f"[fast] {msg}", file=sys.stderr, flush=True)


def run_pipeline(*, conn, csv_path: Path, resolver: ResolverChain,
                 force: bool = False) -> dict:
    """Execute the fast ingest pipeline for one CSV. Returns a summary dict."""
    csv_path = Path(csv_path)
    t0 = time.monotonic()
    _ms = lambda t: int((time.monotonic() - t) * 1000)

    # 1. Load CSV into temp table
    _dbg(f"Loading CSV ({csv_path.name})")
    t = time.monotonic()
    temp_table, row_count = load_csv_to_temp(conn, csv_path)
    _dbg(f"  Loaded {row_count:,} rows in {_ms(t)}ms")
    log_stage(conn, snapshot_id=f"file:{csv_path.name}", stage="load",
              rows_affected=row_count, duration_ms=_ms(t))

    # 2. Snapshot id + idempotency
    snapshot_id = compute_snapshot_id(csv_path, row_count=row_count)
    snapshot_at = extract_snapshot_at(csv_path)
    if not force and is_already_ingested(conn, snapshot_id):
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        _dbg("Already ingested — skipping")
        return {"already_ingested": True, "snapshot_id": snapshot_id}

    conn.execute("BEGIN TRANSACTION")
    try:
        # 3. Record snapshot
        _dbg("Recording snapshot")
        record_snapshot(conn, snapshot_id=snapshot_id, snapshot_at=snapshot_at,
                        source_filename=csv_path.name, row_count=row_count)

        # 4. Register split functions and upsert entities
        _dbg("Upserting entities")
        t = time.monotonic()
        register_split_functions(conn)
        upsert_entities(conn, snapshot_at)
        _dbg(f"  Done in {_ms(t)}ms")
        log_stage(conn, snapshot_id=snapshot_id, stage="entities",
                  rows_affected=row_count, duration_ms=_ms(t))

        # 5. Ownership (reuse existing — needs Python for fnmatch)
        _dbg("Resolving ownership")
        t = time.monotonic()
        _resolve_and_upsert_ownership(conn, resolver)
        _dbg(f"  Done in {_ms(t)}ms")
        log_stage(conn, snapshot_id=snapshot_id, stage="ownership",
                  rows_affected=row_count, duration_ms=_ms(t))

        # 6. Runtime snapshot
        _dbg("Writing runtime snapshot")
        t = time.monotonic()
        write_runtime_snapshot(conn, snapshot_at)
        _dbg(f"  Done in {_ms(t)}ms")
        log_stage(conn, snapshot_id=snapshot_id, stage="runtime_snapshot",
                  rows_affected=row_count, duration_ms=_ms(t))

        # 7. Finding diff
        _dbg("Running finding diff")
        t = time.monotonic()
        diff_counts = diff_and_apply_findings(conn, snapshot_at)
        _dbg(f"  Done in {_ms(t)}ms — {diff_counts}")
        log_stage(conn, snapshot_id=snapshot_id, stage="finding_diff",
                  rows_affected=sum(diff_counts.values()), duration_ms=_ms(t))

        # 8. Rollups
        _dbg("Rebuilding rollups")
        t = time.monotonic()
        rebuild_rollups_for_date(conn, snapshot_at.date())
        _dbg(f"  Done in {_ms(t)}ms")
        log_stage(conn, snapshot_id=snapshot_id, stage="rollups",
                  rows_affected=0, duration_ms=_ms(t))

        total_ms = _ms(t0)
        log_stage(conn, snapshot_id=snapshot_id, stage="total",
                  rows_affected=row_count, duration_ms=total_ms)

        conn.execute("COMMIT")
        _dbg(f"Committed — total {total_ms}ms")
    except Exception:
        _dbg(f"Error — rolling back: {sys.exc_info()[1]}")
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
