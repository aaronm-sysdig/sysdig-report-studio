"""Top-level ingest orchestration. Composes all the steps per spec §6."""
from __future__ import annotations

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
