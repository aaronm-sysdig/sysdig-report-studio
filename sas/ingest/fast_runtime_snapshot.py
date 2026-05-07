"""Bulk runtime snapshot write — single SQL statement."""
from __future__ import annotations

from datetime import datetime


def write_runtime_snapshot(conn, snapshot_at: datetime) -> None:
    """Write workload_runs_image_daily from staging — single bulk INSERT."""
    snapshot_date = snapshot_at.date().isoformat()
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
