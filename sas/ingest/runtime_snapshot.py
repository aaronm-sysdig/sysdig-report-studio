"""Writes one row per (date, cluster, namespace, workload, container, image_id).

The Sysdig vuln CSV has one row per (vulnerability × package × container), so
duplicate rows for the same container carry no pod-count signal. replica_count
is stored as 1 — meaning "this workload/image combination was observed running
on this day." A richer data source (e.g. kubectl feed) can upgrade this later.
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
    unique = df[_GROUP_COLS].drop_duplicates()
    snapshot_date = snapshot_at.date()

    for _, r in unique.iterrows():
        conn.execute(
            """
            INSERT INTO workload_runs_image_daily
              (date, cluster_name, namespace_name, workload_type, workload_name,
               container_name, image_id, replica_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT (date, cluster_name, namespace_name, workload_type,
                         workload_name, container_name, image_id)
            DO NOTHING
            """,
            [
                snapshot_date,
                r["kubernetes_cluster_name"], r["kubernetes_namespace_name"],
                r["kubernetes_workload_type"], r["kubernetes_workload_name"],
                r["kubernetes_container_name"], r["image_id"],
            ],
        )
