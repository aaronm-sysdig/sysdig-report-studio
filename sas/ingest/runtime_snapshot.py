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
