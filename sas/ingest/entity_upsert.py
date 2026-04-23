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
