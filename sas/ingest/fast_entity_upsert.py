"""Bulk entity upserts from staging table — pure SQL, no iteration."""
from __future__ import annotations

from datetime import datetime


def register_split_functions(conn) -> None:
    """Register SQL scalar functions for splitting image_name into repo + tag."""

    def _split_repo(image_name: str) -> str:
        if image_name is None or not image_name:
            return None
        if "@" in image_name:
            repo, _ = image_name.rsplit("@", 1)
            return repo
        if ":" in image_name and "/" not in image_name.rsplit(":", 1)[-1]:
            repo, _ = image_name.rsplit(":", 1)
            return repo
        return image_name

    def _split_tag(image_name: str) -> str:
        if image_name is None or not image_name:
            return None
        if "@" in image_name:
            _, digest = image_name.rsplit("@", 1)
            return digest
        if ":" in image_name and "/" not in image_name.rsplit(":", 1)[-1]:
            _, tag = image_name.rsplit(":", 1)
            return tag
        return "latest"

    conn.create_function("_repo", _split_repo)
    conn.create_function("_tag", _split_tag)


def upsert_entities(conn, snapshot_at: datetime) -> None:
    """Upsert all entity rows from _ingest_staging using bulk SQL."""
    t = snapshot_at.isoformat()

    # --- image ---
    conn.execute(f"""
        INSERT INTO image (image_id, os_name, first_seen, last_seen,
                           current_repository, current_tag)
        SELECT
            image_id,
            MAX(os_name),
            '{t}'::timestamptz,
            '{t}'::timestamptz,
            _repo(MAX(image_name)),
            _tag(MAX(image_name))
        FROM _ingest_staging
        GROUP BY image_id
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
        WHERE _repo(image_name) != ''
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
            MAX(disclosure_date),
            MAX(fix_available_date),
            MAX(cvss_version),
            MAX(vulnerability_severity),
            MAX(cisa_kev_publish_date),
            MAX(cisa_kev_due_date),
            MAX(cisa_kev_known_ransomware),
            '{t}'::timestamptz,
            '{t}'::timestamptz
        FROM _ingest_staging
        GROUP BY vulnerability_name
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
