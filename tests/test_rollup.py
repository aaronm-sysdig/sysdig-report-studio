from datetime import datetime, timezone, date
import pandas as pd
import pytest

from sas.ingest.schema import create_schema, migrate_schema
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
    migrate_schema(db)
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
    migrate_schema(db)
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
    migrate_schema(db)
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
