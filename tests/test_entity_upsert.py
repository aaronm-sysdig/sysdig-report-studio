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
