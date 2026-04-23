from datetime import datetime, timezone, date
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.runtime_snapshot import write_runtime_snapshot


@pytest.fixture
def two_container_df():
    return pd.DataFrame([
        {
            "kubernetes_cluster_name": "cluster-a",
            "kubernetes_namespace_name": "ns-a",
            "kubernetes_workload_type": "Deployment",
            "kubernetes_workload_name": "foo",
            "kubernetes_container_name": "foo-main",
            "image_id": "sha256:aaa",
        },
        {
            "kubernetes_cluster_name": "cluster-a",
            "kubernetes_namespace_name": "ns-a",
            "kubernetes_workload_type": "Deployment",
            "kubernetes_workload_name": "foo",
            "kubernetes_container_name": "foo-main",
            "image_id": "sha256:aaa",
        },  # duplicate — should collapse to replica_count=2
        {
            "kubernetes_cluster_name": "cluster-a",
            "kubernetes_namespace_name": "ns-a",
            "kubernetes_workload_type": "Deployment",
            "kubernetes_workload_name": "bar",
            "kubernetes_container_name": "bar-main",
            "image_id": "sha256:bbb",
        },
    ])


def test_runtime_snapshot_aggregates_replica_counts(db, two_container_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    write_runtime_snapshot(db, two_container_df, snapshot_at)

    rows = db.execute(
        "SELECT workload_name, replica_count FROM workload_runs_image_daily "
        "ORDER BY workload_name"
    ).fetchall()
    assert rows == [("bar", 1), ("foo", 2)]


def test_runtime_snapshot_is_idempotent_on_same_day(db, two_container_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    write_runtime_snapshot(db, two_container_df, snapshot_at)
    write_runtime_snapshot(db, two_container_df, snapshot_at)
    count = db.execute("SELECT count(*) FROM workload_runs_image_daily").fetchone()[0]
    assert count == 2  # same 2 unique rows, not 4


def test_runtime_snapshot_uses_date_of_snapshot_at(db, two_container_df):
    create_schema(db)
    snapshot_at = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    write_runtime_snapshot(db, two_container_df, snapshot_at)
    row = db.execute(
        "SELECT date FROM workload_runs_image_daily LIMIT 1"
    ).fetchone()
    assert row[0] == date(2026, 4, 23)
