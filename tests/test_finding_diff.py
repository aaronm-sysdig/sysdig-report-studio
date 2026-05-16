from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from sas.ingest.schema import create_schema, migrate_schema
from sas.ingest.runtime_snapshot import write_runtime_snapshot
from sas.ingest.entity_upsert import upsert_entities
from sas.ingest.finding_diff import diff_and_apply_findings


def _row(image_id="sha256:aaa", cve="CVE-2026-00001", pkg="libfoo", ver="1.0",
         pkg_path="/lib/libfoo", severity="Critical", risk_accepted=False,
         fix_available=True, in_use=True, public_exploit=False):
    return {
        "vulnerability_name": cve,
        "vulnerability_severity": severity,
        "package_name": pkg,
        "package_version": ver,
        "package_type": "OS",
        "package_path": pkg_path,
        "image_name": "registry/foo:1.0",
        "os_name": "alpine 3.20",
        "cvss_version": "3.0",
        "cvss_score": 9.1,
        "cvss_vector": "AV:N",
        "disclosure_date": pd.Timestamp("2026-01-01T00:00:00Z"),
        "fix_available_date": pd.Timestamp("2026-01-02T00:00:00Z"),
        "fix_version": "1.1",
        "public_exploit": public_exploit,
        "kubernetes_cluster_name": "cluster-a",
        "kubernetes_namespace_name": "ns-a",
        "kubernetes_workload_type": "Deployment",
        "kubernetes_workload_name": "foo",
        "kubernetes_container_name": "foo-main",
        "image_id": image_id,
        "package_in_use": in_use,
        "risk_accepted": risk_accepted,
        "cisa_kev_publish_date": pd.NaT,
        "cisa_kev_due_date": pd.NaT,
        "cisa_kev_known_ransomware": False,
        "fix_available": fix_available,
        "agent_tags": "{}",
        "container_labels": "{}",
        "namespace_labels": "{}",
    }


def _prep(db, df, snapshot_at):
    upsert_entities(db, df, snapshot_at)
    write_runtime_snapshot(db, df, snapshot_at)


def test_new_finding_inserts_open_row(db):
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([_row()])
    _prep(db, df, day1)
    diff_and_apply_findings(db, df, day1)
    rows = db.execute(
        "SELECT cve_id, state, first_seen, last_seen FROM finding_state"
    ).fetchall()
    assert rows == [("CVE-2026-00001", "OPEN", day1, day1)]


def test_reseen_finding_updates_last_seen_only(db):
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    df = pd.DataFrame([_row()])
    _prep(db, df, day1); diff_and_apply_findings(db, df, day1)
    _prep(db, df, day2); diff_and_apply_findings(db, df, day2)
    count = db.execute("SELECT count(*) FROM finding_state").fetchone()[0]
    assert count == 1
    row = db.execute(
        "SELECT first_seen, last_seen, state FROM finding_state"
    ).fetchone()
    assert row[0] == day1
    assert row[1] == day2
    assert row[2] == "OPEN"


def test_disappeared_finding_becomes_stale(db):
    """Image disappears entirely → enters grace period as STALE (not immediately CLOSED)."""
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    df1 = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df1, day1); diff_and_apply_findings(db, df1, day1)
    # Day 2: completely different image in the CSV; sha256:aaa no longer seen
    df2 = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df2, day2); diff_and_apply_findings(db, df2, day2)

    rows = db.execute(
        "SELECT state, reason_code FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchall()
    assert rows == [("OPEN", "STALE")]


def test_disappeared_finding_becomes_stale_not_accepted(db):
    """Sibling risk_accepted=True on a DIFFERENT CVE does NOT trigger ACCEPTED.

    The finding disappears → STALE (grace period). ACCEPTED reason_code is
    only computed when the grace period expires, not on first disappearance.
    """
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    df1 = pd.DataFrame([_row(risk_accepted=False)])
    _prep(db, df1, day1); diff_and_apply_findings(db, df1, day1)
    # Day 2: same image running, CVE-2026-00001 gone, a different CVE present
    df2 = pd.DataFrame([_row(cve="CVE-2026-00002", risk_accepted=True)])
    _prep(db, df2, day2); diff_and_apply_findings(db, df2, day2)
    row = db.execute(
        "SELECT reason_code FROM finding_state WHERE cve_id = 'CVE-2026-00001'"
    ).fetchone()
    # Disappeared finding enters grace period as STALE
    assert row[0] == "STALE"


def test_reopened_finding_creates_new_record_and_increments_reopen_count(db):
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day6 = day1 + timedelta(days=5)  # grace period (3 days) has expired
    day7 = day1 + timedelta(days=6)
    df_with = pd.DataFrame([_row()])
    df_without = pd.DataFrame([_row(cve="CVE-2026-00002")])  # same image, different CVE

    _prep(db, df_with, day1); diff_and_apply_findings(db, df_with, day1)
    # Day 2: disappears → STALE
    _prep(db, df_without, day2); diff_and_apply_findings(db, df_without, day2)
    # Day 6: grace period expired → CLOSED/REMEDIED
    _prep(db, df_without, day6); diff_and_apply_findings(db, df_without, day6)
    # Day 7: reappears → REOPENED
    _prep(db, df_with, day7); diff_and_apply_findings(db, df_with, day7)

    rows = db.execute(
        "SELECT state, reopen_count, is_regression FROM finding_state "
        "WHERE cve_id = 'CVE-2026-00001' ORDER BY first_seen"
    ).fetchall()
    # Two rows for the same natural key: the closed original, and the reopened record.
    assert len(rows) == 2
    closed, reopened = rows
    assert closed[0] == "CLOSED"
    assert reopened[0] == "OPEN"
    assert reopened[1] == 1
    assert reopened[2] is True


def test_days_open_is_computed(db):
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day5 = day1 + timedelta(days=4)
    df = pd.DataFrame([_row()])
    _prep(db, df, day1); diff_and_apply_findings(db, df, day1)
    _prep(db, df, day5); diff_and_apply_findings(db, df, day5)
    row = db.execute("SELECT days_open FROM finding_state").fetchone()
    assert row[0] == 4
