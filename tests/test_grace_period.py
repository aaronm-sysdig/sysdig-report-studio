"""Grace period tests: STALE state, expiry, reappearance."""
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


def test_disappeared_finding_becomes_stale(db):
    """Finding disappears on day 2 → state stays OPEN, reason_code=STALE, grace_period_since set."""
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)

    df_with = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df_with, day1)
    diff_and_apply_findings(db, df_with, day1)

    df_without = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df_without, day2)
    diff_and_apply_findings(db, df_without, day2)

    row = db.execute(
        "SELECT state, reason_code, grace_period_since FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] == "STALE"
    assert row[2] == day2


def test_stale_finding_reappears_clears_stale(db):
    """Finding disappears (STALE) then reappears → STALE is cleared, finding is normal OPEN."""
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=2)

    df_with = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df_with, day1)
    diff_and_apply_findings(db, df_with, day1)

    # Day 2: disappears → STALE
    df_without = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df_without, day2)
    diff_and_apply_findings(db, df_without, day2)

    row = db.execute(
        "SELECT state, reason_code, grace_period_since FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] == "STALE"

    # Day 3: reappears → STALE cleared
    _prep(db, df_with, day3)
    diff_and_apply_findings(db, df_with, day3)

    row = db.execute(
        "SELECT state, reason_code, grace_period_since FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] is None
    assert row[2] is None


def test_stale_finding_expires_after_grace_period(db):
    """STALE finding that stays absent for 3+ days → CLOSED/REMEDIED."""
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day5 = day1 + timedelta(days=4)  # 3 days after day2

    df_with = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df_with, day1)
    diff_and_apply_findings(db, df_with, day1)

    # Day 2: disappears → STALE
    df_without = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df_without, day2)
    diff_and_apply_findings(db, df_without, day2)

    row = db.execute(
        "SELECT state, reason_code FROM finding_state WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] == "STALE"

    # Day 5: grace period expired → CLOSED
    _prep(db, df_without, day5)
    diff_and_apply_findings(db, df_without, day5)

    row = db.execute(
        "SELECT state, reason_code, closed_at FROM finding_state WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row[0] == "CLOSED"
    assert row[1] == "REMEDIED"
    assert row[2] == day5


def test_stale_finding_not_expired_before_grace_period(db):
    """STALE finding at day 2 of grace period (not yet 3 days) → still STALE, not closed."""
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day4 = day1 + timedelta(days=3)  # only 2 days after stale

    df_with = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df_with, day1)
    diff_and_apply_findings(db, df_with, day1)

    # Day 2: disappears → STALE
    df_without = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df_without, day2)
    diff_and_apply_findings(db, df_without, day2)

    # Day 4: only 2 days later → still STALE
    _prep(db, df_without, day4)
    diff_and_apply_findings(db, df_without, day4)

    row = db.execute(
        "SELECT state, reason_code FROM finding_state WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] == "STALE"
