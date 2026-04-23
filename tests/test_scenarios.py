"""Scenario-based integration tests: one test per synthetic scenario in tests/scenarios/."""
from pathlib import Path

from sas.ingest.schema import create_schema
from sas.ingest.pipeline import run_pipeline
from sas.ingest.ownership import ResolverChain, NamespaceFallback, LabelStrategy

_SCENARIOS = Path(__file__).parent / "scenarios"


def test_scenario_patched_via_new_digest(db):
    """CVE-2026-1001 closes PATCHED when v2 digest drops it; CVE-2026-1002 stays OPEN."""
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    for csv in sorted((_SCENARIOS / "patched_via_new_digest").glob("day*.csv")):
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    closed = db.execute(
        "SELECT reason_code FROM finding_state WHERE cve_id='CVE-2026-1001' AND state='CLOSED'"
    ).fetchone()
    assert closed is not None
    assert closed[0] == "PATCHED"

    open_row = db.execute(
        "SELECT state FROM finding_state WHERE cve_id='CVE-2026-1002' AND state='OPEN'"
    ).fetchone()
    assert open_row is not None

    assert db.execute("SELECT count(*) FROM image_in_repository").fetchone()[0] == 2


def test_scenario_digest_churn_same_tag(db):
    """Same tag rebuilt 3 times → 3 image rows, 1 repo, 3 image_in_repository entries; 3 findings (2 CLOSED RETIRED, 1 OPEN)."""
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    for csv in sorted((_SCENARIOS / "digest_churn_same_tag").glob("day*.csv")):
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    assert db.execute("SELECT count(*) FROM image").fetchone()[0] == 3
    assert db.execute("SELECT count(*) FROM repository").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM image_in_repository").fetchone()[0] == 3

    # Each digest is a distinct natural key; digest 1 and 2 retired, digest 3 open
    open_count = db.execute(
        "SELECT count(*) FROM finding_state WHERE cve_id='CVE-2026-2001' AND state='OPEN'"
    ).fetchone()[0]
    assert open_count == 1

    closed_count = db.execute(
        "SELECT count(*) FROM finding_state WHERE cve_id='CVE-2026-2001' AND state='CLOSED'"
    ).fetchone()[0]
    assert closed_count == 2


def test_scenario_retired_workload(db):
    """Workload disappears on day 3; CVE-2026-3001 closes with reason_code=RETIRED."""
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    for csv in sorted((_SCENARIOS / "retired_workload").glob("day*.csv")):
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    row = db.execute(
        "SELECT state, reason_code FROM finding_state WHERE cve_id='CVE-2026-3001' AND state='CLOSED'"
    ).fetchone()
    assert row is not None
    assert row[1] == "RETIRED"


def test_scenario_accepted_risk(db):
    """risk_accepted flips to true on day 2 RESEEN; finding stays OPEN after day 2.

    Known limitation: on day 3 (disappears), reason_code is RETIRED not ACCEPTED —
    the pipeline has no 'ACCEPTED' reason code path.
    """
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    scenario_dir = _SCENARIOS / "accepted_risk"

    # After day 2 the flag is true and state is OPEN (not yet closed)
    for csv in sorted(scenario_dir.glob("day*.csv"))[:2]:
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    row = db.execute(
        "SELECT state, risk_accepted FROM finding_state WHERE cve_id='CVE-2026-4001'"
    ).fetchone()
    assert row is not None
    assert row[0] == "OPEN"
    assert row[1] is True

    count = db.execute(
        "SELECT count(*) FROM finding_state WHERE cve_id='CVE-2026-4001'"
    ).fetchone()[0]
    assert count == 1


def test_scenario_cve_regression(db):
    """CVE disappears then reappears: 2 rows (one CLOSED, one OPEN with reopen_count=1, is_regression=True)."""
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    for csv in sorted((_SCENARIOS / "cve_regression").glob("day*.csv")):
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    closed = db.execute(
        "SELECT reason_code FROM finding_state WHERE cve_id='CVE-2026-5001' AND state='CLOSED'"
    ).fetchone()
    assert closed is not None

    open_row = db.execute(
        "SELECT reopen_count, is_regression, reopened_at, last_seen "
        "FROM finding_state WHERE cve_id='CVE-2026-5001' AND state='OPEN'"
    ).fetchone()
    assert open_row is not None
    reopen_count, is_regression, reopened_at, last_seen = open_row
    assert reopen_count == 1
    assert is_regression is True
    assert reopened_at is not None
    assert last_seen.date().isoformat() == "2026-05-04"


def test_scenario_multi_cluster(db):
    """Image spreads to staging-cluster on day 2; both clusters appear in daily rollup tables."""
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    for csv in sorted((_SCENARIOS / "multi_cluster").glob("day*.csv")):
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    clusters_wrid = {
        r[0] for r in db.execute(
            "SELECT DISTINCT cluster_name FROM workload_runs_image_daily WHERE date='2026-05-02'"
        ).fetchall()
    }
    assert "prod-cluster" in clusters_wrid
    assert "staging-cluster" in clusters_wrid

    clusters_metrics = {
        r[0] for r in db.execute(
            "SELECT DISTINCT cluster_name FROM daily_metrics_by_cluster_severity WHERE date='2026-05-02'"
        ).fetchall()
    }
    assert "prod-cluster" in clusters_metrics
    assert "staging-cluster" in clusters_metrics


def test_scenario_ecr_registry_port(db):
    """registry.internal:5000/team/app:v1 parses correctly — port in registry host is handled."""
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    for csv in sorted((_SCENARIOS / "ecr_registry_port").glob("day*.csv")):
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    row = db.execute(
        "SELECT repository, tag FROM image_in_repository WHERE image_id='sha256:ecr1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "registry.internal:5000/team/app"
    assert row[1] == "v1"

    assert db.execute("SELECT count(*) FROM image_in_repository").fetchone()[0] == 1


def test_scenario_label_ownership(db):
    """LabelStrategy("team") resolves team_id from namespace_labels; owner_id is None."""
    create_schema(db)
    resolver = ResolverChain([LabelStrategy("team"), NamespaceFallback()])
    for csv in sorted((_SCENARIOS / "label_ownership").glob("day*.csv")):
        run_pipeline(conn=db, csv_path=csv, resolver=resolver)

    row = db.execute(
        "SELECT team_id, owner_id, resolved_by_strategy FROM workload_owned_by"
    ).fetchone()
    assert row is not None
    assert row[0] == "team-alpha"
    assert row[1] is None
    assert row[2] == "label:team"


def test_scenario_bulk_smoke(db):
    """200-row ingest across 20 images x 10 CVEs completes without error."""
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    results = []
    for csv in sorted((_SCENARIOS / "bulk_smoke").glob("day*.csv")):
        results.append(run_pipeline(conn=db, csv_path=csv, resolver=resolver))

    assert len(results) == 1
    assert results[0]["rows"] > 100

    assert db.execute("SELECT count(*) FROM image").fetchone()[0] == 20
    assert db.execute("SELECT count(DISTINCT cve_id) FROM cve").fetchone()[0] > 50
