from datetime import datetime, timezone
import pandas as pd
import pytest

from sas.ingest.schema import create_schema
from sas.ingest.pipeline import run_pipeline
from sas.ingest.ownership import ResolverChain, NamespaceFallback


def test_run_pipeline_end_to_end_on_minimal_csv(db, fixtures_dir):
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    result = run_pipeline(
        conn=db,
        csv_path=fixtures_dir / "minimal_valid.csv",
        resolver=resolver,
    )
    assert result["new"] >= 1
    assert result["snapshot_id"]
    # verify data landed
    assert db.execute("SELECT count(*) FROM finding_state").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM daily_metrics_by_image").fetchone()[0] >= 1
    assert db.execute("SELECT count(*) FROM snapshot").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM ingest_log").fetchone()[0] >= 1


def test_rerunning_same_csv_is_noop(db, fixtures_dir):
    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])
    run_pipeline(conn=db, csv_path=fixtures_dir / "minimal_valid.csv", resolver=resolver)
    result2 = run_pipeline(conn=db, csv_path=fixtures_dir / "minimal_valid.csv", resolver=resolver)
    assert result2["already_ingested"] is True


def test_pipeline_rolls_back_on_mid_flight_failure(db, fixtures_dir, monkeypatch):
    """If a stage mid-pipeline raises, all prior writes in the same ingest roll back.

    This guarantees re-running after a crash is a clean retry, not a partial-state recovery.
    """
    from sas.ingest.schema import create_schema
    from sas.ingest.pipeline import run_pipeline
    from sas.ingest.ownership import ResolverChain, NamespaceFallback
    import sas.ingest.pipeline as pipeline_mod

    create_schema(db)
    resolver = ResolverChain([NamespaceFallback()])

    # Force the rollup stage (late in the pipeline) to explode.
    # Patch the name in pipeline's own namespace (direct import binding).
    def boom(*a, **kw):
        raise RuntimeError("simulated rollup failure")
    monkeypatch.setattr(pipeline_mod, "rebuild_rollups_for_date", boom)

    with pytest.raises(RuntimeError, match="simulated rollup failure"):
        run_pipeline(
            conn=db,
            csv_path=fixtures_dir / "minimal_valid.csv",
            resolver=resolver,
        )

    # Everything must have rolled back: no snapshot, no entities, no findings, no rollups.
    assert db.execute("SELECT count(*) FROM snapshot").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM image").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM finding_state").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM ingest_log").fetchone()[0] == 0
