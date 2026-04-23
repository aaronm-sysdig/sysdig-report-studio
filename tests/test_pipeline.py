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
