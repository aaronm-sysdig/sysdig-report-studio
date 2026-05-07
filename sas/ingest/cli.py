"""`python -m sas.ingest <csv>` — the public command."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from sas.ingest.config import get_config
from sas.ingest.schema import create_schema, migrate_schema
from sas.ingest.ownership import (
    ResolverChain, LabelStrategy, MappingFileStrategy, NamespaceFallback,
)
from sas.ingest.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sas.ingest",
        description="Ingest a Sysdig vulnerability CSV into the SAS analytics store.",
    )
    parser.add_argument("csv", type=Path, help="Path to the Sysdig CSV export")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest even if snapshot_id already recorded")
    parser.add_argument("--fast", action="store_true",
                        help="Use DuckDB-native pipeline (no Pandas, ~10x faster)")
    parser.add_argument("--legacy", action="store_true",
                        help="Legacy v1 format (gz, different columns, filters Critical+High)")
    args = parser.parse_args(argv)

    cfg = get_config()
    cfg.ensure_data_dir()

    resolver = _build_default_resolver(cfg)

    conn = duckdb.connect(str(cfg.duckdb_path))
    try:
        create_schema(conn)
        migrate_schema(conn)

        if args.legacy:
            from sas.ingest.fast_pipeline import run_pipeline as fast_run_pipeline
            from sas.ingest.legacy_loader import load_legacy_csv
            # Patch the loader for this run
            import sas.ingest.fast_pipeline as fp
            orig_load = fp.load_csv_to_temp
            fp.load_csv_to_temp = lambda c, p: load_legacy_csv(c, p, severities=['Critical', 'High'])
            try:
                result = fast_run_pipeline(
                    conn=conn, csv_path=args.csv,
                    resolver=resolver, force=args.force,
                )
            finally:
                fp.load_csv_to_temp = orig_load
        elif args.fast:
            from sas.ingest.fast_pipeline import run_pipeline as fast_run_pipeline
            result = fast_run_pipeline(
                conn=conn, csv_path=args.csv,
                resolver=resolver, force=args.force,
            )
        else:
            result = run_pipeline(
                conn=conn, csv_path=args.csv,
                resolver=resolver, force=args.force,
            )
    finally:
        conn.close()

    if result.get("already_ingested"):
        print(f"already ingested: snapshot_id={result['snapshot_id']} — skipping")
        return 0

    print(
        f"ingested {result['rows']} rows in {result['total_ms']}ms "
        f"(new={result.get('new', 0)} reseen={result.get('reseen', 0)} "
        f"reopened={result.get('reopened', 0)} closed={result.get('closed', 0)})"
    )
    return 0


def _build_default_resolver(cfg) -> ResolverChain:
    strategies = [
        LabelStrategy(label="team"),
        LabelStrategy(label="cost-center"),
    ]
    if cfg.ownership_mapping_path.exists():
        strategies.append(MappingFileStrategy(path=cfg.ownership_mapping_path))
    strategies.append(NamespaceFallback())
    return ResolverChain(strategies)


if __name__ == "__main__":
    sys.exit(main())
