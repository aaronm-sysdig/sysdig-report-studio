"""`python -m sas.ingest <csv>` — the public command."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb

from sas.ingest.config import get_config
from sas.ingest.schema import create_schema, migrate_schema
from sas.ingest.ownership import (
    ResolverChain, LabelStrategy, MappingFileStrategy, NamespaceFallback,
)
from sas.ingest.pipeline import run_pipeline


def _print_snapshots(conn) -> int:
    row = conn.execute(
        "SELECT MIN(snapshot_at), MAX(snapshot_at), COUNT(*) FROM snapshot"
    ).fetchone()

    if row[0] is None:
        print("no snapshots ingested yet")
        return 0

    min_date, max_date, count = row
    print(f"earliest: {min_date.date()}")
    print(f"latest:   {max_date.date()}")
    print(f"total snapshots: {count}")
    return 0


def _list_reports(token: str) -> int:
    from sas.ingest.api_client import list_schedules

    schedules = list_schedules(token)
    print(f"{'ID':<34} {'Name':<42} {'Entity':<12} {'Schedule'}")
    print(f"{'-'*34} {'-'*42} {'-'*12} {'-'*20}")
    for s in schedules:
        print(f"{s['id']:<34} {s.get('name', ''):<42} {s.get('entityType', ''):<12} {s.get('schedule', '')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sas.ingest",
        description="Ingest a Sysdig vulnerability CSV into the SAS analytics store.",
        epilog=(
            "examples:\n"
            "  python -m sas.ingest findings.csv\n"
            "  python -m sas.ingest --fast findings.csv\n"
            "  python -m sas.ingest --snapshots\n"
            "  python -m sas.ingest --list-reports --api-token $SYSDIG_API_TOKEN\n"
            "  python -m sas.ingest --sync-reports --report SCHEDULE_ID --api-token $SYSDIG_API_TOKEN\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- CSV ingestion ---
    _csv = parser.add_argument_group("CSV ingestion")
    _csv.add_argument("csv", type=Path, nargs="?", default=None,
                      help="Path to the Sysdig CSV export")
    _csv.add_argument("--force", action="store_true",
                      help="Re-ingest even if snapshot_id already recorded")
    _csv.add_argument("--fast", action="store_true",
                      help="Use DuckDB-native pipeline (no Pandas, ~10x faster)")
    _csv.add_argument("--legacy", action="store_true",
                      help="Legacy v1 format (gz, different columns, filters Critical+High)")

    # --- Database info ---
    _db = parser.add_argument_group("database info")
    _db.add_argument("--snapshots", action="store_true",
                     help="Print earliest/latest snapshot dates and exit")

    # --- API sync ---
    _api = parser.add_argument_group("API sync (--api-token required)")
    _api.add_argument("--list-reports", action="store_true",
                      help="List available report schedules (id, name, entity, schedule) and exit")
    _api.add_argument("--sync-reports", action="store_true",
                      help="Sync reports from Sysdig API: discover, download missing, ingest")
    _api.add_argument("--report", type=str, default=None,
                      help="Schedule ID to sync (required with --sync-reports)")
    _api.add_argument("--api-token", type=str, default=None,
                      help="Sysdig API bearer token "
                      "(or set SECURE_API_TOKEN / SYSDIG_API_TOKEN env vars)")
    _api.add_argument("--download-path", type=Path, default=None,
                      help="Directory to save downloaded archives and CSVs "
                      "(default: ~/sysdig-vuln-data/downloads)")
    args = parser.parse_args(argv)

    modes = [args.snapshots, args.sync_reports, args.list_reports]
    active_modes = sum(modes)

    if (argv is None or argv == []) and active_modes == 0:
        parser.print_help()
        return 0

    if active_modes == 0 and args.csv is None:
        parser.error(
            "the csv argument is required when not using --snapshots, "
            "--sync-reports, or --list-reports"
        )

    if active_modes > 1:
        parser.error("use only one of --snapshots, --sync-reports, --list-reports")

    # Resolve API token for any mode that needs it
    if (args.sync_reports or args.list_reports) and not args.api_token:
        args.api_token = (
            os.environ.get("SECURE_API_TOKEN")
            or os.environ.get("SYSDIG_API_TOKEN")
        )
    if (args.sync_reports or args.list_reports) and not args.api_token:
        parser.error(
            "--api-token is required for --sync-reports and --list-reports "
            "(pass it on the CLI or set SECURE_API_TOKEN or SYSDIG_API_TOKEN)"
        )

    if args.sync_reports and not args.report:
        parser.error("--report SCHEDULE_ID is required with --sync-reports")

    cfg = get_config()
    cfg.ensure_data_dir()

    conn = duckdb.connect(str(cfg.duckdb_path))
    try:
        create_schema(conn)
        migrate_schema(conn)

        if args.snapshots:
            return _print_snapshots(conn)

        if args.list_reports:
            return _list_reports(args.api_token)

        if args.sync_reports:
            resolver = _build_default_resolver(cfg)
            from sas.ingest.api_client import sync_and_ingest
            stats = sync_and_ingest(
                conn, args.api_token, resolver,
                force=args.force,
                schedule_id=args.report,
                download_dir=args.download_path,
            )
            print(f"\n--- sync complete ---")
            print(f"  downloaded: {stats['downloaded']}")
            print(f"  ingested:   {stats['ingested']}")
            print(f"  skipped:    {stats['skipped']}")
            if stats.get("errors"):
                print(f"  errors:     {stats['errors']}")
            return 0

        resolver = _build_default_resolver(cfg)

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
