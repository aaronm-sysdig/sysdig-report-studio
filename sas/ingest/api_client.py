"""Sysdig Reporting API client — discover and download scheduled reports."""
from __future__ import annotations

import gzip
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# Default host (eu1 region); override via SYSDIG_HOST env var.
_DEFAULT_HOST = "eu1.app.sysdig.com"


def _host() -> str:
    return os.environ.get("SYSDIG_HOST", _DEFAULT_HOST)


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    return s


def _get_json(token: str, path: str) -> Any:
    url = f"https://{_host()}{path}"
    with _session(token) as s:
        resp = s.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()


def _download_file(token: str, path: str, dest: Path) -> Path:
    """Download a binary report file, saving to *dest*."""
    url = f"https://{_host()}{path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _session(token) as s:
        resp = s.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    return dest


def list_schedules(token: str) -> list[dict]:
    """Return the list of configured report schedules."""
    return _get_json(token, "/api/scanning/reporting/v2/schedules")


def list_schedule_reports(token: str, schedule_id: str) -> list[dict]:
    """Return historical report runs for a given schedule."""
    return _get_json(
        token, f"/api/scanning/reporting/v2/schedules/{schedule_id}/reports"
    )


def download_report(
    token: str, schedule_id: str, report_id: str, dest: Path
) -> Path:
    """Download a specific report file by schedule+report ID."""
    return _download_file(
        token,
        f"/api/scanning/reporting/v2/schedules/{schedule_id}/reports/{report_id}/download",
        dest,
    )


def download_latest_report(token: str, schedule_id: str, dest: Path) -> Path:
    """Download the most recently generated report for a schedule."""
    return _download_file(
        token,
        f"/api/scanning/reporting/v2/schedules/{schedule_id}/download",
        dest,
    )


def decompress_report(archive_path: Path, output_dir: Path) -> Path:
    """Decompress a .gz or .zip report archive, returning the CSV path.

    If the file is already plain CSV (no recognised extension), return as-is.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".gz":
        csv_path = output_dir / archive_path.stem  # strip .gz
        with gzip.open(archive_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return csv_path

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                # Fallback: extract first file
                csv_files = zf.namelist()
            csv_path = output_dir / csv_files[0]
            with zf.open(csv_files[0]) as f_in, open(csv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            return csv_path

    # Already plain text / CSV
    return archive_path


def parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp to a timezone-aware datetime."""
    if not ts:
        return None
    # Handle 'Z' suffix
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def latest_snapshot_date(conn) -> datetime | None:
    """Return end-of-day for MAX(snapshot_at) from the snapshot table.

    Returns 23:59:59 UTC of the latest snapshot date so that reports
    completed on the same day are not re-fetched.
    """
    row = conn.execute("SELECT DATE(MAX(snapshot_at)) FROM snapshot").fetchone()
    d = row[0]
    if d is None:
        return None
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)


def find_missing_reports(
    token: str,
    schedules: list[dict],
    cutoff: datetime,
    *,
    schedule_id: str | None = None,
) -> list[tuple[str, str, str, datetime]]:
    """Discover reports whose completedAt > cutoff.

    If *schedule_id* is given, only that schedule is checked (bypasses
    the entityType filter so the user can sync any schedule they choose).

    Returns a list of (schedule_id, report_id, report_name, completed_at).
    """
    missing: list[tuple[str, str, str, datetime]] = []

    for sched in schedules:
        sid = sched["id"]

        # --report filter: exact ID match
        if schedule_id and sid != schedule_id:
            continue

        # Without --report, skip non-k8s schedules
        if not schedule_id and sched.get("entityType", "") != "k8s":
            continue

        schedule_name = sched.get("name", sid)

        try:
            reports = list_schedule_reports(token, sid)
        except Exception as exc:
            print(
                f"warning: could not list reports for '{schedule_name}': {exc}",
                file=sys.stderr,
            )
            continue

        for rpt in reports:
            completed = parse_iso(rpt.get("completedAt"))
            if completed and completed > cutoff:
                missing.append((
                    sid,
                    rpt["id"],
                    schedule_name,
                    completed,
                ))

    # Sort by completed_at so we ingest oldest-first
    missing.sort(key=lambda t: t[3])
    return missing


def sync_and_ingest(
    conn,
    token: str,
    resolver,
    *,
    force: bool = False,
    download_dir: Path | None = None,
    schedule_id: str | None = None,
) -> dict:
    """Full sync: discover missing reports, download, decompress, ingest.

    Returns a summary dict with counts.
    """
    if download_dir is None:
        download_dir = Path(os.environ.get(
            "SAS_DATA_DIR", str(Path.home() / "sysdig-vuln-data")
        )) / "downloads"

    download_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine cutoff
    cutoff = latest_snapshot_date(conn)
    if cutoff:
        print(f"latest snapshot in DB: {cutoff.date()}")
    else:
        print("no snapshots in DB — will fetch all available reports")
        cutoff = datetime(2000, 1, 1, tzinfo=timezone.utc)

    # 2. Discover schedules and missing reports
    print("fetching report schedules from API ...")
    schedules = list_schedules(token)
    missing = find_missing_reports(token, schedules, cutoff, schedule_id=schedule_id)

    if not missing:
        print("no missing reports — database is up to date")
        return {"downloaded": 0, "ingested": 0, "skipped": 0}

    print(f"found {len(missing)} missing report(s) to ingest:")
    for sid, rid, name, completed in missing:
        print(f"  [{completed.date()}] {name}  (report={rid})")

    # 3. Download and ingest each
    stats = {"downloaded": 0, "ingested": 0, "skipped": 0, "errors": 0}

    for schedule_id, report_id, schedule_name, completed in missing:
        safe_name = (
            schedule_name.replace(" ", "_")
            .replace("/", "_")
            .replace(":", "_")
        )
        date_str = completed.strftime("%Y_%m_%d")
        archive_name = f"{safe_name}_{date_str}.gz"
        archive_path = download_dir / archive_name

        print(f"\n--- {schedule_name} ({date_str}) ---")

        # Download (skip if already saved)
        if archive_path.exists():
            print(f"  archive exists: {archive_path.name} — skipping download")
        else:
            try:
                print(f"  downloading -> {archive_path.name} ...")
                download_report(token, schedule_id, report_id, archive_path)
                stats["downloaded"] += 1
            except Exception as exc:
                print(f"  ERROR downloading: {exc}", file=sys.stderr)
                stats["errors"] += 1
                continue

        # Decompress
        csv_path = download_dir / f"{safe_name}_{date_str}.csv"
        if csv_path.exists():
            print(f"  CSV exists: {csv_path.name} — skipping decompress")
        else:
            try:
                csv_path = decompress_report(archive_path, download_dir)
            except Exception as exc:
                print(f"  ERROR decompressing: {exc}", file=sys.stderr)
                stats["errors"] += 1
                continue

        # Ingest
        try:
            from sas.ingest.fast_pipeline import run_pipeline as _fp_run
            result = _fp_run(
                conn=conn, csv_path=csv_path,
                resolver=resolver, force=force,
            )
            if result.get("already_ingested"):
                print(f"  already ingested — skipping")
                stats["skipped"] += 1
            else:
                print(
                    f"  ingested {result['rows']} rows in "
                    f"{result['total_ms']}ms"
                )
                stats["ingested"] += 1
        except Exception as exc:
            print(f"  ERROR ingesting: {exc}", file=sys.stderr)
            stats["errors"] += 1

    return stats
