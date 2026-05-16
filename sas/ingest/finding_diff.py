"""The heart of the ingest: diff today's findings against the current OPEN state.

Four transition types:
  - NEW         → INSERT OPEN row with first_seen = snapshot_at
  - RESEEN      → UPDATE last_seen, drift columns (severity, cvss, etc.)
  - REOPENED    → INSERT a new row with reopened_at, reopen_count=prev+1
  - DISAPPEARED → UPDATE state=CLOSED, reason_code computed from ReasonContext

Reason-code detection uses the graph snapshot taken THIS ingest:
  - image_still_runs_anywhere: look in workload_runs_image_daily for today
  - newer_digest_exists_without_cve: check image_in_repository for any image
    in the same repository with a later first_seen without this CVE
  - cve_missing_from_feed: was this CVE present in any row of today's CSV?
  - risk_accepted flip: read the prior OPEN row vs any incoming row referencing
    the same natural key
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, date, timedelta, timezone

import pandas as pd

from sas.ingest.reason_code import ReasonContext, compute_reason_code, GRACE_PERIOD_DAYS


def _dbg(msg: str) -> None:
    print(f"[old-diff] {msg}", file=sys.stderr, flush=True)


_DRIFT_COLUMNS = {
    "severity", "cvss_score", "fix_available", "fix_version",
    "risk_accepted", "public_exploit", "in_use", "cisa_kev_known_ransomware",
}


def _natural_key(r) -> tuple:
    return (
        r["image_id"], r["vulnerability_name"], r["package_name"],
        r["package_version"], r["package_path"],
    )


def _row_to_fs_values(r, snapshot_at: datetime):  # noqa: ARG001
    return {
        "image_id": r["image_id"],
        "cve_id": r["vulnerability_name"],
        "package_name": r["package_name"],
        "package_version": r["package_version"],
        "package_path": r["package_path"],
        "severity": r["vulnerability_severity"],
        "cvss_score": (
            float(r["cvss_score"]) if pd.notna(r["cvss_score"]) else None
        ),
        "in_use": bool(r["package_in_use"]),
        "fix_available": bool(r["fix_available"]),
        "fix_version": (
            r["fix_version"] if pd.notna(r["fix_version"]) else None
        ),
        "risk_accepted": bool(r["risk_accepted"]),
        "public_exploit": bool(r["public_exploit"]),
        "cisa_kev_known_ransomware": bool(r["cisa_kev_known_ransomware"]),  # NEW
    }


def diff_and_apply_findings(
    conn, df: pd.DataFrame, snapshot_at: datetime
) -> dict:
    """Compare today's findings against current OPEN state, apply transitions.

    Returns counts: {"new": N, "reseen": N, "reopened": N, "closed": N}
    """
    t0 = time.monotonic()
    _ms = lambda: int((time.monotonic() - t0) * 1000)
    counts = {"new": 0, "reseen": 0, "reopened": 0, "closed": 0}
    today = snapshot_at.date()
    today_cve_ids = set(df["vulnerability_name"].unique())

    # Today's natural keys
    _dbg(f"Building today_keys from {len(df):,} rows (iterrows)...")
    t = time.monotonic()
    today_keys = {_natural_key(r): r for _, r in df.iterrows()}
    _dbg(f"  ✓ {len(today_keys):,} unique natural keys in {int((time.monotonic()-t)*1000)}ms")

    # Current OPEN findings
    _dbg(f"Fetching OPEN findings...")
    t = time.monotonic()
    open_rows = conn.execute(
        """
        SELECT finding_id, image_id, cve_id, package_name, package_version,
               package_path, risk_accepted, first_seen, reopen_count
        FROM finding_state
        WHERE state = 'OPEN'
        """
    ).fetchall()
    open_by_key = {
        (r[1], r[2], r[3], r[4], r[5]): {
            "finding_id": r[0],
            "risk_accepted_was": bool(r[6]),
            "first_seen": r[7],
            "reopen_count": r[8] or 0,
        }
        for r in open_rows
    }
    _dbg(f"  ✓ {len(open_by_key):,} OPEN findings in {int((time.monotonic()-t)*1000)}ms")

    # 1. NEW + RESEEN + REOPENED
    _dbg(f"Processing {len(today_keys):,} today findings (per-row SQL)...")
    t = time.monotonic()
    for i, (key, r) in enumerate(today_keys.items()):
        v = _row_to_fs_values(r, snapshot_at)
        if key in open_by_key:
            # RESEEN — update last_seen, drift columns, days_open
            prior = open_by_key[key]
            days_open = (today - prior["first_seen"].date()).days
            conn.execute(
                """
                UPDATE finding_state SET
                  last_seen = ?, severity = ?, cvss_score = ?, in_use = ?,
                  fix_available = ?, fix_version = ?, risk_accepted = ?,
                  public_exploit = ?, cisa_kev_known_ransomware = ?, days_open = ?,
                  reason_code = NULL, grace_period_since = NULL
                WHERE finding_id = ?
                """,
                [snapshot_at, v["severity"], v["cvss_score"], v["in_use"],
                 v["fix_available"], v["fix_version"], v["risk_accepted"],
                 v["public_exploit"], v["cisa_kev_known_ransomware"],
                 days_open, prior["finding_id"]],
            )
            counts["reseen"] += 1
        else:
            # NEW or REOPENED — check closed history
            closed_prior = conn.execute(
                """
                SELECT finding_id, reopen_count FROM finding_state
                WHERE image_id = ? AND cve_id = ? AND package_name = ?
                  AND package_version = ? AND package_path = ?
                  AND state = 'CLOSED'
                ORDER BY closed_at DESC LIMIT 1
                """,
                [key[0], key[1], key[2], key[3], key[4]],
            ).fetchone()
            if closed_prior is not None:
                new_reopen_count = (closed_prior[1] or 0) + 1
                _insert_finding_row(
                    conn, v, snapshot_at,
                    reopened_at=snapshot_at,
                    reopen_count=new_reopen_count,
                    is_regression=True,
                )
                counts["reopened"] += 1
            else:
                _insert_finding_row(
                    conn, v, snapshot_at,
                    reopened_at=None,
                    reopen_count=0,
                    is_regression=False,
                )
                counts["new"] += 1
        # Progress every 10k
        if (i + 1) % 10000 == 0:
            _dbg(f"  progress: {i+1:,}/{len(today_keys):,} in {_ms()}ms")
    _dbg(f"  ✓ NEW/RESEEN/REOPENED done in {int((time.monotonic()-t)*1000)}ms — new={counts['new']}, reseen={counts['reseen']}, reopened={counts['reopened']}")

    # 2. DISAPPEARED — OPEN rows whose natural key wasn't in today
    disapp_count = sum(1 for key in open_by_key if key not in today_keys)
    _dbg(f"Processing {disapp_count:,} DISAPPEARED (grace period logic)...")
    t = time.monotonic()

    # 2a. Expire old STALE findings past grace period
    cutoff = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc) - timedelta(days=GRACE_PERIOD_DAYS)
    expired_rows = conn.execute(
        """
        UPDATE finding_state SET
          state = 'CLOSED',
          reason_code = 'REMEDIED',
          closed_at = ?,
          days_open = date_diff('day', first_seen, ?)
        WHERE reason_code = 'STALE'
          AND grace_period_since <= ?
        RETURNING finding_id
        """,
        [snapshot_at, today, cutoff],
    ).fetchall()
    expired_count = len(expired_rows)
    counts["closed"] += expired_count

    # 2b. Mark newly disappeared as STALE (exclude already-STALE)
    for key, prior in open_by_key.items():
        if key in today_keys:
            continue
        # Check if this finding is now CLOSED (expired above) or already STALE
        fs_row = conn.execute(
            "SELECT state, reason_code FROM finding_state WHERE finding_id = ?",
            [prior["finding_id"]],
        ).fetchone()
        if fs_row:
            if fs_row[0] == "CLOSED":
                continue
            if fs_row[1] == "STALE":
                continue  # Already STALE, don't reset grace_period_since

        conn.execute(
            """
            UPDATE finding_state SET
              reason_code = 'STALE',
              grace_period_since = ?
            WHERE finding_id = ?
            """,
            [snapshot_at, prior["finding_id"]],
        )
    _dbg(f"  ✓ DISAPPEARED done in {int((time.monotonic()-t)*1000)}ms — expired={expired_count}, stale={disapp_count - expired_count}")
    _dbg(f"Total diff: {_ms()}ms")

    return counts


def _insert_finding_row(
    conn, v, snapshot_at, *, reopened_at, reopen_count, is_regression
):
    conn.execute(
        """
        INSERT INTO finding_state (
          finding_id, image_id, cve_id, package_name, package_version,
          package_path, severity, cvss_score, in_use, fix_available,
          fix_version, risk_accepted, public_exploit, cisa_kev_known_ransomware,
          first_seen, last_seen,
          state, reason_code, closed_at, reopened_at, reopen_count,
          days_open, is_regression
        ) VALUES (
          nextval('seq_finding_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, 'OPEN', NULL, NULL, ?, ?, 0, ?
        )
        """,
        [v["image_id"], v["cve_id"], v["package_name"], v["package_version"],
         v["package_path"], v["severity"], v["cvss_score"], v["in_use"],
         v["fix_available"], v["fix_version"], v["risk_accepted"],
         v["public_exploit"], v["cisa_kev_known_ransomware"],
         snapshot_at, snapshot_at,
         reopened_at, reopen_count, is_regression],
    )


def _build_reason_context(
    conn, *, image_id: str, cve_id: str, risk_accepted_was: bool,
    today: date, today_cve_ids: set, df: pd.DataFrame,
) -> ReasonContext:
    # image_still_runs_anywhere: any workload running this image_id today?
    row = conn.execute(
        "SELECT 1 FROM workload_runs_image_daily "
        "WHERE date = ? AND image_id = ? LIMIT 1",
        [today, image_id],
    ).fetchone()
    image_still_runs = row is not None

    # newer_digest_exists_without_cve: find same repo, a digest observed
    # later, whose findings on today do not include this CVE
    repo_row = conn.execute(
        "SELECT repository FROM image_in_repository "
        "WHERE image_id = ? LIMIT 1",
        [image_id],
    ).fetchone()
    newer_without_cve = False
    if repo_row is not None:
        repo = repo_row[0]
        newer_digests = conn.execute(
            """
            SELECT iir.image_id FROM image_in_repository iir
            JOIN image i ON i.image_id = iir.image_id
            WHERE iir.repository = ? AND i.first_seen > (
              SELECT first_seen FROM image WHERE image_id = ?
            )
            """,
            [repo, image_id],
        ).fetchall()
        newer_ids = {r[0] for r in newer_digests}
        if newer_ids:
            today_with_cve = set(
                df.loc[
                    df["vulnerability_name"] == cve_id, "image_id"
                ].unique()
            )
            if not (newer_ids & today_with_cve):
                newer_without_cve = True

    # cve_missing_from_feed: did today's CSV contain this CVE at all?
    cve_missing = cve_id not in today_cve_ids

    # risk_accepted flip: did the same (image_id, cve_id) arrive today with
    # risk_accepted=True? Must match same natural key — a sibling finding
    # being accepted does not count.
    mask = (
        (df["image_id"] == image_id)
        & (df["vulnerability_name"] == cve_id)
        & (df["risk_accepted"] == True)  # noqa: E712
    )
    risk_is_now = bool(df[mask].shape[0] > 0)

    return ReasonContext(
        risk_accepted_was=risk_accepted_was,
        risk_accepted_is=risk_is_now and not risk_accepted_was,
        newer_digest_exists_without_cve=newer_without_cve,
        image_still_runs_anywhere=image_still_runs,
        cve_missing_from_feed=cve_missing,
    )
