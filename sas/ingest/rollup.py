"""Rebuild the daily_metrics_* rollup tables for a given date.

Idempotent: DELETE then INSERT for the target date. Always safe to re-run.

Counts by reason code follow the spec's categorization:
  - count_fixed_patched: CLOSED with reason_code=PATCHED
  - count_fixed_retired: CLOSED with reason_code IN (RETIRED, SCALED_TO_ZERO)
  - count_fixed_accepted: CLOSED with reason_code=ACCEPTED
  - count_fixed_other:   CLOSED with reason_code IN (FEED_WITHDRAWN, UNKNOWN, REMEDIED)
    (REMEDIED = fast pipeline's simplified code for DISAPPEARED findings)
"""
from __future__ import annotations

import sys
import time
from datetime import date


def _dbg(msg: str) -> None:
    print(f"[rollup] {msg}", file=sys.stderr, flush=True)


def rebuild_rollups_for_date(conn, target: date) -> None:
    t0 = time.monotonic()
    _ms = lambda: int((time.monotonic() - t0) * 1000)
    _dbg(f"Rebuilding rollups for {target}... {_ms()}ms")
    _rebuild_by_image(conn, target)
    _dbg(f"  by_image done {_ms()}ms")
    # _rebuild_by_workload(conn, target)  # DEPRECATED: inflated (image findings × workloads running them)
    # _dbg(f"  by_workload done {_ms()}ms")
    _rebuild_by_repository(conn, target)
    _dbg(f"  by_repository done {_ms()}ms")
    _rebuild_by_cluster_severity(conn, target)
    _dbg(f"  by_cluster_severity done {_ms()}ms")
    _dbg(f"All rollups done for {target} in {_ms()}ms")


def _rebuild_by_image(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_image WHERE date = ?", [target]
    )
    # Use daily_open_snapshot for historical OPEN counts (captured at ingest time).
    # Use daily_closed_snapshot for CLOSED transitions (images with no OPEN left).
    # Use finding_state for NEW and REOPENED transitions.
    conn.execute(
        """
        INSERT INTO daily_metrics_by_image (
          date, image_id,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed, mttr_sum, mttr_count
        )
        SELECT
          ? AS date,
          all_img.image_id,
          COALESCE(dos.count_open_critical, 0),
          COALESCE(dos.count_open_high, 0),
          COALESCE(dos.count_open_medium, 0),
          COALESCE(dos.count_open_low, 0),
          COALESCE(dos.count_open, 0),
          COALESCE(tr.count_new, 0),
          0 AS count_fixed_patched,
          0 AS count_fixed_retired,
          0 AS count_fixed_accepted,
          COALESCE(dcs.count_closed, 0) AS count_fixed_other,
          COALESCE(tr.count_regressed, 0),
          COALESCE(tr.mttr_sum, 0),
          COALESCE(tr.mttr_count, 0)
        FROM (
          SELECT image_id FROM daily_open_snapshot WHERE date = ?
          UNION
          SELECT image_id FROM daily_closed_snapshot WHERE date = ?
        ) all_img
        LEFT JOIN daily_open_snapshot dos ON dos.image_id = all_img.image_id AND dos.date = ?
        LEFT JOIN daily_closed_snapshot dcs ON dcs.image_id = all_img.image_id AND dcs.date = ?
        LEFT JOIN (
          SELECT
            image_id,
            SUM(CASE WHEN state='OPEN' AND CAST(first_seen AS DATE) = ? THEN 1 ELSE 0 END) AS count_new,
            SUM(CASE WHEN reopened_at IS NOT NULL
               AND CAST(reopened_at AS DATE) = ? THEN 1 ELSE 0 END) AS count_regressed,
            SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
               THEN days_open ELSE 0 END) AS mttr_sum,
            SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
               THEN 1 ELSE 0 END) AS mttr_count
          FROM finding_state
          GROUP BY image_id
        ) tr ON tr.image_id = all_img.image_id
        """,
        [target, target, target, target, target, target, target, target, target],
    )


# ---------------------------------------------------------------------------
# DEPRECATED: _rebuild_by_workload — produces inflated counts because image-level
# findings are summed once per workload running that image. Kept for reference.
# ---------------------------------------------------------------------------
def _rebuild_by_workload(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_workload WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_workload (
          date, cluster_name, namespace_name, workload_type, workload_name,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed, mttr_sum, mttr_count, replica_count
        )
        SELECT
          wri.date,
          wri.cluster_name, wri.namespace_name, wri.workload_type, wri.workload_name,
          SUM(COALESCE(dos.count_open_critical, 0)),
          SUM(COALESCE(dos.count_open_high, 0)),
          SUM(COALESCE(dos.count_open_medium, 0)),
          SUM(COALESCE(dos.count_open_low, 0)),
          SUM(COALESCE(dos.count_open, 0)),
          SUM(COALESCE(tr.count_new, 0)),
          0, 0, 0,
          SUM(COALESCE(dcs.count_closed, 0)),
          SUM(COALESCE(tr.count_regressed, 0)),
          SUM(COALESCE(tr.mttr_sum, 0)),
          SUM(COALESCE(tr.mttr_count, 0)),
          SUM(wri.replica_count)
        FROM (
          SELECT DISTINCT cluster_name, namespace_name, workload_type, workload_name, image_id, date, replica_count
          FROM workload_runs_image_daily WHERE date = ?
        ) wri
        LEFT JOIN daily_open_snapshot dos ON dos.image_id = wri.image_id AND dos.date = wri.date
        LEFT JOIN daily_closed_snapshot dcs ON dcs.image_id = wri.image_id AND dcs.date = wri.date
        LEFT JOIN (
          SELECT image_id,
            SUM(CASE WHEN state='OPEN' AND CAST(first_seen AS DATE) = ? THEN 1 ELSE 0 END) AS count_new,
            SUM(CASE WHEN reopened_at IS NOT NULL AND CAST(reopened_at AS DATE) = ? THEN 1 ELSE 0 END) AS count_regressed,
            SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ? THEN days_open ELSE 0 END) AS mttr_sum,
            SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ? THEN 1 ELSE 0 END) AS mttr_count
          FROM finding_state GROUP BY image_id
        ) tr ON tr.image_id = wri.image_id
        GROUP BY wri.date, wri.cluster_name, wri.namespace_name,
                 wri.workload_type, wri.workload_name
        """,
        [target, target, target, target, target],
    )


def _rebuild_by_repository(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_repository WHERE date = ?", [target]
    )
    # Use daily_open_snapshot for accurate per-image OPEN counts,
    # aggregate through distinct image->repository mapping.
    conn.execute(
        """
        INSERT INTO daily_metrics_by_repository (
          date, repository,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed
        )
        SELECT
          ? AS date,
          img_repo.repository,
          SUM(COALESCE(dos.count_open_critical, 0)),
          SUM(COALESCE(dos.count_open_high, 0)),
          SUM(COALESCE(dos.count_open_medium, 0)),
          SUM(COALESCE(dos.count_open_low, 0)),
          SUM(COALESCE(dos.count_open, 0)),
          SUM(COALESCE(tr.count_new, 0)),
          0, 0, 0,
          SUM(COALESCE(dcs.count_closed, 0)),
          SUM(COALESCE(tr.count_regressed, 0))
        FROM (SELECT DISTINCT image_id, repository FROM image_in_repository) img_repo
        LEFT JOIN daily_open_snapshot dos ON dos.image_id = img_repo.image_id AND dos.date = ?
        LEFT JOIN daily_closed_snapshot dcs ON dcs.image_id = img_repo.image_id AND dcs.date = ?
        LEFT JOIN (
          SELECT image_id,
            SUM(CASE WHEN state='OPEN' AND CAST(first_seen AS DATE) = ? THEN 1 ELSE 0 END) AS count_new,
            SUM(CASE WHEN reopened_at IS NOT NULL AND CAST(reopened_at AS DATE) = ? THEN 1 ELSE 0 END) AS count_regressed
          FROM finding_state GROUP BY image_id
        ) tr ON tr.image_id = img_repo.image_id
        GROUP BY img_repo.repository
        """,
        [target, target, target, target, target],
    )


def _rebuild_by_cluster_severity(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_cluster_severity WHERE date = ?", [target]
    )
    # Use daily_open_snapshot severity breakdown, pivot to rows.
    conn.execute(
        """
        INSERT INTO daily_metrics_by_cluster_severity (date, cluster_name, severity, count_open)
        SELECT ? AS date, cluster_name, severity, count_open FROM (
          SELECT wri.cluster_name,
            'Critical' AS severity, SUM(COALESCE(dos.count_open_critical, 0)) AS count_open
          FROM (SELECT DISTINCT image_id, cluster_name FROM workload_runs_image_daily WHERE date = ?) wri
          LEFT JOIN daily_open_snapshot dos ON dos.image_id = wri.image_id AND dos.date = ?
          GROUP BY wri.cluster_name
          UNION ALL
          SELECT wri.cluster_name,
            'High', SUM(COALESCE(dos.count_open_high, 0))
          FROM (SELECT DISTINCT image_id, cluster_name FROM workload_runs_image_daily WHERE date = ?) wri
          LEFT JOIN daily_open_snapshot dos ON dos.image_id = wri.image_id AND dos.date = ?
          GROUP BY wri.cluster_name
          UNION ALL
          SELECT wri.cluster_name,
            'Medium', SUM(COALESCE(dos.count_open_medium, 0))
          FROM (SELECT DISTINCT image_id, cluster_name FROM workload_runs_image_daily WHERE date = ?) wri
          LEFT JOIN daily_open_snapshot dos ON dos.image_id = wri.image_id AND dos.date = ?
          GROUP BY wri.cluster_name
          UNION ALL
          SELECT wri.cluster_name,
            'Low', SUM(COALESCE(dos.count_open_low, 0))
          FROM (SELECT DISTINCT image_id, cluster_name FROM workload_runs_image_daily WHERE date = ?) wri
          LEFT JOIN daily_open_snapshot dos ON dos.image_id = wri.image_id AND dos.date = ?
          GROUP BY wri.cluster_name
        )
        WHERE count_open > 0
        """,
        [target, target, target, target, target, target, target, target, target],
    )
