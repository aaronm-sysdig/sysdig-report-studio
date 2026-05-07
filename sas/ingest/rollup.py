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

from datetime import date


def rebuild_rollups_for_date(conn, target: date) -> None:
    _rebuild_by_image(conn, target)
    _rebuild_by_workload(conn, target)
    _rebuild_by_team(conn, target)
    _rebuild_by_repository(conn, target)
    _rebuild_by_cluster_severity(conn, target)


def _rebuild_by_image(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_image WHERE date = ?", [target]
    )
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
          image_id,
          SUM(CASE WHEN state='OPEN' AND severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='OPEN' AND CAST(first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     AND reason_code IN ('FEED_WITHDRAWN','UNKNOWN','REMEDIED') THEN 1 ELSE 0 END),
          SUM(CASE WHEN reopened_at IS NOT NULL
                     AND CAST(reopened_at AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     THEN days_open ELSE 0 END),
          SUM(CASE WHEN state='CLOSED' AND CAST(closed_at AS DATE) = ?
                     THEN 1 ELSE 0 END)
        FROM finding_state
        GROUP BY image_id
        """,
        [target, target, target, target, target, target, target, target, target],
    )


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
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND CAST(fs.first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('FEED_WITHDRAWN','UNKNOWN','REMEDIED') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.reopened_at IS NOT NULL
                     AND CAST(fs.reopened_at AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN fs.days_open ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN 1 ELSE 0 END),
          SUM(wri.replica_count)
        FROM workload_runs_image_daily wri
        LEFT JOIN finding_state fs ON fs.image_id = wri.image_id
        WHERE wri.date = ?
        GROUP BY wri.date, wri.cluster_name, wri.namespace_name,
                 wri.workload_type, wri.workload_name
        """,
        [target, target, target, target, target, target, target, target, target],
    )


def _rebuild_by_team(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_team WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_team (
          date, team_id,
          count_open_critical, count_open_high, count_open_medium, count_open_low,
          count_open, count_new,
          count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
          count_regressed, mttr_sum, mttr_count
        )
        SELECT
          ? AS date,
          wo.team_id,
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND CAST(fs.first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('FEED_WITHDRAWN','UNKNOWN','REMEDIED') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.reopened_at IS NOT NULL
                     AND CAST(fs.reopened_at AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN fs.days_open ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     THEN 1 ELSE 0 END)
        FROM workload_runs_image_daily wri
        JOIN workload_owned_by wo ON
             wo.cluster_name = wri.cluster_name
         AND wo.namespace_name = wri.namespace_name
         AND wo.workload_type = wri.workload_type
         AND wo.workload_name = wri.workload_name
        LEFT JOIN finding_state fs ON fs.image_id = wri.image_id
        WHERE wri.date = ?
        GROUP BY wo.team_id
        """,
        [target, target, target, target, target, target, target, target, target, target],
    )


def _rebuild_by_repository(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_repository WHERE date = ?", [target]
    )
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
          iir.repository,
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Critical' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='High' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Medium' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND fs.severity='Low' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='OPEN' AND CAST(fs.first_seen AS DATE) = ? THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='PATCHED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('RETIRED','SCALED_TO_ZERO') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code='ACCEPTED' THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.state='CLOSED' AND CAST(fs.closed_at AS DATE) = ?
                     AND fs.reason_code IN ('FEED_WITHDRAWN','UNKNOWN','REMEDIED') THEN 1 ELSE 0 END),
          SUM(CASE WHEN fs.reopened_at IS NOT NULL
                     AND CAST(fs.reopened_at AS DATE) = ? THEN 1 ELSE 0 END)
        FROM image_in_repository iir
        LEFT JOIN finding_state fs ON fs.image_id = iir.image_id
        GROUP BY iir.repository
        """,
        [target, target, target, target, target, target, target],
    )


def _rebuild_by_cluster_severity(conn, target: date) -> None:
    conn.execute(
        "DELETE FROM daily_metrics_by_cluster_severity WHERE date = ?", [target]
    )
    conn.execute(
        """
        INSERT INTO daily_metrics_by_cluster_severity (date, cluster_name, severity, count_open)
        SELECT
          ? AS date, wri.cluster_name, fs.severity, COUNT(DISTINCT fs.finding_id)
        FROM workload_runs_image_daily wri
        JOIN finding_state fs ON fs.image_id = wri.image_id
        WHERE wri.date = ? AND fs.state = 'OPEN'
        GROUP BY wri.cluster_name, fs.severity
        """,
        [target, target],
    )
