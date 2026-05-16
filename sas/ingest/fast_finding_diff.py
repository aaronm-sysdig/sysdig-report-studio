"""Set-based finding diff — bulk SQL for all transitions."""
from __future__ import annotations

from datetime import datetime, timedelta

from sas.ingest.reason_code import GRACE_PERIOD_DAYS


def diff_and_apply_findings(conn, snapshot_at: datetime) -> dict:
    """Compare today's findings against current OPEN state, apply transitions.

    Fully set-based: bulk UPDATE for RESEEN, bulk INSERT for NEW and REOPENED,
    bulk UPDATE for DISAPPEARED (with reason code fallback).
    """
    today = snapshot_at.date()
    t = snapshot_at.isoformat()

    # 1. Create temp table of today's findings (deduplicated by natural key)
    # GROUP BY natural key — same image+cve+package can have different attr values in CSV
    conn.execute("""
        CREATE OR REPLACE TEMPORARY TABLE _today_findings AS
        SELECT
            image_id,
            vulnerability_name AS cve_id,
            package_name,
            package_version,
            package_path,
            MAX(vulnerability_severity) AS severity,
            MAX(CAST(cvss_score AS DOUBLE)) AS cvss_score,
            BOOL_OR(package_in_use) AS in_use,
            BOOL_OR(fix_available) AS fix_available,
            ANY_VALUE(fix_version) AS fix_version,
            BOOL_OR(risk_accepted) AS risk_accepted,
            BOOL_OR(public_exploit) AS public_exploit,
            BOOL_OR(cisa_kev_known_ransomware) AS cisa_kev_known_ransomware
        FROM _ingest_staging
        GROUP BY image_id, vulnerability_name, package_name, package_version, package_path
    """)

    # 2. LEFT JOIN against OPEN: classify RESEEN vs NEW_OR_REOPENED
    conn.execute("""
        CREATE OR REPLACE TEMPORARY TABLE _classified AS
        SELECT
            tf.*,
            fs.finding_id,
            fs.risk_accepted AS risk_accepted_was,
            fs.first_seen AS prior_first_seen,
            fs.reopen_count AS prior_reopen_count,
            CASE
                WHEN fs.finding_id IS NOT NULL THEN 'RESEEN'
                ELSE 'NEW_OR_REOPENED'
            END AS transition
        FROM _today_findings tf
        LEFT JOIN finding_state fs ON
            fs.image_id = tf.image_id
            AND fs.cve_id = tf.cve_id
            AND fs.package_name = tf.package_name
            AND fs.package_version = tf.package_version
            AND fs.package_path = tf.package_path
            AND fs.state = 'OPEN'
    """)

    # 3. Handle RESEEN — bulk UPDATE
    conn.execute(f"""
        UPDATE finding_state SET
            last_seen = '{t}'::timestamptz,
            severity = c.severity,
            cvss_score = c.cvss_score,
            in_use = c.in_use,
            fix_available = c.fix_available,
            fix_version = c.fix_version,
            risk_accepted = c.risk_accepted,
            public_exploit = c.public_exploit,
            cisa_kev_known_ransomware = c.cisa_kev_known_ransomware,
            days_open = date_diff('day', c.prior_first_seen, '{today}'::date),
            reason_code = NULL,
            grace_period_since = NULL
        FROM _classified c
        WHERE c.transition = 'RESEEN'
          AND c.finding_id = finding_state.finding_id
    """)
    resseen_count = conn.execute(
        "SELECT COUNT(*) FROM _classified "
        "WHERE transition = 'RESEEN' AND finding_id IS NOT NULL"
    ).fetchone()[0]

    # 4. Classify NEW vs REOPENED with a single LEFT JOIN against CLOSED history
    conn.execute("""
        CREATE OR REPLACE TEMPORARY TABLE _new_vs_reopened AS
        SELECT
            c.*,
            cl.finding_id AS closed_finding_id,
            cl.reopen_count AS closed_reopen_count
        FROM _classified c
        LEFT JOIN (
            SELECT * FROM finding_state fs2
            WHERE state = 'CLOSED'
              AND (fs2.image_id, fs2.cve_id, fs2.package_name,
                   fs2.package_version, fs2.package_path, fs2.closed_at) IN (
                SELECT image_id, cve_id, package_name, package_version,
                       package_path, MAX(closed_at)
                FROM finding_state
                WHERE state = 'CLOSED'
                GROUP BY image_id, cve_id, package_name, package_version, package_path
            )
        ) cl ON
            cl.image_id = c.image_id
            AND cl.cve_id = c.cve_id
            AND cl.package_name = c.package_name
            AND cl.package_version = c.package_version
            AND cl.package_path = c.package_path
        WHERE c.transition = 'NEW_OR_REOPENED'
    """)

    # 4a. Bulk INSERT truly NEW findings (no closed history)
    conn.execute(f"""
        INSERT INTO finding_state (
          finding_id, image_id, cve_id, package_name, package_version,
          package_path, severity, cvss_score, in_use, fix_available,
          fix_version, risk_accepted, public_exploit, cisa_kev_known_ransomware,
          first_seen, last_seen,
          state, reason_code, closed_at, reopened_at, reopen_count,
          days_open, is_regression
        )
        SELECT
            nextval('seq_finding_id'),
            image_id, cve_id, package_name, package_version, package_path,
            severity, cvss_score, in_use, fix_available, fix_version,
            risk_accepted, public_exploit, cisa_kev_known_ransomware,
            '{t}'::timestamptz, '{t}'::timestamptz,
            'OPEN', NULL, NULL, NULL, 0, 0, FALSE
        FROM _new_vs_reopened
        WHERE closed_finding_id IS NULL
    """)
    new_count = conn.execute(
        "SELECT COUNT(*) FROM _new_vs_reopened WHERE closed_finding_id IS NULL"
    ).fetchone()[0]

    # 4b. Bulk INSERT REOPENED findings (have closed history)
    conn.execute(f"""
        INSERT INTO finding_state (
          finding_id, image_id, cve_id, package_name, package_version,
          package_path, severity, cvss_score, in_use, fix_available,
          fix_version, risk_accepted, public_exploit, cisa_kev_known_ransomware,
          first_seen, last_seen,
          state, reason_code, closed_at, reopened_at, reopen_count,
          days_open, is_regression
        )
        SELECT
            nextval('seq_finding_id'),
            image_id, cve_id, package_name, package_version, package_path,
            severity, cvss_score, in_use, fix_available, fix_version,
            risk_accepted, public_exploit, cisa_kev_known_ransomware,
            '{t}'::timestamptz, '{t}'::timestamptz,
            'OPEN', NULL, NULL, '{t}'::timestamptz,
            COALESCE(closed_reopen_count, 0) + 1, 0, TRUE
        FROM _new_vs_reopened
        WHERE closed_finding_id IS NOT NULL
    """)
    reopened_count = conn.execute(
        "SELECT COUNT(*) FROM _new_vs_reopened WHERE closed_finding_id IS NOT NULL"
    ).fetchone()[0]

    # 5. GRACE PERIOD — handle disappeared findings
    # 5a. Capture and expire STALE findings past the grace period
    today_date = snapshot_at.date()
    cutoff = today_date - timedelta(days=GRACE_PERIOD_DAYS)
    cutoff_iso = cutoff.replace(hour=23, minute=59, second=59).isoformat()

    conn.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _expired_stale AS
        SELECT * FROM finding_state
        WHERE reason_code = 'STALE'
          AND grace_period_since <= '{cutoff_iso}'::timestamptz
    """)

    expired_count = conn.execute(
        "SELECT COUNT(*) FROM _expired_stale"
    ).fetchone()[0]

    conn.execute(f"""
        UPDATE finding_state SET
          state = 'CLOSED',
          reason_code = 'REMEDIED',
          closed_at = '{t}'::timestamptz,
          days_open = date_diff('day', finding_state.first_seen, '{today}'::date)
        FROM _expired_stale
        WHERE finding_state.finding_id = _expired_stale.finding_id
    """)

    # 5b. Mark newly disappeared as STALE (OPEN findings not in today's data, excluding already-STALE)
    conn.execute("""
        CREATE OR REPLACE TEMPORARY TABLE _disappeared AS
        SELECT fs.*
        FROM finding_state fs
        LEFT JOIN _today_findings tf ON
            tf.image_id = fs.image_id
            AND tf.cve_id = fs.cve_id
            AND tf.package_name = fs.package_name
            AND tf.package_version = fs.package_version
            AND tf.package_path = fs.package_path
        WHERE fs.state = 'OPEN'
          AND fs.reason_code IS DISTINCT FROM 'STALE'
          AND tf.image_id IS NULL
    """)

    disapp_count = conn.execute(
        "SELECT COUNT(*) FROM _disappeared"
    ).fetchone()[0]

    if disapp_count > 0:
        conn.execute(f"""
            UPDATE finding_state SET
              reason_code = 'STALE',
              grace_period_since = '{t}'::timestamptz
            FROM _disappeared d
            WHERE finding_state.finding_id = d.finding_id
        """)

    closed_count = expired_count

    # 6. Write daily OPEN snapshot for historical rollup accuracy
    # This captures the OPEN count per image at this point in time,
    # so rollups rebuilt later can reconstruct historical trends.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_open_snapshot (
            date DATE,
            image_id VARCHAR,
            count_open INTEGER,
            count_open_critical INTEGER,
            count_open_high INTEGER,
            count_open_medium INTEGER,
            count_open_low INTEGER,
            UNIQUE(date, image_id)
        )
    """)

    conn.execute(f"""
        INSERT INTO daily_open_snapshot (
            date, image_id, count_open,
            count_open_critical, count_open_high,
            count_open_medium, count_open_low
        )
        SELECT
            '{today}'::DATE AS date,
            image_id,
            COUNT(*) AS count_open,
            SUM(CASE WHEN severity = 'Critical' THEN 1 ELSE 0 END),
            SUM(CASE WHEN severity = 'High' THEN 1 ELSE 0 END),
            SUM(CASE WHEN severity = 'Medium' THEN 1 ELSE 0 END),
            SUM(CASE WHEN severity = 'Low' THEN 1 ELSE 0 END)
        FROM finding_state
        WHERE state = 'OPEN'
        GROUP BY image_id
        ON CONFLICT (date, image_id) DO UPDATE SET
            count_open = EXCLUDED.count_open,
            count_open_critical = EXCLUDED.count_open_critical,
            count_open_high = EXCLUDED.count_open_high,
            count_open_medium = EXCLUDED.count_open_medium,
            count_open_low = EXCLUDED.count_open_low
    """)

    # 7. Write daily CLOSED transitions snapshot (from _disappeared before cleanup)
    # Captures closed counts per image for images that may have no OPEN findings left.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_closed_snapshot (
            date DATE,
            image_id VARCHAR,
            count_closed INTEGER,
            UNIQUE(date, image_id)
        )
    """)

    conn.execute(f"""
        INSERT INTO daily_closed_snapshot (date, image_id, count_closed)
        SELECT '{today}'::DATE AS date, image_id, COUNT(*) AS count_closed
        FROM (
            SELECT image_id FROM _expired_stale
            UNION ALL
            SELECT image_id FROM _disappeared
        )
        GROUP BY image_id
        ON CONFLICT (date, image_id) DO UPDATE SET
            count_closed = EXCLUDED.count_closed
    """)

    # Cleanup
    conn.execute("DROP TABLE IF EXISTS _today_findings")
    conn.execute("DROP TABLE IF EXISTS _classified")
    conn.execute("DROP TABLE IF EXISTS _new_vs_reopened")
    conn.execute("DROP TABLE IF EXISTS _expired_stale")
    conn.execute("DROP TABLE IF EXISTS _disappeared")

    return {"new": new_count, "reseen": resseen_count, "reopened": reopened_count,
            "closed": closed_count}
