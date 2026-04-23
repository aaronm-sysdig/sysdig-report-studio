"""DuckDB schema for Sysdig Analytics Studio. All DDL lives here.

Matches §4 of the design spec.
"""

# ---------------------------------------------------------------------------
# Shared rollup metric column fragments
# ---------------------------------------------------------------------------
# These 11 count columns appear in every rollup table.  Define once so that
# adding a new metric only requires a single edit here.
_ROLLUP_METRIC_COLS_SQL = """
    count_open_critical INTEGER,
    count_open_high INTEGER,
    count_open_medium INTEGER,
    count_open_low INTEGER,
    count_open INTEGER,
    count_new INTEGER,
    count_fixed_patched INTEGER,
    count_fixed_retired INTEGER,
    count_fixed_accepted INTEGER,
    count_fixed_other INTEGER,
    count_regressed INTEGER"""

# MTTR columns are meaningful at image, workload, and team grain but NOT at
# repository grain (repository-level join would require image-level data).
_ROLLUP_MTTR_COLS_SQL = """
    mttr_sum INTEGER,
    mttr_count INTEGER"""


EXPECTED_TABLES = [
    "image",
    "repository",
    "cve",
    "package",
    "cluster",
    "namespace",
    "workload",
    "team",
    "owner",
    "image_in_repository",
    "workload_runs_image_daily",
    "namespace_in_cluster",
    "workload_in_namespace",
    "workload_owned_by",
    "finding_state",
    "daily_metrics_by_image",
    "daily_metrics_by_workload",
    "daily_metrics_by_team",
    "daily_metrics_by_repository",
    "daily_metrics_by_cluster_severity",
    "ingest_log",
    "snapshot",
]


_DDL = [
    # --- Entities ---
    """
    CREATE TABLE IF NOT EXISTS image (
        image_id VARCHAR PRIMARY KEY,
        os_name VARCHAR,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ,
        current_repository VARCHAR,
        current_tag VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS repository (
        repository VARCHAR PRIMARY KEY,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cve (
        cve_id VARCHAR PRIMARY KEY,
        disclosure_date TIMESTAMPTZ,
        fix_available_date TIMESTAMPTZ,
        cvss_version VARCHAR,
        initial_severity VARCHAR,
        cisa_kev_publish_date TIMESTAMPTZ,
        cisa_kev_due_date TIMESTAMPTZ,
        cisa_kev_known_ransomware BOOLEAN,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS package (
        package_name VARCHAR,
        package_type VARCHAR,
        PRIMARY KEY (package_name, package_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cluster (
        cluster_name VARCHAR PRIMARY KEY,
        distribution VARCHAR,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS namespace (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ,
        PRIMARY KEY (cluster_name, namespace_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ,
        PRIMARY KEY (cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team (
        team_id VARCHAR PRIMARY KEY,
        display_name VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS owner (
        owner_id VARCHAR PRIMARY KEY,
        display_name VARCHAR
    )
    """,
    # --- Relationships ---
    """
    CREATE TABLE IF NOT EXISTS image_in_repository (
        image_id VARCHAR,
        repository VARCHAR,
        tag VARCHAR,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ,
        PRIMARY KEY (image_id, repository, tag)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload_runs_image_daily (
        date DATE,
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        container_name VARCHAR,
        image_id VARCHAR,
        replica_count INTEGER,
        PRIMARY KEY (date, cluster_name, namespace_name, workload_type, workload_name, container_name, image_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS namespace_in_cluster (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        PRIMARY KEY (cluster_name, namespace_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload_in_namespace (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        PRIMARY KEY (cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workload_owned_by (
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,
        team_id VARCHAR,
        owner_id VARCHAR,
        resolved_by_strategy VARCHAR,
        resolved_from VARCHAR,
        PRIMARY KEY (cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    # --- State log ---
    """
    CREATE TABLE IF NOT EXISTS finding_state (
        finding_id BIGINT PRIMARY KEY,
        image_id VARCHAR,
        cve_id VARCHAR,
        package_name VARCHAR,
        package_version VARCHAR,
        package_path VARCHAR,
        severity VARCHAR,
        cvss_score DOUBLE,
        in_use BOOLEAN,
        fix_available BOOLEAN,
        fix_version VARCHAR,
        risk_accepted BOOLEAN,
        public_exploit BOOLEAN,
        first_seen TIMESTAMPTZ,
        last_seen TIMESTAMPTZ,
        state VARCHAR,
        reason_code VARCHAR,
        closed_at TIMESTAMPTZ,
        reopened_at TIMESTAMPTZ,
        reopen_count INTEGER DEFAULT 0,
        days_open INTEGER,
        is_regression BOOLEAN DEFAULT FALSE
    )
    """,
    # Natural-key lookup index on finding_state
    """
    CREATE INDEX IF NOT EXISTS idx_finding_state_natural_key
    ON finding_state (image_id, cve_id, package_name, package_version, package_path)
    """,
    # --- Rollups ---
    f"""
    CREATE TABLE IF NOT EXISTS daily_metrics_by_image (
        date DATE,
        image_id VARCHAR,{_ROLLUP_METRIC_COLS_SQL},{_ROLLUP_MTTR_COLS_SQL},
        PRIMARY KEY (date, image_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS daily_metrics_by_workload (
        date DATE,
        cluster_name VARCHAR,
        namespace_name VARCHAR,
        workload_type VARCHAR,
        workload_name VARCHAR,{_ROLLUP_METRIC_COLS_SQL},{_ROLLUP_MTTR_COLS_SQL},
        replica_count INTEGER,
        PRIMARY KEY (date, cluster_name, namespace_name, workload_type, workload_name)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS daily_metrics_by_team (
        date DATE,
        team_id VARCHAR,{_ROLLUP_METRIC_COLS_SQL},{_ROLLUP_MTTR_COLS_SQL},
        PRIMARY KEY (date, team_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS daily_metrics_by_repository (
        date DATE,
        repository VARCHAR,{_ROLLUP_METRIC_COLS_SQL},
        -- mttr not meaningful at repository grain (image-level join required)
        PRIMARY KEY (date, repository)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_metrics_by_cluster_severity (
        date DATE,
        cluster_name VARCHAR,
        severity VARCHAR,
        count_open INTEGER,
        PRIMARY KEY (date, cluster_name, severity)
    )
    """,
    # --- Operational ---
    """
    CREATE TABLE IF NOT EXISTS snapshot (
        snapshot_id VARCHAR PRIMARY KEY,
        snapshot_at TIMESTAMPTZ,
        source_filename VARCHAR,
        row_count INTEGER,
        ingested_at TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_log (
        snapshot_id VARCHAR,
        stage VARCHAR,
        rows_affected INTEGER,
        duration_ms INTEGER,
        logged_at TIMESTAMPTZ,
        PRIMARY KEY (snapshot_id, stage)
    )
    """,
]


def create_schema(conn) -> None:
    """Create all SAS tables. Idempotent — safe to call on an existing DB."""
    for stmt in _DDL:
        conn.execute(stmt)
