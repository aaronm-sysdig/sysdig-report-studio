"""
Database layer for Sysdig Report Studio.

Handles all the SQLite bits - storing report templates, schedules, and generated PDFs.
Templates are the main thing here; schedules are just an optional extra per template.
"""
import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional

# Default data directory - can be overridden for K8s PV mount
DATA_DIR = os.environ.get("SYSDIG_REPORT_DATA_DIR", "./data")
DB_PATH = os.path.join(DATA_DIR, "sysdig_reports.db")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory for dict-like access."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database schema."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    # Report templates - the main entity containing report config and optional schedule
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            config_yaml TEXT NOT NULL,
            -- Schedule settings (optional - NULL means no schedule)
            schedule_enabled INTEGER DEFAULT 0,
            schedule_frequency TEXT,
            schedule_hour INTEGER,
            schedule_day_of_week INTEGER,
            schedule_day_of_month INTEGER,
            schedule_timezone TEXT DEFAULT 'UTC',
            schedule_retain_count INTEGER DEFAULT 10,
            schedule_last_run TEXT,
            schedule_next_run TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Generated reports - PDFs created from templates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            status TEXT NOT NULL,
            error_message TEXT,
            generated_at TEXT NOT NULL,
            FOREIGN KEY (template_id) REFERENCES report_templates(id) ON DELETE CASCADE
        )
    """)

    # Migrate old schema if needed
    _migrate_old_schema(cursor)

    conn.commit()
    conn.close()


def _migrate_old_schema(cursor):
    """Migrate from old schedule-centric schema if it exists."""
    # Check if old 'schedules' table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schedules'")
    if cursor.fetchone():
        # Migrate data from schedules to report_templates
        cursor.execute("SELECT * FROM schedules")
        old_schedules = cursor.fetchall()

        for sched in old_schedules:
            cursor.execute("""
                INSERT INTO report_templates
                (name, config_yaml, schedule_enabled, schedule_frequency, schedule_hour,
                 schedule_day_of_week, schedule_day_of_month, schedule_timezone,
                 schedule_retain_count, schedule_last_run, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sched['name'], sched['config_yaml'], sched['enabled'],
                sched['frequency'], sched['hour'], sched['day_of_week'],
                sched['day_of_month'], sched['timezone'], sched['retain_count'],
                sched['last_run'], sched['created_at'], sched['updated_at']
            ))
            new_template_id = cursor.lastrowid

            # Migrate associated reports
            cursor.execute("SELECT * FROM reports WHERE schedule_id = ?", (sched['id'],))
            old_reports = cursor.fetchall()
            for report in old_reports:
                cursor.execute("""
                    INSERT INTO generated_reports
                    (template_id, filename, file_path, file_size, status, error_message, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    new_template_id, report['filename'], report['file_path'],
                    report['file_size'], report['status'], report['error_message'],
                    report['generated_at']
                ))

        # Drop old tables
        cursor.execute("DROP TABLE IF EXISTS reports")
        cursor.execute("DROP TABLE IF EXISTS schedules")


# =============================================================================
# REPORT TEMPLATE CRUD OPERATIONS
# =============================================================================

def create_template(name: str, config_yaml: str) -> int:
    """Create a new report template. Returns the template ID."""
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    cursor.execute("""
        INSERT INTO report_templates (name, config_yaml, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """, (name, config_yaml, now, now))

    template_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return template_id


def get_template(template_id: int) -> Optional[dict]:
    """Get a single template by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM report_templates WHERE id = ?", (template_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def get_all_templates() -> list[dict]:
    """Get all report templates."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM report_templates ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def update_template(template_id: int, **kwargs) -> bool:
    """Update a template. Pass any column names as kwargs."""
    if not kwargs:
        return False

    conn = get_connection()
    cursor = conn.cursor()

    kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()

    set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [template_id]

    cursor.execute(f"UPDATE report_templates SET {set_clause} WHERE id = ?", values)

    success = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return success


def delete_template(template_id: int) -> bool:
    """Delete a template and its associated generated reports."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get report files to delete
    cursor.execute("SELECT file_path FROM generated_reports WHERE template_id = ?", (template_id,))
    reports = cursor.fetchall()

    # Delete report files
    for report in reports:
        if os.path.exists(report["file_path"]):
            try:
                os.remove(report["file_path"])
            except OSError:
                pass

    # Delete from database (CASCADE will handle generated_reports table)
    cursor.execute("DELETE FROM report_templates WHERE id = ?", (template_id,))

    success = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return success


def update_template_schedule(
    template_id: int,
    enabled: bool,
    frequency: Optional[str] = None,
    hour: Optional[int] = None,
    day_of_week: Optional[int] = None,
    day_of_month: Optional[int] = None,
    timezone: str = "UTC",
    retain_count: int = 10
) -> bool:
    """Update or set the schedule for a template."""
    return update_template(
        template_id,
        schedule_enabled=1 if enabled else 0,
        schedule_frequency=frequency,
        schedule_hour=hour,
        schedule_day_of_week=day_of_week,
        schedule_day_of_month=day_of_month,
        schedule_timezone=timezone,
        schedule_retain_count=retain_count
    )


def disable_template_schedule(template_id: int) -> bool:
    """Disable the schedule for a template."""
    return update_template(template_id, schedule_enabled=0)


def get_scheduled_templates() -> list[dict]:
    """Get all templates with enabled schedules."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM report_templates WHERE schedule_enabled = 1")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# =============================================================================
# GENERATED REPORT CRUD OPERATIONS
# =============================================================================

def create_generated_report(
    template_id: int,
    filename: str,
    file_path: str,
    status: str = "success",
    file_size: Optional[int] = None,
    error_message: Optional[str] = None
) -> int:
    """Create a generated report record. Returns the report ID."""
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    cursor.execute("""
        INSERT INTO generated_reports
        (template_id, filename, file_path, file_size, status, error_message, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (template_id, filename, file_path, file_size, status, error_message, now))

    report_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Enforce retention policy
    enforce_retention(template_id)

    return report_id


def get_reports_for_template(template_id: int) -> list[dict]:
    """Get all generated reports for a template, newest first."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM generated_reports
        WHERE template_id = ?
        ORDER BY generated_at DESC
    """, (template_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_latest_report(template_id: int) -> Optional[dict]:
    """Get the most recent generated report for a template."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM generated_reports
        WHERE template_id = ?
        ORDER BY generated_at DESC
        LIMIT 1
    """, (template_id,))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def delete_generated_report(report_id: int) -> bool:
    """Delete a generated report and its file."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get file path
    cursor.execute("SELECT file_path FROM generated_reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()

    if row and os.path.exists(row["file_path"]):
        try:
            os.remove(row["file_path"])
        except OSError:
            pass

    cursor.execute("DELETE FROM generated_reports WHERE id = ?", (report_id,))

    success = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return success


def enforce_retention(template_id: int):
    """Delete old reports beyond the retention count for a template."""
    template = get_template(template_id)
    if not template:
        return

    retain_count = template.get("schedule_retain_count") or 10

    conn = get_connection()
    cursor = conn.cursor()

    # Get reports beyond retention limit
    cursor.execute("""
        SELECT id, file_path FROM generated_reports
        WHERE template_id = ?
        ORDER BY generated_at DESC
        LIMIT -1 OFFSET ?
    """, (template_id, retain_count))

    old_reports = cursor.fetchall()

    for report in old_reports:
        if os.path.exists(report["file_path"]):
            try:
                os.remove(report["file_path"])
            except OSError:
                pass
        cursor.execute("DELETE FROM generated_reports WHERE id = ?", (report["id"],))

    conn.commit()
    conn.close()


def get_all_generated_reports() -> list[dict]:
    """Get all generated reports across all templates."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT gr.*, rt.name as template_name
        FROM generated_reports gr
        JOIN report_templates rt ON gr.template_id = rt.id
        ORDER BY gr.generated_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_database_path() -> str:
    """Return the path to the database file."""
    return DB_PATH


def get_reports_directory() -> str:
    """Return the path to the reports directory."""
    return REPORTS_DIR


# Initialize database on module import
init_database()
