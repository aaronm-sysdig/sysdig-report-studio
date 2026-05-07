"""DuckDB-native CSV loader — streams, no Pandas."""
from __future__ import annotations

import re
from pathlib import Path

import duckdb


_BOOL_COLS = {
    "public_exploit",
    "package_in_use",
    "risk_accepted",
    "cisa_kev_known_ransomware",
    "fix_available",
}

_DATE_COLS = {
    "disclosure_date",
    "fix_available_date",
    "cisa_kev_publish_date",
    "cisa_kev_due_date",
}


def _normalize(col: str) -> str:
    s = col.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_csv_to_temp(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> tuple[str, int]:
    """Load a Sysdig CSV into a DuckDB temp table. Returns (table_name, row_count).

    Uses DuckDB's native CSV reader which streams — no full in-memory load.
    Columns are normalized (lowercase, underscores) and types coerced.
    """
    temp_table = "_ingest_staging"
    csv_path = csv_path.resolve()  # DuckDB read_csv_auto needs absolute path

    # Discover raw column names
    sample = conn.execute(f"SELECT * FROM read_csv_auto('{csv_path}') LIMIT 0")
    raw_columns = [desc[0] for desc in sample.description]
    norm_columns = [_normalize(c) for c in raw_columns]

    # Build alias clause: "Raw Column Name" AS "normalized_name"
    aliases = ", ".join(f'"{rc}" AS "{nc}"' for rc, nc in zip(raw_columns, norm_columns))

    # Build cast expressions for each normalized column
    cast_cols = []
    for nc in norm_columns:
        if nc in _BOOL_COLS:
            cast_cols.append(
                f'CASE WHEN "{nc}" IS NULL THEN FALSE '
                f'ELSE "{nc}"::BOOLEAN END AS "{nc}"'
            )
        elif nc in _DATE_COLS:
            cast_cols.append(f'TRY_CAST("{nc}" AS TIMESTAMPTZ) AS "{nc}"')
        else:
            cast_cols.append(f'"{nc}"')

    select_cols = ", ".join(cast_cols)

    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
    conn.execute(f"""
        CREATE TEMPORARY TABLE {temp_table} AS
        SELECT {select_cols}
        FROM (SELECT {aliases} FROM read_csv_auto('{csv_path}'))
    """)

    count = conn.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]

    return temp_table, count
