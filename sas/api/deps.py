"""FastAPI dependency: provides a read-only DuckDB connection per request.

Reads SAS_DATA_DIR from environment via sas.ingest.config.get_config().
The ingest pipeline owns the write connection; the API is read-only.
"""

from typing import Generator

import duckdb
from fastapi import HTTPException

from sas.ingest.config import get_config


def get_db() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """FastAPI dependency — yields a read-only DuckDB connection, closes on exit."""
    cfg = get_config()
    path = cfg.duckdb_path
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found at {path}. Run ingest first.",
        )
    conn = duckdb.connect(str(path), read_only=True)
    try:
        yield conn
    finally:
        conn.close()
