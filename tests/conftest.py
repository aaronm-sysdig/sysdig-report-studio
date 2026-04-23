import duckdb
import pytest
from pathlib import Path


@pytest.fixture
def db():
    """Fresh in-memory DuckDB for each test."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def fixtures_dir():
    """Path to tests/fixtures/ directory."""
    return Path(__file__).parent / "fixtures"
