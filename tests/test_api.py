"""API tests — health check and CORS middleware verification."""

import duckdb
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from sas.api.main import app

client = TestClient(app)


def test_healthz_returns_ok_when_db_exists(tmp_path, monkeypatch):
    """GET /healthz returns 200 + {"status": "ok"} when the DB file exists."""
    db_path = tmp_path / "sas.duckdb"
    # Create a minimal DuckDB file so read-only connect works
    conn = duckdb.connect(str(db_path))
    conn.close()

    import sas.api.main as main_module

    class _FakeConfig:
        duckdb_path = db_path

    monkeypatch.setattr(main_module, "get_config", lambda: _FakeConfig())

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_returns_503_when_db_missing(tmp_path, monkeypatch):
    """GET /healthz returns 503 when the DB file does not exist."""
    import sas.api.main as main_module

    class _FakeConfig:
        duckdb_path = tmp_path / "does_not_exist.duckdb"

    monkeypatch.setattr(main_module, "get_config", lambda: _FakeConfig())

    response = client.get("/healthz")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert "error" in data


def test_app_has_cors_middleware():
    """CORS middleware is installed on the FastAPI app."""
    middleware_classes = [m.cls for m in app.user_middleware]
    assert CORSMiddleware in middleware_classes
