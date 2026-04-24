"""Tests for the three API routes: /query, /widgets/catalog, /entities/{lens}.

Uses FastAPI TestClient with a real in-memory DuckDB seeded with minimal data
and dependency override via app.dependency_overrides.
"""

import duckdb
import pytest
from fastapi.testclient import TestClient

from sas.api.main import app
from sas.api.deps import get_db
from sas.ingest.schema import create_schema
from sas.query.primitives import Query, TimeWindow, Filter


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_db():
    """TestClient backed by a seeded in-memory DuckDB, with get_db overridden."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)

    # Seed minimal entity rows
    conn.execute(
        "INSERT INTO image VALUES ('sha256:abc', 'linux', NOW(), NOW(), 'myrepo', 'v1')"
    )
    conn.execute(
        "INSERT INTO cve VALUES ('CVE-2025-0001', NULL, NULL, 'v3', 'Critical', NULL, NULL, FALSE, NOW(), NOW())"
    )
    # Seed one open finding so count_open has data
    conn.execute("""
        INSERT INTO finding_state VALUES
        (1, 'sha256:abc', 'CVE-2025-0001', 'openssl', '1.0', '/usr/lib', 'Critical', 9.8,
         TRUE, TRUE, '1.1', FALSE, TRUE, FALSE,
         '2026-04-20'::TIMESTAMP, '2026-04-23'::TIMESTAMP, 'OPEN', 'NEW', NULL, NULL, 0, 3, FALSE)
    """)

    def _override():
        try:
            yield conn
        finally:
            pass  # don't close — fixture handles lifecycle

    app.dependency_overrides[get_db] = _override
    yield TestClient(app), conn
    app.dependency_overrides.clear()
    conn.close()


# ---------------------------------------------------------------------------
# 1. POST /api/query — happy path
# ---------------------------------------------------------------------------

def test_query_endpoint_returns_result(client_with_db):
    """POST /api/query with a valid Query returns 200 with series + exec_time_ms."""
    client, _ = client_with_db
    payload = {
        "lens": "Image",
        "traversal": [],
        "time": {"mode": "last_n_snapshots", "n": 7, "granularity": "day"},
        "measure": "count_open",
        "filters": [],
        "group_by": [],
        "order_by": None,
        "limit": None,
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "series" in data
    assert "exec_time_ms" in data


# ---------------------------------------------------------------------------
# 2. POST /api/query — invalid body → 422
# ---------------------------------------------------------------------------

def test_query_endpoint_validates_body(client_with_db):
    """POST /api/query with invalid JSON body returns 422."""
    client, _ = client_with_db
    # Missing required fields (lens, time, measure)
    response = client.post("/api/query", json={"not_a_query": True})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. GET /api/widgets/catalog — returns 10 entries
# ---------------------------------------------------------------------------

def test_widgets_catalog_returns_10_definitions(client_with_db):
    """GET /api/widgets/catalog returns 200 and exactly 10 widget definitions."""
    client, _ = client_with_db
    response = client.get("/api/widgets/catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert len(catalog) == 10
    for widget in catalog:
        assert "id" in widget
        assert "title" in widget
        assert "widget_type" in widget
        assert "query" in widget


# ---------------------------------------------------------------------------
# 4. Each catalog widget query dict is constructable as a Query dataclass
# ---------------------------------------------------------------------------

def test_widget_catalog_widgets_are_valid_queries(client_with_db):
    """Each widget in the catalog has a query dict that constructs a valid Query."""
    client, _ = client_with_db
    response = client.get("/api/widgets/catalog")
    assert response.status_code == 200
    catalog = response.json()

    for widget in catalog:
        q_dict = widget["query"]
        tw_dict = q_dict["time"]
        tw = TimeWindow(
            mode=tw_dict["mode"],
            n=tw_dict.get("n"),
            start=tw_dict.get("start"),
            end=tw_dict.get("end"),
            granularity=tw_dict.get("granularity", "day"),
        )
        filters = [
            Filter(field=f["field"], operator=f["operator"], value=f["value"])
            for f in q_dict.get("filters", [])
        ]
        q = Query(
            lens=q_dict["lens"],
            traversal=q_dict.get("traversal", []),
            time=tw,
            measure=q_dict["measure"],
            filters=filters,
            group_by=q_dict.get("group_by", []),
        )
        assert q.lens is not None
        assert q.measure is not None


# ---------------------------------------------------------------------------
# 5. GET /api/entities/{lens} — returns rows for seeded data
# ---------------------------------------------------------------------------

def test_entities_endpoint_returns_rows(client_with_db):
    """GET /api/entities/Image returns 200 + list with at least 1 entry."""
    client, _ = client_with_db
    response = client.get("/api/entities/Image")
    assert response.status_code == 200
    rows = response.json()
    assert isinstance(rows, list)
    assert len(rows) >= 1
    # Each row should have 'id' and 'label' keys
    assert "id" in rows[0]
    assert "label" in rows[0]


# ---------------------------------------------------------------------------
# 6. GET /api/entities/{lens} — unknown lens → 422
# ---------------------------------------------------------------------------

def test_entities_endpoint_unknown_lens_returns_422(client_with_db):
    """GET /api/entities/NotALens returns 422."""
    client, _ = client_with_db
    response = client.get("/api/entities/NotALens")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. GET /api/findings — paginated rows
# ---------------------------------------------------------------------------

def test_list_findings_returns_paginated(client_with_db):
    client, conn = client_with_db
    # The fixture seeds at least one finding via the existing setup
    res = client.get("/api/findings?limit=10")
    assert res.status_code == 200
    body = res.json()
    assert "rows" in body
    assert "total" in body
    assert isinstance(body["rows"], list)


def test_list_findings_invalid_severity_returns_422(client_with_db):
    client, _ = client_with_db
    res = client.get("/api/findings?severity=Bogus")
    assert res.status_code == 422
