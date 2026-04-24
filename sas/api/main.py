"""FastAPI application entry point for Sysdig Analytics Studio API."""

import duckdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sas.api.routes import query, widgets, entities, findings as findings_router
from sas.ingest.config import get_config

app = FastAPI(
    title="Sysdig Analytics Studio API",
    description="Query engine for vulnerability trend analytics.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tightened in production via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router,   prefix="/api")
app.include_router(widgets.router, prefix="/api")
app.include_router(entities.router, prefix="/api")
app.include_router(findings_router.router, prefix="/api", tags=["findings"])


@app.get("/healthz", tags=["ops"])
def healthz():
    """Health check — verifies the DuckDB file exists and is readable."""
    cfg = get_config()
    path = cfg.duckdb_path
    if not path.exists():
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": f"Database not found at {path}"},
        )
    try:
        conn = duckdb.connect(str(path), read_only=True)
        conn.execute("SELECT 1").fetchone()
        conn.close()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": str(exc)},
        )
    return {"status": "ok"}
