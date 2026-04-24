# Sysdig Analytics Studio (SAS)

Query engine and API for vulnerability trend analytics over time, backed by DuckDB.

## Ingest

```bash
python -m sas.ingest ~/downloads/phoenix-vuln-findings-2026_04_23.csv
```

Writes to `~/sysdig-vuln-data/sas.duckdb` by default. Override with `SAS_DATA_DIR`.

## API

```bash
# Start the server
python -m sas.api.run
# → http://localhost:8000/docs   (OpenAPI)
# → http://localhost:8000/healthz

# Quick health check
curl http://localhost:8000/healthz
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/query` | Execute a Query, returns QueryResult |
| `GET`  | `/api/widgets/catalog` | 10 starter widget definitions |
| `GET`  | `/api/entities/{lens}` | Entity values for UI pickers |

## Tests

```bash
.venv/bin/pytest -v
```
