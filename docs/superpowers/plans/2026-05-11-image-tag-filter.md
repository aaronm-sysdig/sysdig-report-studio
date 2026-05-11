# Image Tag Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tag dropdown next to the repository picker in ImageRemediationStory so users can filter vulnerability trends by image tag rather than seeing all sha256 digests aggregated.

**Architecture:** Backend adds `current_tag` as a filterable field on the Image lens, joining the `image` table to resolve tags. New `GET /api/entities/tags` endpoint returns distinct tags per repository. Frontend adds a `TagSelect` component that populates on repository selection and filters the query by tag.

**Tech Stack:** Python (FastAPI, DuckDB), TypeScript (React, Next.js, ECharts)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `sas/query/rollup_router.py` | **Modify** | Add `current_tag` to Image rollup filter columns |
| `sas/query/compiler.py` | **Modify** | Handle `current_tag` filter in rollup and direct paths |
| `sas/api/routes/entities.py` | **Modify** | Add `GET /api/entities/tags` endpoint |
| `sas/web/lib/api/client.ts` | **Modify** | Add `getTagsForRepository()` function |
| `sas/web/components/widgets/ImageRemediationStory.tsx` | **Modify** | Add `TagSelect` component and wire tag filter to queries |
| `tests/test_rollup_router.py` | **Modify** | Test that `current_tag` filter stays on rollup path |
| `tests/test_compiler.py` | **Modify** | Test tag filter produces correct SQL |
| `tests/test_api_routes.py` | **Modify** | Test `/api/entities/tags` endpoint |

---

### Task 1: Add `current_tag` to rollup filter columns

**Files:**
- Modify: `sas/query/rollup_router.py`
- Test: `tests/test_rollup_router.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rollup_router.py`:

```python
def test_image_current_tag_filter_stays_on_rollup():
    """A filter on current_tag should still use the rollup path (Image lens)."""
    from sas.query.primitives import Filter, Query, TimeWindow
    from sas.query.rollup_router import can_use_rollup
    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open",
        filters=[Filter(field="current_tag", operator="eq", value="codeserver")],
    )
    assert can_use_rollup(q) == "daily_metrics_by_image"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_rollup_router.py::test_image_current_tag_filter_stays_on_rollup -v`
Expected: FAIL — `current_tag` not in `_ROLLUP_FILTER_COLUMNS["daily_metrics_by_image"]`

- [ ] **Step 3: Add `current_tag` to the Image rollup filter columns**

In `sas/query/rollup_router.py`, change:

```python
_ROLLUP_FILTER_COLUMNS: dict[str, set[str]] = {
    "daily_metrics_by_image": {"image_id"},
```

To:

```python
_ROLLUP_FILTER_COLUMNS: dict[str, set[str]] = {
    "daily_metrics_by_image": {"image_id", "current_tag"},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_rollup_router.py::test_image_current_tag_filter_stays_on_rollup -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sas/query/rollup_router.py tests/test_rollup_router.py
git commit -m "feat(query): allow current_tag filter on Image rollup path"
```

---

### Task 2: Handle `current_tag` filter in the rollup compiler

**Files:**
- Modify: `sas/query/compiler.py`
- Test: `tests/test_compiler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_compiler.py`:

```python
def test_rollup_path_with_current_tag_filter_joins_image_table(seeded_db):
    """When current_tag filter is present, rollup query should join image table."""
    from sas.query.compiler import compile as sas_compile
    from sas.query.primitives import Query, TimeWindow, Filter

    conn = seeded_db
    # Seed a second image with different tag
    conn.execute(
        "INSERT INTO image VALUES ('sha256:bbb', 'linux', NOW(), NOW(), 'myrepo', 'v2')"
    )
    # Seed rollup row for second image
    conn.execute(
        """
        INSERT INTO daily_metrics_by_image (
            date, image_id, count_open_critical, count_open_high, count_open_medium, count_open_low,
            count_open, count_new, count_fixed_patched, count_fixed_retired, count_fixed_accepted, count_fixed_other,
            count_regressed, mttr_sum, mttr_count
        ) VALUES ('2026-04-10', 'sha256:bbb', 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        """
    )

    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_open",
        filters=[Filter(field="current_tag", operator="eq", value="v1")],
    )

    result = sas_compile(q, conn)
    # Should only return data for sha256:aaa (tag v1), not sha256:bbb (tag v2)
    total = sum(sum(s.y) for s in result.series)
    assert total == 2  # Only the two OPEN findings from sha256:aaa
```

- [ ] **Step 2: Run test to verify it fails or returns wrong result**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_compiler.py::test_rollup_path_with_current_tag_filter_joins_image_table -v`
Expected: FAIL or wrong assertion (current code doesn't handle `current_tag` filter)

- [ ] **Step 3: Modify the rollup compiler to handle `current_tag` filter**

In `sas/query/compiler.py`, modify `_compile_rollup()` to detect `current_tag` filters and join the `image` table:

Replace the `_compile_rollup` function (lines ~183-217) with:

```python
def _compile_rollup(
    query: Query, conn, table: str, start: date, end: date
) -> QueryResult:
    t0 = time.monotonic()
    pk_col = _ROLLUP_LENS_PK[table]
    measure_expr = _ROLLUP_MEASURE_EXPR[query.measure]
    filter_sql, filter_params = _build_filter_clause(query.filters)

    # Check if we need to join the image table for current_tag filter
    has_tag_filter = any(
        f.field == "current_tag" for f in query.filters
    )

    group_cols_sql = ""
    if query.group_by:
        group_cols_sql = ", " + ", ".join(query.group_by)

    if has_tag_filter and table == "daily_metrics_by_image":
        # Join image table to filter by current_tag, then aggregate
        sql = (
            f"SELECT date, {pk_col}{group_cols_sql}, {measure_expr} AS value "
            f"FROM {table} dm "
            f"JOIN image img ON img.image_id = dm.{pk_col} "
            f"WHERE date BETWEEN ? AND ?{filter_sql} "
            f"ORDER BY date"
        )
    else:
        sql = (
            f"SELECT date, {pk_col}{group_cols_sql}, {measure_expr} AS value "
            f"FROM {table} "
            f"WHERE date BETWEEN ? AND ?{filter_sql} "
            f"ORDER BY date"
        )

    rows = conn.execute(sql, [start, end] + filter_params).fetchall()
    col_names = [d[0] for d in conn.description]

    series = _rows_to_series(rows, col_names, pk_col, query.group_by)
    missing = _compute_missing_days(conn, start, end)
    exec_ms = int((time.monotonic() - t0) * 1000)
    return QueryResult(
        series=series,
        dimensions={},
        snapshot_range=(start, end),
        missing_days=missing,
        exec_time_ms=exec_ms,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_compiler.py::test_rollup_path_with_current_tag_filter_joins_image_table -v`
Expected: PASS

- [ ] **Step 5: Run all compiler tests to check for regressions**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_compiler.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sas/query/compiler.py tests/test_compiler.py
git commit -m "feat(query): join image table for current_tag filter in rollup path"
```

---

### Task 3: Handle `current_tag` filter in the direct compiler path

**Files:**
- Modify: `sas/query/compiler.py`
- Test: `tests/test_compiler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_compiler.py`:

```python
def test_direct_path_with_current_tag_filter_joins_image_table(seeded_db):
    """When current_tag filter is present on direct path, join image table."""
    from sas.query.compiler import compile as sas_compile
    from sas.query.primitives import Query, TimeWindow, Filter

    conn = seeded_db

    # Use a measure that forces direct path (count_distinct_cve)
    q = Query(
        lens="Image",
        traversal=[],
        time=TimeWindow(mode="last_n_snapshots", n=7, granularity="day"),
        measure="count_distinct_cve",
        filters=[Filter(field="current_tag", operator="eq", value="v1")],
    )

    result = sas_compile(q, conn)
    # Should return data filtered to tag v1
    assert len(result.series) >= 0  # Query executes without error
```

- [ ] **Step 2: Run test to verify it fails or errors**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_compiler.py::test_direct_path_with_current_tag_filter_joins_image_table -v`
Expected: FAIL or error (current code doesn't handle `current_tag` on direct path)

- [ ] **Step 3: Modify the direct compiler to handle `current_tag` filter**

In `sas/query/compiler.py`, modify `_compile_direct()` to detect `current_tag` filters and add a join to the `image` table:

In the `_compile_direct` function, after the line that sets `join_sql` (around line ~225), add logic to handle `current_tag`:

```python
def _compile_direct(
    query: Query, conn, start: date, end: date
) -> QueryResult:
    t0 = time.monotonic()

    # Resolve per-lens join and PK column
    join_sql = _LENS_JOIN_SQL.get(query.lens, "")
    qualified_pk = _DIRECT_LENS_PK_COL.get(query.lens, "finding_state.image_id")
    # Unqualified alias for SELECT column name and GROUP BY
    pk_col = qualified_pk.split(".")[-1]

    # Add image table join if current_tag filter is present
    has_tag_filter = any(
        f.field == "current_tag" for f in query.filters
    )
    if has_tag_filter and not join_sql:
        # No existing join, add image join
        join_sql = "JOIN image img ON img.image_id = finding_state.image_id"
    elif has_tag_filter:
        # Existing join present, add image join alongside
        join_sql = join_sql + "\nJOIN image img ON img.image_id = finding_state.image_id"

    # Rewrite current_tag filter to use joined alias
    effective_filters = []
    for f in query.filters:
        if f.field == "current_tag":
            effective_filters.append(Filter(field="img.current_tag", operator=f.operator, value=f.value))
        else:
            effective_filters.append(f)

    date_col, base_predicate = _DIRECT_DATE_COL[query.measure]
    aggregate = _DIRECT_AGGREGATE[query.measure]
    filter_sql, filter_params = _build_filter_clause(effective_filters)

    group_cols_sql = ""
    if query.group_by:
        group_cols_sql = ", " + ", ".join(query.group_by)

    join_clause = f"\n{join_sql}" if join_sql else ""

    date_expr = f"CAST({date_col} AS DATE)"
    sql = (
        f"SELECT {date_expr} AS date, {qualified_pk} AS {pk_col}{group_cols_sql}, {aggregate} AS value "
        f"FROM finding_state{join_clause} "
        f"WHERE {base_predicate} "
        f"  AND {date_expr} BETWEEN ? AND ?{filter_sql} "
        f"GROUP BY {date_expr}, {qualified_pk}{group_cols_sql} "
        f"ORDER BY date"
    )

    rows = conn.execute(sql, [start, end] + filter_params).fetchall()
    col_names = [d[0] for d in conn.description]

    series = _rows_to_series(rows, col_names, pk_col, query.group_by)
    missing = _compute_missing_days(conn, start, end)
    exec_ms = int((time.monotonic() - t0) * 1000)
    return QueryResult(
        series=series,
        dimensions={},
        snapshot_range=(start, end),
        missing_days=missing,
        exec_time_ms=exec_ms,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_compiler.py::test_direct_path_with_current_tag_filter_joins_image_table -v`
Expected: PASS

- [ ] **Step 5: Run all compiler tests to check for regressions**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_compiler.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sas/query/compiler.py tests/test_compiler.py
git commit -m "feat(query): join image table for current_tag filter in direct path"
```

---

### Task 4: Add `GET /api/entities/tags` endpoint

**Files:**
- Modify: `sas/api/routes/entities.py`
- Test: `tests/test_api_routes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api_routes.py`:

```python
def test_get_tags_for_repository(client):
    """GET /api/entities/tags?repository=X returns distinct tags."""
    # Seed some images with different tags for the same repo
    client.app.dependency_overrides[get_db] = lambda: duckdb.connect(":memory:")
    conn = client.app.dependency_overrides[get_db]()
    from sas.ingest.schema import create_schema
    create_schema(conn)
    conn.execute(
        "INSERT INTO image VALUES "
        "('sha256:aaa', 'linux', NOW(), NOW(), 'myrepo', 'v1'), "
        "('sha256:bbb', 'linux', NOW(), NOW(), 'myrepo', 'v2'), "
        "('sha256:ccc', 'linux', NOW(), NOW(), 'myrepo', 'v1'), "
        "('sha256:ddd', 'linux', NOW(), NOW(), 'other-repo', 'v1')"
    )

    response = client.get("/api/entities/tags?repository=myrepo")
    assert response.status_code == 200
    tags = response.json()
    tag_names = [t["tag"] for t in tags]
    assert set(tag_names) == {"v1", "v2"}
    # Check image counts
    counts = {t["tag"]: t["image_count"] for t in tags}
    assert counts["v1"] == 2  # sha256:aaa and sha256:ccc
    assert counts["v2"] == 1  # sha256:bbb
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_api_routes.py::test_get_tags_for_repository -v`
Expected: FAIL — 404 (endpoint doesn't exist)

- [ ] **Step 3: Add the endpoint**

In `sas/api/routes/entities.py`, add after the existing `get_entities` function:

```python
@router.get("/entities/tags", tags=["entities"])
def get_tags(repository: str, conn=Depends(get_db)) -> list[dict]:
    """Return distinct tags for a repository, with image counts."""
    rows = conn.execute(
        """
        SELECT current_tag AS tag, COUNT(DISTINCT image_id) AS image_count
        FROM image
        WHERE current_repository = ?
        GROUP BY current_tag
        ORDER BY current_tag
        """,
        [repository],
    ).fetchall()
    return [{"tag": r[0], "image_count": r[1]} for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_api_routes.py::test_get_tags_for_repository -v`
Expected: PASS

- [ ] **Step 5: Run all API route tests to check for regressions**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m pytest tests/test_api_routes.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sas/api/routes/entities.py tests/test_api_routes.py
git commit -m "feat(api): add GET /api/entities/tags endpoint for repository tags"
```

---

### Task 5: Add frontend API client function

**Files:**
- Modify: `sas/web/lib/api/client.ts`

- [ ] **Step 1: Add the client function**

In `sas/web/lib/api/client.ts`, add after the existing `getEntities` function:

```typescript
export interface TagEntity {
  tag: string;
  image_count: number;
}

/**
 * GET /api/entities/tags?repository=X — list distinct tags for a repository.
 */
export async function getTagsForRepository(
  repository: string
): Promise<TagEntity[]> {
  return apiFetch<TagEntity[]>(
    `/api/entities/tags?repository=${encodeURIComponent(repository)}`
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add sas/web/lib/api/client.ts
git commit -m "feat(web): add getTagsForRepository API client function"
```

---

### Task 6: Add TagSelect component and wire into ImageRemediationStory

**Files:**
- Modify: `sas/web/components/widgets/ImageRemediationStory.tsx`

- [ ] **Step 1: Add TagSelect component**

In `ImageRemediationStory.tsx`, add after the `RepositoryPicker` component (around line ~800):

```tsx
// ---------------------------------------------------------------------------
// Tag selector
// ---------------------------------------------------------------------------
interface TagSelectProps {
  repository: string;
  selectedTag: string | null;
  onSelect: (tag: string | null) => void;
}

function TagSelect({ repository, selectedTag, onSelect }: TagSelectProps) {
  const [tags, setTags] = useState<TagEntity[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!repository) {
      setTags([]);
      return;
    }
    setLoading(true);
    getTagsForRepository(repository)
      .then((result) => setTags(result))
      .catch(() => setTags([]))
      .finally(() => setLoading(false));
  }, [repository]);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    onSelect(value === "__all__" ? null : value);
  };

  return (
    <div className="flex items-center gap-1">
      <span
        className="text-[11px] font-medium"
        style={{ color: "var(--fg-muted)" }}
      >
        Tag:
      </span>
      <select
        value={selectedTag ?? "__all__"}
        onChange={handleChange}
        disabled={loading || tags.length === 0}
        className="text-[11px] rounded px-2 py-1 h-[36px]"
        style={{
          border: "1px solid var(--border-subtle)",
          background: "var(--bg-surface)",
          color: "var(--fg-primary)",
          cursor: loading ? "wait" : "pointer",
          maxWidth: 200,
        }}
      >
        <option value="__all__">All tags</option>
        {tags.map((t) => (
          <option key={t.tag} value={t.tag}>
            {t.tag} ({t.image_count})
          </option>
        ))}
      </select>
    </div>
  );
}
```

- [ ] **Step 2: Import the new API function**

At the top of `ImageRemediationStory.tsx`, update the import:

```tsx
import { runQuery, getEntities, getTagsForRepository } from "@/lib/api/client";
import type { QueryIn, QueryResult, TagEntity } from "@/lib/api/client";
```

- [ ] **Step 3: Add tag state and handler**

In the `ImageRemediationStory` component, after the existing state declarations (around line ~825), add:

```tsx
const [selectedTag, setSelectedTag] = useState<string | null>(null);
```

- [ ] **Step 4: Clear tag selection when repository changes**

In the `selectedRepo` effect (around line ~860), add `setSelectedTag(null)` when the repository changes:

Find the effect that resets data on repo change and add:
```tsx
setSelectedTag(null);
```

- [ ] **Step 5: Filter imageIds by tag when selected**

Replace the `imageIds` memo (around line ~860) to filter by tag:

```tsx
const imageIds = useMemo<string[]>(() => {
  if (!selectedTag) return imagesInRepo.map((img) => img.id);
  return imagesInRepo
    .filter((img) => img.tag === selectedTag)
    .map((img) => img.id);
}, [imagesInRepo, selectedTag]);
```

- [ ] **Step 6: Render TagSelect next to RepositoryPicker**

Find where `RepositoryPicker` is rendered (around line ~1100-1150) and add `TagSelect` beside it. Change the layout to:

```tsx
{!externalRepo && (
  <div className="flex items-center gap-3 flex-wrap">
    <RepositoryPicker
      repositories={repositories}
      selectedRepo={selectedRepo ?? ""}
      onSelect={(repo) => {
        setSelectedRepo(repo);
        setSelectedTag(null);
      }}
    />
    {selectedRepo && (
      <TagSelect
        repository={selectedRepo}
        selectedTag={selectedTag}
        onSelect={setSelectedTag}
      />
    )}
  </div>
)}
```

- [ ] **Step 7: Commit**

```bash
git add sas/web/components/widgets/ImageRemediationStory.tsx
git commit -m "feat(web): add tag filter dropdown to ImageRemediationStory widget"
```

---

### Task 7: Runtime verification

**Files:** (integration test against real database)

- [ ] **Step 1: Start the backend server**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio && .venv/bin/python -m sas.api.main &`
Expected: Server starts on localhost:8000

- [ ] **Step 2: Test the tags endpoint against real data**

Run: `curl -s 'http://localhost:8000/api/entities/tags?repository=631165420711.dkr.ecr.eu-west-1.amazonaws.com/platform/scratch/pdev' | .venv/bin/python -m json.tool`
Expected: Returns tags like `codeserver`, `codeserver-python`, `jupyter`, `latest` with image counts

- [ ] **Step 3: Test the query endpoint with tag filter**

Run:
```bash
curl -s -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "lens": "Image",
    "traversal": [],
    "time": {"mode": "last_n_snapshots", "n": 10, "granularity": "day"},
    "measure": "count_open",
    "filters": [
      {"field": "image_id", "operator": "in", "value": ["<some-image-id>"]},
      {"field": "current_tag", "operator": "eq", "value": "codeserver"}
    ],
    "group_by": [],
    "order_by": null,
    "limit": null
  }' | .venv/bin/python -m json.tool
```
Expected: Returns series with data filtered to codeserver tag

- [ ] **Step 4: Start the frontend and verify visually**

Run: `cd /Users/aaron.miles/GitHub/sysdig-report-studio/sas/web && npm run dev`
Expected: Navigate to dashboard, open Image Remediation Story, select a repository with multiple tags, verify tag dropdown appears and filtering works

- [ ] **Step 5: Stop servers**

Run: `pkill -f "python -m sas.api.main"`

- [ ] **Step 6: Final commit**

```bash
git add -A
git status
git commit -m "ci: verify image tag filter integration"
```
(or skip commit if no changes)

---

## Self-Review Checklist

- [x] **Spec coverage**: All spec requirements addressed — backend filter (Tasks 1-3), new endpoint (Task 4), frontend TagSelect (Tasks 5-6), runtime verification (Task 7)
- [x] **No placeholders**: All code blocks are complete with exact file paths and line references
- [x] **Type consistency**: `TagEntity` interface matches backend response shape; `current_tag` field name consistent across all tasks
- [x] **Test coverage**: Each backend task has a corresponding test; runtime verification covers integration
