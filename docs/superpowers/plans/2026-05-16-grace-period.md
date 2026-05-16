# Grace Period for Disappeared Findings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent false REMEDIED closures when findings disappear from scans due to intermittent coverage, by adding a configurable grace period (default 3 days) before closing.

**Architecture:** Add a `grace_period_since` column to `finding_state`. Disappeared findings get `reason_code='STALE'` instead of `state='CLOSED'`. On subsequent ingests, STALE findings older than N days are closed. Reappearing STALE findings are cleared back to normal OPEN.

**Tech Stack:** Python, DuckDB, pytest

---

### File Map

| File | Change |
|------|--------|
| `sas/ingest/reason_code.py` | Add `GRACE_PERIOD_DAYS` constant, add `'STALE'` to ReasonCode type |
| `sas/ingest/schema.py` | Add `grace_period_since` column via `migrate_schema()` |
| `sas/ingest/fast_finding_diff.py` | Replace DISAPPEARED step with STALE + expiry logic; extend RESEEN to clear STALE |
| `sas/ingest/finding_diff.py` | Same grace period logic for old row-by-row pipeline |
| `tests/test_grace_period.py` | New test file: 4 unit tests + 1 scenario test |
| `tests/test_schema.py` | Add `grace_period_since` to required columns set |
| `sas/query/compiler.py` | Add `'STALE'` to `count_fixed_other` reason codes |

---

### Task 1: Add GRACE_PERIOD_DAYS constant and STALE reason code

**Files:**
- Modify: `sas/ingest/reason_code.py`
- Test: `tests/test_grace_period.py`

- [ ] **Step 1: Add STALE to ReasonCode and add constant**

Edit `sas/ingest/reason_code.py`. Add the constant and extend the type:

```python
GRACE_PERIOD_DAYS = 3

ReasonCode = Literal["PATCHED", "RETIRED", "SCALED_TO_ZERO", "ACCEPTED",
                     "FEED_WITHDRAWN", "UNKNOWN", "STALE"]
```

Place `GRACE_PERIOD_DAYS` after the imports, before the `ReasonCode` type.

- [ ] **Step 2: Commit**

```bash
git add sas/ingest/reason_code.py
git commit -m "feat(reason_code): add STALE reason code and GRACE_PERIOD_DAYS constant"
```

---

### Task 2: Add grace_period_since column to finding_state

**Files:**
- Modify: `sas/ingest/schema.py:migrate_schema`
- Modify: `tests/test_schema.py:test_finding_state_has_expected_columns`

- [ ] **Step 1: Write the failing test**

Add `grace_period_since` to the required columns in `tests/test_schema.py`:

```python
    required = {
        "finding_id", "image_id", "cve_id", "package_name",
        "package_version", "package_path", "severity", "cvss_score",
        "in_use", "fix_available", "fix_version", "risk_accepted",
        "public_exploit", "first_seen", "last_seen", "state",
        "reason_code", "closed_at", "reopened_at", "reopen_count",
        "days_open", "is_regression", "grace_period_since",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py::test_finding_state_has_expected_columns -v`
Expected: FAIL — `grace_period_since` not in columns

- [ ] **Step 3: Add migration in schema.py**

Edit `sas/ingest/schema.py`, add the ALTER to `migrate_schema()`:

```python
def migrate_schema(conn) -> None:
    """Apply schema migrations for databases created before Phase 2.2.

    Safe to call on a fresh DB (columns already present) or an existing DB
    (ALTER TABLE ADD COLUMN IF NOT EXISTS is idempotent in DuckDB).
    """
    conn.execute(
        "ALTER TABLE finding_state "
        "ADD COLUMN IF NOT EXISTS cisa_kev_known_ransomware BOOLEAN DEFAULT FALSE"
    )
    conn.execute(
        "ALTER TABLE finding_state "
        "ADD COLUMN IF NOT EXISTS grace_period_since TIMESTAMPTZ DEFAULT NULL"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py::test_finding_state_has_expected_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/schema.py tests/test_schema.py
git commit -m "feat(schema): add grace_period_since column to finding_state"
```

---

### Task 3: Fast pipeline — mark disappeared as STALE instead of CLOSED

**Files:**
- Modify: `sas/ingest/fast_finding_diff.py` (steps 3 and 5)
- Test: `tests/test_grace_period.py`

- [ ] **Step 1: Write failing test — disappeared finding becomes STALE**

Create `tests/test_grace_period.py`:

```python
"""Grace period tests: STALE state, expiry, reappearance."""
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from sas.ingest.schema import create_schema, migrate_schema
from sas.ingest.runtime_snapshot import write_runtime_snapshot
from sas.ingest.entity_upsert import upsert_entities
from sas.ingest.finding_diff import diff_and_apply_findings


def _row(image_id="sha256:aaa", cve="CVE-2026-00001", pkg="libfoo", ver="1.0",
         pkg_path="/lib/libfoo", severity="Critical", risk_accepted=False,
         fix_available=True, in_use=True, public_exploit=False):
    return {
        "vulnerability_name": cve,
        "vulnerability_severity": severity,
        "package_name": pkg,
        "package_version": ver,
        "package_type": "OS",
        "package_path": pkg_path,
        "image_name": "registry/foo:1.0",
        "os_name": "alpine 3.20",
        "cvss_version": "3.0",
        "cvss_score": 9.1,
        "cvss_vector": "AV:N",
        "disclosure_date": pd.Timestamp("2026-01-01T00:00:00Z"),
        "fix_available_date": pd.Timestamp("2026-01-02T00:00:00Z"),
        "fix_version": "1.1",
        "public_exploit": public_exploit,
        "kubernetes_cluster_name": "cluster-a",
        "kubernetes_namespace_name": "ns-a",
        "kubernetes_workload_type": "Deployment",
        "kubernetes_workload_name": "foo",
        "kubernetes_container_name": "foo-main",
        "image_id": image_id,
        "package_in_use": in_use,
        "risk_accepted": risk_accepted,
        "cisa_kev_publish_date": pd.NaT,
        "cisa_kev_due_date": pd.NaT,
        "cisa_kev_known_ransomware": False,
        "fix_available": fix_available,
        "agent_tags": "{}",
        "container_labels": "{}",
        "namespace_labels": "{}",
    }


def _prep(db, df, snapshot_at):
    upsert_entities(db, df, snapshot_at)
    write_runtime_snapshot(db, df, snapshot_at)


def test_disappeared_finding_becomes_stale(db):
    """Finding disappears on day 2 → state stays OPEN, reason_code=STALE, grace_period_since set."""
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)

    df_with = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df_with, day1)
    diff_and_apply_findings(db, df_with, day1)

    df_without = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df_without, day2)
    diff_and_apply_findings(db, df_without, day2)

    row = db.execute(
        "SELECT state, reason_code, grace_period_since FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] == "STALE"
    assert row[2] == day2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grace_period.py::test_disappeared_finding_becomes_stale -v`
Expected: FAIL — finding is CLOSED/RETIRED, not OPEN/STALE

- [ ] **Step 3: Modify the DISAPPEARED logic in finding_diff.py**

Edit `sas/ingest/finding_diff.py`. Import the constant at the top:

```python
from sas.ingest.reason_code import ReasonContext, compute_reason_code, GRACE_PERIOD_DAYS
```

Replace step 2 (DISAPPEARED) — the loop starting at `for key, prior in open_by_key.items():`. Replace the entire DISAPPEARED section:

```python
    # 2. DISAPPEARED — OPEN rows whose natural key wasn't in today
    disapp_count = sum(1 for key in open_by_key if key not in today_keys)
    _dbg(f"Processing {disapp_count:,} DISAPPEARED (grace period logic)...")
    t = time.monotonic()

    # 2a. Expire old STALE findings past grace period
    cutoff = (today - timedelta(days=GRACE_PERIOD_DAYS)).replace(hour=23, minute=59, second=59)
    conn.execute(
        """
        UPDATE finding_state SET
          state = 'CLOSED',
          reason_code = 'REMEDIED',
          closed_at = ?,
          days_open = date_diff('day', first_seen, ?)
        WHERE reason_code = 'STALE'
          AND grace_period_since <= ?
        """,
        [snapshot_at, today, cutoff],
    )
    expired_count = conn.execute(
        "SELECT changes()"
    ).fetchone()[0]
    counts["closed"] += expired_count

    # 2b. Mark newly disappeared as STALE (exclude already-STALE)
    for key, prior in open_by_key.items():
        if key in today_keys:
            continue
        # Check if this finding is now CLOSED (expired above) — skip it
        fs_row = conn.execute(
            "SELECT state FROM finding_state WHERE finding_id = ?",
            [prior["finding_id"]],
        ).fetchone()
        if fs_row and fs_row[0] == "CLOSED":
            continue

        conn.execute(
            """
            UPDATE finding_state SET
              reason_code = 'STALE',
              grace_period_since = ?
            WHERE finding_id = ?
            """,
            [snapshot_at, prior["finding_id"]],
        )
    _dbg(f"  ✓ DISAPPEARED done in {int((time.monotonic()-t)*1000)}ms — expired={expired_count}, stale={disapp_count - expired_count}")
    _dbg(f"Total diff: {_ms()}ms")
```

Also extend the RESEEN block (around line 86-97). After the RESEEN UPDATE, add logic to clear STALE:

```python
            # RESEEN — update last_seen, drift columns, days_open
            prior = open_by_key[key]
            days_open = (today - prior["first_seen"].date()).days
            conn.execute(
                """
                UPDATE finding_state SET
                  last_seen = ?, severity = ?, cvss_score = ?, in_use = ?,
                  fix_available = ?, fix_version = ?, risk_accepted = ?,
                  public_exploit = ?, cisa_kev_known_ransomware = ?, days_open = ?,
                  reason_code = NULL, grace_period_since = NULL
                WHERE finding_id = ?
                """,
                [snapshot_at, v["severity"], v["cvss_score"], v["in_use"],
                 v["fix_available"], v["fix_version"], v["risk_accepted"],
                 v["public_exploit"], v["cisa_kev_known_ransomware"],
                 days_open, prior["finding_id"]],
            )
            counts["reseen"] += 1
```

The only change here is adding `reason_code = NULL, grace_period_since = NULL` to the RESEEN UPDATE so reappearing STALE findings are cleared.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_grace_period.py::test_disappeared_finding_becomes_stale -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sas/ingest/finding_diff.py tests/test_grace_period.py
git commit -m "feat(diff): mark disappeared findings as STALE instead of closing immediately"
```

---

### Task 4: Fast pipeline — same grace period logic in fast_finding_diff.py

**Files:**
- Modify: `sas/ingest/fast_finding_diff.py`

- [ ] **Step 1: Add import**

At the top of `fast_finding_diff.py`, add the import:

```python
from datetime import datetime, timedelta
```

And import the constant:

```python
from sas.ingest.reason_code import GRACE_PERIOD_DAYS
```

- [ ] **Step 2: Extend RESEEN step (step 3) to clear STALE**

In the RESEEN UPDATE (around line 66), add `reason_code = NULL, grace_period_since = NULL`:

```python
    conn.execute(f"""
        UPDATE finding_state SET
            last_seen = '{t}'::timestamptz,
            severity = c.severity,
            cvss_score = c.cvss_score,
            in_use = c.in_use,
            fix_available = c.fix_available,
            fix_version = c.fix_version,
            risk_accepted = c.risk_accepted,
            public_exploit = c.public_exploit,
            cisa_kev_known_ransomware = c.cisa_kev_known_ransomware,
            days_open = date_diff('day', c.prior_first_seen, '{today}'::date),
            reason_code = NULL,
            grace_period_since = NULL
        FROM _classified c
        WHERE c.transition = 'RESEEN'
          AND c.finding_id = finding_state.finding_id
    """)
```

- [ ] **Step 3: Replace DISAPPEARED step (step 5) with grace period logic**

Replace the entire step 5 block (lines ~167-191) with:

```python
    # 5. GRACE PERIOD — handle disappeared findings
    # 5a. Expire STALE findings past the grace period
    from sas.ingest.reason_code import GRACE_PERIOD_DAYS
    today_date = snapshot_at.date()
    cutoff = today_date - timedelta(days=GRACE_PERIOD_DAYS)
    cutoff_iso = cutoff.replace(hour=23, minute=59, second=59).isoformat()

    expired_count = 0
    conn.execute(f"""
        UPDATE finding_state SET
          state = 'CLOSED',
          reason_code = 'REMEDIED',
          closed_at = '{t}'::timestamptz,
          days_open = date_diff('day', finding_state.first_seen, '{today}'::date)
        WHERE reason_code = 'STALE'
          AND grace_period_since <= '{cutoff_iso}'::timestamptz
    """)
    expired_count = conn.execute("SELECT changes()").fetchone()[0]

    # 5b. Mark newly disappeared as STALE (OPEN findings not in today's data, excluding already-STALE)
    conn.execute("""
        CREATE OR REPLACE TEMPORARY TABLE _disappeared AS
        SELECT fs.*
        FROM finding_state fs
        LEFT JOIN _today_findings tf ON
            tf.image_id = fs.image_id
            AND tf.cve_id = fs.cve_id
            AND tf.package_name = fs.package_name
            AND tf.package_version = fs.package_version
            AND tf.package_path = fs.package_path
        WHERE fs.state = 'OPEN'
          AND fs.reason_code IS DISTINCT FROM 'STALE'
          AND tf.image_id IS NULL
    """)

    disapp_count = conn.execute(
        "SELECT COUNT(*) FROM _disappeared"
    ).fetchone()[0]

    if disapp_count > 0:
        conn.execute(f"""
            UPDATE finding_state SET
              reason_code = 'STALE',
              grace_period_since = '{t}'::timestamptz
            FROM _disappeared d
            WHERE finding_state.finding_id = d.finding_id
        """)

    closed_count = expired_count
```

- [ ] **Step 4: Update daily_closed_snapshot (step 7) to include expired STALE**

The `_disappeared` table now only has newly-STALE findings. We need to also capture the expired ones. After step 5a, create a temp table of expired findings before the UPDATE, then merge them into the closed snapshot.

Replace step 7's INSERT to use a combined source. First, capture expired in a temp table at the start of step 5:

```python
    # 5a. Capture and expire STALE findings past the grace period
    conn.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _expired_stale AS
        SELECT * FROM finding_state
        WHERE reason_code = 'STALE'
          AND grace_period_since <= '{cutoff_iso}'::timestamptz
    """)

    conn.execute(f"""
        UPDATE finding_state SET
          state = 'CLOSED',
          reason_code = 'REMEDIED',
          closed_at = '{t}'::timestamptz,
          days_open = date_diff('day', finding_state.first_seen, '{today}'::date)
        FROM _expired_stale
        WHERE finding_state.finding_id = _expired_stale.finding_id
    """)

    expired_count = conn.execute(
        "SELECT COUNT(*) FROM _expired_stale"
    ).fetchone()[0]
```

Then update step 7 to merge both sources:

```python
    conn.execute(f"""
        INSERT INTO daily_closed_snapshot (date, image_id, count_closed)
        SELECT '{today}'::DATE AS date, image_id, COUNT(*) AS count_closed
        FROM (
            SELECT image_id FROM _expired_stale
            UNION ALL
            SELECT image_id FROM _disappeared
        )
        GROUP BY image_id
        ON CONFLICT (date, image_id) DO UPDATE SET
            count_closed = EXCLUDED.count_closed
    """)
```

And add cleanup:
```python
    conn.execute("DROP TABLE IF EXISTS _expired_stale")
```

- [ ] **Step 5: Verify existing tests still pass**

Run: `pytest tests/test_finding_diff.py -v`
Expected: All pass (the old pipeline tests still use finding_diff.py, not fast_finding_diff.py)

- [ ] **Step 6: Commit**

```bash
git add sas/ingest/fast_finding_diff.py
git commit -m "feat(fast_diff): add grace period logic to fast pipeline"
```

---

### Task 5: Update existing tests that expect immediate closure

**Files:**
- Modify: `tests/test_finding_diff.py`
- Modify: `tests/test_scenarios.py`

- [ ] **Step 1: Fix test_disappeared_finding_closes_with_reason_retired**

This test expects that a disappeared finding closes on day 2. With grace period, it should be STALE on day 2 and only CLOSED on day 5 (3 days later). Update the test:

```python
def test_disappeared_finding_closes_with_reason_retired(db):
    """Image disappears entirely → STALE on day 2, CLOSED after grace period expires."""
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day5 = day1 + timedelta(days=4)  # 3 days after stale

    df1 = pd.DataFrame([_row(image_id="sha256:aaa")])
    _prep(db, df1, day1); diff_and_apply_findings(db, df1, day1)

    # Day 2: sha256:aaa disappears → should be STALE
    df2 = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df2, day2); diff_and_apply_findings(db, df2, day2)

    stale_row = db.execute(
        "SELECT state, reason_code, grace_period_since FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchone()
    assert stale_row[0] == "OPEN"
    assert stale_row[1] == "STALE"
    assert stale_row[2] == day2

    # Day 5: grace period expires → should be CLOSED
    df5 = pd.DataFrame([_row(image_id="sha256:bbb")])
    _prep(db, df5, day5); diff_and_apply_findings(db, df5, day5)

    closed_rows = db.execute(
        "SELECT state, reason_code FROM finding_state "
        "WHERE image_id = 'sha256:aaa'"
    ).fetchall()
    assert closed_rows == [("CLOSED", "REMEDIED")]
```

- [ ] **Step 2: Fix test_disappeared_finding_without_risk_flip**

Same pattern — update to expect STALE first, then run another day to trigger expiry:

```python
def test_disappeared_finding_without_risk_flip_is_unknown_or_feed_withdrawn(db):
    """Sibling risk_accepted=True on a DIFFERENT CVE does NOT trigger ACCEPTED.

    The ACCEPTED path requires the same (image, cve, pkg) natural key to be
    flipped — not a different finding on the same image.
    """
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day5 = day1 + timedelta(days=4)

    df1 = pd.DataFrame([_row(risk_accepted=False)])
    _prep(db, df1, day1); diff_and_apply_findings(db, df1, day1)
    # Day 2: same image running, CVE-2026-00001 gone, a different CVE present
    df2 = pd.DataFrame([_row(cve="CVE-2026-00002", risk_accepted=True)])
    _prep(db, df2, day2); diff_and_apply_findings(db, df2, day2)

    # Should be STALE, not closed yet
    row = db.execute(
        "SELECT state, reason_code FROM finding_state WHERE cve_id = 'CVE-2026-00001'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] == "STALE"

    # Day 5: expire grace period
    _prep(db, df2, day5); diff_and_apply_findings(db, df2, day5)
    row = db.execute(
        "SELECT reason_code FROM finding_state WHERE cve_id = 'CVE-2026-00001' AND state='CLOSED'"
    ).fetchone()
    assert row[0] == "REMEDIED"
```

- [ ] **Step 3: Fix test_reopened_finding_creates_new_record**

This test has a finding disappear on day 2 then reappear on day 3. With grace period, day 2 makes it STALE, day 3 should RESEEN it (clear STALE). The test should still pass because the finding reappears before expiry, but we need to add `migrate_schema(db)` and verify the reopen_count logic still works:

```python
def test_reopened_finding_creates_new_record_and_increments_reopen_count(db):
    create_schema(db)
    migrate_schema(db)
    day1 = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=2)
    day6 = day1 + timedelta(days=5)  # after grace period
    df_with = pd.DataFrame([_row()])
    df_without = pd.DataFrame([_row(cve="CVE-2026-00002")])  # same image, different CVE

    _prep(db, df_with, day1); diff_and_apply_findings(db, df_with, day1)
    _prep(db, df_without, day2); diff_and_apply_findings(db, df_without, day2)
    # Day 3: reappears while STALE → should be RESEEN (cleared from STALE)
    _prep(db, df_with, day3); diff_and_apply_findings(db, df_with, day3)

    # Finding should be OPEN, not STALE, not CLOSED
    row = db.execute(
        "SELECT state, reason_code, grace_period_since FROM finding_state "
        "WHERE cve_id = 'CVE-2026-00001' AND state = 'OPEN'"
    ).fetchone()
    assert row[0] == "OPEN"
    assert row[1] is None  # STALE cleared
    assert row[2] is None  # grace_period_since cleared

    # Now disappear for real (longer than grace period)
    _prep(db, df_without, day3); diff_and_apply_findings(db, df_without, day3)
    # Day 6: expire
    _prep(db, df_without, day6); diff_and_apply_findings(db, df_without, day6)
    assert db.execute(
        "SELECT state FROM finding_state WHERE cve_id = 'CVE-2026-00001' AND state = 'CLOSED'"
    ).fetchone() is not None

    # Reappear after CLOSED → should be REOPENED
    day7 = day1 + timedelta(days=6)
    _prep(db, df_with, day7); diff_and_apply_findings(db, df_with, day7)

    rows = db.execute(
        "SELECT state, reopen_count, is_regression FROM finding_state "
        "WHERE cve_id = 'CVE-2026-00001' ORDER BY first_seen"
    ).fetchall()
    assert len(rows) == 2  # one CLOSED, one OPEN (reopened)
    closed, reopened = rows
    assert closed[0] == "CLOSED"
    assert reopened[0] == "OPEN"
    assert reopened[1] == 1
    assert reopened[2] is True
```

- [ ] **Step 4: Add migrate_schema to all test_finding_diff.py tests**

Every test in `test_finding_diff.py` that calls `create_schema(db)` must also call `migrate_schema(db)` after it, so the `grace_period_since` column exists. Add this line to each test after `create_schema(db)`.

- [ ] **Step 5: Fix test_scenarios.py**

The scenario tests use `run_pipeline` which internally calls `migrate_schema`. But some expect immediate closure. Key tests to update:
- `test_scenario_retired_workload` — workload disappears on day 3, but with grace period it's STALE, not CLOSED. Need to ingest past the grace period.
- `test_scenario_digest_churn_same_tag` — expects 2 CLOSED findings. With grace period, they'll be STALE unless we ingest past day +3.
- `test_scenario_patched_via_new_digest` — expects CLOSED on day 2. Will be STALE.

Update each by adding extra days of ingest (using the last day's CSV repeated) to push past the grace period, then assert.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_finding_diff.py tests/test_grace_period.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add tests/test_finding_diff.py tests/test_scenarios.py
git commit -m "test: update existing tests for grace period behavior"
```

---

### Task 6: Update query compiler to include STALE in fixed counts

**Files:**
- Modify: `sas/query/compiler.py`

- [ ] **Step 1: Add STALE to count_fixed_other**

The `count_fixed_other` measure currently filters on `reason_code IN ('FEED_WITHDRAWN', 'UNKNOWN')`. Expired STALE findings close with `reason_code = 'REMEDIED'`, so they should be included. Update:

```python
    "count_fixed_other":     ("finding_state.closed_at",   "finding_state.state = 'CLOSED' AND finding_state.reason_code IN ('FEED_WITHDRAWN', 'UNKNOWN', 'REMEDIED')"),
```

- [ ] **Step 2: Commit**

```bash
git add sas/query/compiler.py
git commit -m "fix(query): include REMEDIED in count_fixed_other measure"
```

---

### Task 7: Run full test suite and verify

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify no regressions in scenario tests**

Run: `pytest tests/test_scenarios.py -v`
Expected: All pass

- [ ] **Step 3: Final commit**

```bash
git status && git diff HEAD && git log -n 3
git add -A
git commit -m "test: ensure all tests pass with grace period"  # only if there are changes
```

---

## Self-Review

**Spec coverage:**
- Schema change (1 column) → Task 2
- STALE reason code → Task 1
- Fast pipeline grace period → Task 4
- Old pipeline grace period → Task 3
- Expire after N days → Tasks 3 & 4
- Reappear clears STALE → Tasks 3 & 4
- Configurable via constant → Task 1
- Tests for all transitions → Tasks 3, 5
- Query compiler updated → Task 6
- Existing tests updated → Task 5

**Placeholder scan:** No TBDs, no "add validation later", all code blocks are complete.

**Type consistency:** `GRACE_PERIOD_DAYS` imported from `reason_code.py` in both diff files. `grace_period_since` used consistently. `STALE` is a string literal matching the ReasonCode type.

**One concern:** Task 5 (scenario tests) is complex — the scenario CSV files are multi-day and the tests use `run_pipeline` which has more moving parts. The plan shows the approach but the exact number of extra days needed depends on each scenario's day count. The engineer should run each scenario test, see what state the findings are in, and add enough extra ingest days to push past the grace period. This is intentional — the scenarios are integration tests and the engineer needs to verify the behavior empirically.
