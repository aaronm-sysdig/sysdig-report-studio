# Integration: Report Studio + Analytics Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge Prakash's `sysdig-coding` analytics app into `sysdig-report-studio` to form a single unified platform with shared sidebar navigation and global auth config.

**Architecture:** Multi-module, single entry point. A thin `app.py` handles navigation and global config. Each section (4 analytics pages + report studio) lives in its own module. Existing modules (`database.py`, `scheduler.py`, `pdf_generator.py`, `charts.py`) are untouched.

**Tech Stack:** Python, Streamlit, Plotly, pandas, reportlab, pyyaml, streamlit-sortables, requests

**Source repos:**
- Ours: `/Users/aaron.miles/GitHub/sysdig-report-studio/`
- Prakash's: `/Users/aaron.miles/GitHub/sysdig-coding/`

---

## Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add streamlit-sortables**

Open `requirements.txt` and add this line (Prakash uses it, we don't currently list it):

```
streamlit-sortables>=0.3.0
```

Final file should contain:
```
streamlit>=1.28.0
pandas>=2.0.0
pyyaml>=6.0
requests>=2.28.0
plotly>=5.18.0
kaleido>=0.2.1
reportlab>=4.0.0
streamlit-sortables>=0.3.0
```

**Step 2: Verify install**

```bash
pip install -r requirements.txt
```

Expected: all packages install or confirm already satisfied, no errors.

---

## Task 2: Expand config.py

**Files:**
- Modify: `config.py`

**Step 1: Add `get_api_config()` helper and `VULN_DATA_DIR` constant**

The current `config.py` only has region mappings. Add a `get_api_config()` function that reads the global auth from `st.session_state`, and a `VULN_DATA_DIR` constant to replace Prakash's hardcoded `~/sysdig-vuln-data/` path.

Replace the entire contents of `config.py` with:

```python
"""
Shared configuration for Sysdig Report Studio.

Region mappings and API helpers used across all modules.
"""
from pathlib import Path
import streamlit as st

# Region to hostname mapping
SYSDIG_REGIONS = {
    "APJ": "app.au1.sysdig.com",
    "US East": "secure.sysdig.com",
    "EU": "eu1.app.sysdig.com",
    "EU North": "app.eu2.sysdig.com",
    "US West": "us2.app.sysdig.com",
    "India": "app.in1.sysdig.com",
    "US West (GCP)": "app.us4.sysdig.com",
    "ME Central": "app.me2.sysdig.com"
}

# Directory for storing registry vulnerability snapshots
VULN_DATA_DIR = Path.home() / "sysdig-vuln-data"


def get_sysdig_host(region: str) -> str:
    """Get the Sysdig API hostname for a given region."""
    return SYSDIG_REGIONS.get(region, f"app.{region}.sysdig.com")


def get_api_config() -> tuple[str, str]:
    """
    Return (api_token, base_url) from global session state.

    All modules call this instead of reading env vars or managing their own
    sidebar auth inputs. Returns empty strings if not yet configured.
    """
    token = st.session_state.get("global_api_token", "")
    region = st.session_state.get("global_region", "APJ")
    host = get_sysdig_host(region)
    base_url = f"https://{host}"
    return token, base_url
```

**Step 2: Verify import works**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
python -c "from config import get_api_config, VULN_DATA_DIR, SYSDIG_REGIONS; print('OK')"
```

Expected: `OK`

---

## Task 3: Create posture_analytics.py

**Files:**
- Create: `posture_analytics.py`
- Source: `/Users/aaron.miles/GitHub/sysdig-coding/app.py` lines 64–85, 391–1657 (posture utility + chart functions + page function... wait — posture page is 1658–1866)

Actually the correct ranges from Prakash's `app.py`:
- `extract_date_from_filename`: lines 64–85 (shared utility)
- `load_data`: lines 391–443
- `load_multiple_files`: lines 444–509
- `create_executive_charts`: lines 510–614
- `create_person_charts`: lines 615–682
- `create_security_charts`: lines 683–777
- `create_trend_charts`: lines 778–881
- `create_downloadable_reports`: lines 882–938
- `posture_analytics_page`: lines 1658–1866

**Step 1: Create the file**

Create `posture_analytics.py` with this structure:

```python
"""
Posture Analytics page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
Chart functions return go.Figure objects; render_page() handles all st.plotly_chart() calls.
"""
import gzip
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── helpers, chart functions, and page function below ──
```

**Step 2: Copy functions from Prakash's app.py**

Copy the following functions verbatim from `/Users/aaron.miles/GitHub/sysdig-coding/app.py` into `posture_analytics.py`:

- `extract_date_from_filename` (lines 64–85)
- `load_data` (lines 391–443)
- `load_multiple_files` (lines 444–509)
- `create_executive_charts` (lines 510–614)
- `create_person_charts` (lines 615–682)
- `create_security_charts` (lines 683–777)
- `create_trend_charts` (lines 778–881)
- `create_downloadable_reports` (lines 882–938)
- `posture_analytics_page` (lines 1658–1866)

**Step 3: Rename page function**

Rename `posture_analytics_page` → `render_page` at the end of the file.

**Step 4: Verify chart functions return figures (not render them)**

Check `create_executive_charts`, `create_person_charts`, `create_security_charts`, `create_trend_charts`. Their docstrings already declare return tuples of figures. Confirm none of them call `st.plotly_chart()` internally. If any do, split them: move the `st.plotly_chart(fig)` call up to `render_page()` and have the helper return the `go.Figure`.

**Step 5: Verify import**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
python -c "import posture_analytics; print('OK')"
```

Expected: `OK` (no Streamlit runtime errors on import)

---

## Task 4: Create registry_analytics.py

**Files:**
- Create: `registry_analytics.py`
- Source: `/Users/aaron.miles/GitHub/sysdig-coding/app.py` lines 151–390, 939–1657

Ranges from Prakash's `app.py`:
- `SEVERITY_COLORS`, `SEVERITY_ORDER`: lines 102–112 (copy as module constants)
- `PLOTLY_LAYOUT`: lines 132–137 (copy as module constant)
- `fetch_registry_results`: lines 151–196
- `save_results_to_disk`: lines 197–224
- `list_saved_snapshots`: lines 225–260
- `load_snapshot`: lines 261–285
- `normalize_image_data`: lines 286–390
- `create_vuln_executive_charts`: lines 939–1148
- `create_vuln_trend_charts`: lines 1149–1300
- `_init_vuln_layout_state`: lines 1301–1310
- `_render_dashboard_widgets`: lines 1311–1366
- `vuln_analytics_page`: lines 1367–1657

**Step 1: Create the file**

```python
"""
Registry Vulnerability Analytics page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
Chart functions return go.Figure / dict-of-figures; render_page() handles all st.plotly_chart() calls.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from streamlit_sortables import sort_items

from config import get_api_config, VULN_DATA_DIR
```

**Step 2: Copy functions from Prakash's app.py**

Copy functions listed above verbatim. Note: `save_results_to_disk` and `list_saved_snapshots` reference `VULN_DATA_DIR` — in Prakash's code this is `Path.home() / "sysdig-vuln-data"`. Replace those references with `VULN_DATA_DIR` from our `config.py` (it's the same value, now centralised).

**Step 3: Rename page function**

Rename `vuln_analytics_page` → `render_page`.

**Step 4: Update auth in render_page**

In `render_page()` (was `vuln_analytics_page`), Prakash reads auth from environment:
```python
api_token = os.environ.get("SYSDIG_API_TOKEN", "")
```

Replace with:
```python
api_token, base_url = get_api_config()
```

Remove the sidebar block that shows API token status from env var — the global config in `app.py` handles this now. If `api_token` is empty, show:
```python
if not api_token:
    st.info("Set your API token in the sidebar to get started.")
    return
```

Also update `fetch_registry_results` calls to pass `base_url` if needed (check the function signature — it may use the module-level `SYSDIG_API_BASE` constant; replace that reference with `base_url` from `get_api_config()`).

**Step 5: Verify chart functions return figures**

`create_vuln_executive_charts` returns a dict of figures — confirmed already correct. `create_vuln_trend_charts` — verify it returns figures, not calls `st.plotly_chart` internally.

**Step 6: Verify import**

```bash
python -c "import registry_analytics; print('OK')"
```

Expected: `OK`

---

## Task 5: Create cve_risk.py

**Files:**
- Create: `cve_risk.py`
- Source: `/Users/aaron.miles/GitHub/sysdig-coding/app.py` lines 114–137, 1867–2358

Ranges:
- CVE constants (`CVE_DEFAULT_BASE`, `CVE_API_TIMEOUT`, etc.): lines 118–130
- `PLOTLY_LAYOUT`: lines 132–137
- `_cve_headers`: lines 1867–1874
- `_fetch_top_cves`: lines 1875–1898
- `_normalize_cve`: lines 1899–1918
- `_load_cves_with_progress`: lines 1919–1936
- `_cve_chart_severity_donut`: lines 1937–1950 (already returns go.Figure ✓)
- `_cve_chart_fix_donut`: lines 1951–1963 (already returns go.Figure ✓)
- `_cve_chart_epss_dist`: lines 1964–1990 (already returns go.Figure ✓)
- `_cve_chart_key_flags`: lines 1991–2007 (already returns go.Figure ✓)
- `_cve_render_section`: lines 2008–2235
- `cve_risk_page`: lines 2236–2358

**Step 1: Create the file**

```python
"""
CVE Risk Overview page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
All chart helpers already return go.Figure — no seam refactor needed.
"""
import re
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import get_api_config
```

**Step 2: Copy constants and functions verbatim**

Copy all constants and functions listed above. Do NOT copy the `st.text_input` blocks for API token/base URL from the sidebar section of `cve_risk_page` — these will be replaced.

**Step 3: Rename page function**

Rename `cve_risk_page` → `render_page`.

**Step 4: Update auth in render_page**

In `render_page()`, the sidebar currently has:
```python
api_base = st.text_input("Sysdig Base URL", value=os.environ.get("SYSDIG_API_BASE", CVE_DEFAULT_BASE), ...)
api_token = st.text_input("API Token", value=os.environ.get("SYSDIG_API_TOKEN", ""), ...)
```

Replace both with:
```python
api_token, api_base = get_api_config()
```

Remove those two `st.text_input` calls from the sidebar. Keep the "Refresh CVE data" button and metadata display in the sidebar.

If `api_token` is empty, show:
```python
if not api_token:
    st.info("Set your API token in the sidebar to get started.")
    return
```

**Step 5: Verify import**

```bash
python -c "import cve_risk; print('OK')"
```

Expected: `OK`

---

## Task 6: Create engineering_fix.py

**Files:**
- Create: `engineering_fix.py`
- Source: `/Users/aaron.miles/GitHub/sysdig-coding/app.py` lines 2359–2564

**Step 1: Create the file**

```python
"""
Engineering Fix View page for Sysdig Report Studio.

Extracted from Prakash's sysdig-coding/app.py.
CSV-upload driven — no API auth required for this page.
"""
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
```

**Step 2: Copy `engineering_fix_page` verbatim** (lines 2359–2564)

**Step 3: Rename**

Rename `engineering_fix_page` → `render_page`.

**Step 4: Check for auth usage**

This page is CSV-upload driven. Verify it does not read API token or base URL — if it does, update to use `get_api_config()` from `config.py`.

**Step 5: Verify import**

```bash
python -c "import engineering_fix; print('OK')"
```

Expected: `OK`

---

## Task 7: Create report_studio.py

**Files:**
- Create: `report_studio.py`
- Source: `app.py` (current) — extract all report builder logic

**Step 1: Create the file**

```python
"""
Report Studio page for Sysdig Report Studio.

Extracted from app.py. Handles report design, preview, PDF scheduling,
and template management.
"""
import json
import os
import urllib.parse
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

import database as db
from charts import create_chart_figure, calculate_trend_insights, create_insights_dataframe
from config import SYSDIG_REGIONS, get_sysdig_host, get_api_config
from scheduler import get_scheduler, run_template_now, start_scheduler, stop_scheduler
```

**Step 2: Copy from app.py into report_studio.py**

Move the following from the current `app.py`:

- `WIDGET_CATEGORIES` constant (lines 38–70)
- `ALL_WIDGETS` and `REGISTRY_WIDGET_KEYS` and `POSTURE_WIDGET_KEYS` (lines 72–79)
- All API helper functions: `fetch_zones`, `fetch_sysql_data`, `fetch_vulnerability_history` (lines ~111–246)
- All data utility functions: `format_datetime_columns`, `normalize_api_data`, `format_display_date` (lines ~247–298)
- `render_chart` function (lines ~299–354)
- All tab logic (Design, Preview, Reports) — lines ~565 onward
- All helper functions: `move_block`, `delete_block`, `_render_block_with_controls` (lines ~648–719)

**Step 3: Wrap tab logic in render_page()**

The Design/Preview/Reports tab block (currently at module level in `app.py`) needs to be wrapped in a `render_page()` function:

```python
def render_page(api_token: str, region: str, cust_name: str):
    """Render the Report Studio page (design, preview, reports tabs)."""
    # ... existing tab logic here ...
```

The sidebar Element Designer section should also move here — it renders when Report Studio is the selected tool. Wrap it in:

```python
def render_sidebar(api_token: str, region: str):
    """Render Report Studio sidebar controls (Element Designer)."""
    # ... existing Element Designer sidebar block ...
```

**Step 4: Update auth references**

Where `app.py` currently reads `api_token` and `region` from inline sidebar widgets, update `render_page()` and `render_sidebar()` to accept them as parameters (passed in from `app.py`'s global config).

**Step 5: Verify import**

```bash
python -c "import report_studio; print('OK')"
```

Expected: `OK`

---

## Task 8: Rewrite app.py

**Files:**
- Modify: `app.py` (strip to navigation shell + global config)

**Step 1: Replace app.py with the navigation shell**

After completing Tasks 3–7, replace `app.py` with:

```python
"""
Sysdig Report Studio — Main entry point.

Navigation and global config only. All page logic lives in dedicated modules.
"""
import streamlit as st

from config import SYSDIG_REGIONS
import posture_analytics
import registry_analytics
import cve_risk
import engineering_fix
import report_studio

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sysdig Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS (keep existing styling from old app.py) ───────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; }
    [data-testid="stSidebarHeader"] { display: none !important; }
    [data-testid="stSidebarContent"] { padding-top: 2rem !important; }
    [data-testid="stSidebar"] hr { margin: 0.5rem 0; }
    h1 { padding-top: 0 !important; padding-bottom: 0.5rem !important; margin-top: 0 !important; margin-bottom: 0.5rem !important; }
    h2, h3 { margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Global config sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo1.png", width=300)
    st.divider()

    region = st.selectbox("Region", list(SYSDIG_REGIONS.keys()), key="global_region")
    api_token = st.text_input("API Token", type="password", key="global_api_token")
    cust_name = st.text_input("Customer Name", value="Acme Corp", key="global_cust_name")

    st.divider()

    tool = st.radio(
        "Select Tool",
        options=[
            "📋  Posture Analytics",
            "🔍  Registry Vulnerabilities",
            "📊  CVE Risk Overview",
            "🔧  Engineering Fix View",
            "📄  Report Studio",
        ],
        index=0,
    )

    st.divider()

# ── Tool-specific sidebar + main content ──────────────────────────────────────
if tool == "📋  Posture Analytics":
    posture_analytics.render_page()
elif tool == "🔍  Registry Vulnerabilities":
    registry_analytics.render_page()
elif tool == "📊  CVE Risk Overview":
    cve_risk.render_page()
elif tool == "🔧  Engineering Fix View":
    engineering_fix.render_page()
else:
    report_studio.render_sidebar(api_token, region)
    report_studio.render_page(api_token, region, cust_name)
```

**Step 2: Verify the app launches**

```bash
cd /Users/aaron.miles/GitHub/sysdig-report-studio
streamlit run app.py
```

Expected: App opens in browser, sidebar shows logo + global config + tool selector, default page (Posture Analytics) renders without errors.

---

## Task 9: Retire posture.py and registry_vulns.py

**Files:**
- Delete: `posture.py`
- Delete: `registry_vulns.py`

**Step 1: Confirm nothing imports them**

```bash
grep -r "import posture\|from posture\|import registry_vulns\|from registry_vulns" /Users/aaron.miles/GitHub/sysdig-report-studio/ --include="*.py"
```

Expected: no matches (after Task 8 rewrite).

**Step 2: Delete the files**

```bash
rm posture.py registry_vulns.py
```

**Step 3: Verify app still launches**

```bash
streamlit run app.py
```

Expected: still works without errors.

---

## Task 10: Smoke Test All Sections

Manual checklist — run `streamlit run app.py` and verify each section:

**Global Config**
- [ ] Logo renders in sidebar
- [ ] Region selector changes region (verify `st.session_state.global_region` updates)
- [ ] API token input is masked

**Posture Analytics**
- [ ] Page renders with upload prompt when no files loaded
- [ ] Upload a CSV — executive and security charts render
- [ ] Upload 2+ CSVs — trend tab appears and renders
- [ ] Download reports button works

**Registry Vulnerabilities**
- [ ] Page renders with prompt if no API token set
- [ ] With token: Fetch Latest Data button triggers API call
- [ ] Snapshots list renders in sidebar
- [ ] Dashboard charts render from a loaded snapshot

**CVE Risk Overview**
- [ ] Page renders with prompt if no API token set
- [ ] With token: CVE data loads, in-use/not-in-use sections render
- [ ] Refresh button clears and reloads data

**Engineering Fix View**
- [ ] Page renders with file upload prompt
- [ ] Upload a valid CSV — fix list renders

**Report Studio**
- [ ] Element Designer appears in sidebar when Report Studio is selected
- [ ] Design tab: add an element, configure it, fetch data
- [ ] Preview tab: element renders as chart
- [ ] Reports tab: save, load, schedule all work
- [ ] PDF generation works

---

## Notes

**go.Figure seam status:**
- `create_executive_charts` — returns tuple including figures ✓
- `create_vuln_executive_charts` — returns dict of figures ✓
- CVE chart helpers (`_cve_chart_*`) — return `go.Figure` ✓
- `create_security_charts`, `create_trend_charts`, `create_person_charts`, `create_vuln_trend_charts` — verify during Task 3/4; refactor if they call `st.plotly_chart` internally

**Auth pattern across modules:**
- Analytics modules call `get_api_config()` from `config.py`
- Report Studio receives `api_token` and `region` as parameters from `app.py`
- No module has its own sidebar auth inputs

**Session state namespacing:**
- Posture: `posture_*` keys
- Registry: `vuln_*` and `reg_*` keys
- CVE: `t1_*` keys (already namespaced in Prakash's code)
- Report Studio: existing keys unchanged
