"""Bullish — Runtime Vulnerability Findings Dashboard

Data source: Sysdig Reporting v2 API
  Schedule: "PG - K8 Workload Vulnerability Findings"
  Endpoint: GET /api/scanning/reporting/v2/schedules/{id}/download  →  CSV

Filter applied locally:
  - In Use == true  (package actively executing at runtime)

Grouping: image-centric — one expandable ticket per unique container image.
"""
import csv
import io
import time
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

REPORT_SCHEDULE_NAME = "PG - K8 Workload Vulnerability Findings"

# Map sidebar region label → Reporting v2 API base URL (app. hostnames)
REGION_APP_BASE: dict[str, str] = {
    "US East (North Virginia)":    "https://secure.sysdig.com",
    "US West (Oregon, AWS)":       "https://us2.app.sysdig.com",
    "US-3":                        "https://us3.app.sysdig.com",
    "US West (Dallas, GCP)":       "https://us4.app.sysdig.com",
    "EU Central (Frankfurt)":      "https://eu1.app.sysdig.com",
    "EU North (Stockholm)":        "https://eu2.app.sysdig.com",
    "Asia Pacific (Sydney)":       "https://au1.app.sysdig.com",
    "Middle East (Dammam, GCP)":   "https://me2.app.sysdig.com",
    "Asia Pacific South (Mumbai)": "https://in1.app.sysdig.com",
}

SEV_COL = {
    "Critical": "#D72638",
    "High":     "#F4821F",
    "Medium":   "#3B82F6",
    "Low":      "#6B7280",
}
SEV_ORD = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


# ── Helpers ───────────────────────────────────────────────────────────────────

def short_image(img: str) -> str:
    """Return a human-readable short form of a full image URI."""
    if "@sha256:" in img:
        base   = img.split("@sha256:")[0]
        digest = img.split("@sha256:")[1][:12]
        name   = base.split("/")[-1]
        parent = base.split("/")[-2] if base.count("/") >= 2 else ""
        prefix = f"{parent}/" if parent and len(parent) < 25 else ""
        return f"{prefix}{name}@sha256:{digest}..."
    parts = img.split("/")
    name  = parts[-1]
    if len(parts) >= 2 and len(name) < 12:
        return f"{parts[-2]}/{name}"
    return name


def _get(url: str, hdrs: dict, params: dict | None = None,
         timeout: int = 60, retries: int = 4):
    """GET with exponential backoff on 429 / 5xx."""
    for attempt in range(retries):
        r = requests.get(url, headers=hdrs, params=params, timeout=timeout)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Failed after {retries} retries: {url}")


# ── API ───────────────────────────────────────────────────────────────────────

def auto_detect_region(api_token: str) -> tuple[str, str] | None:
    """Probe all known app base URLs with a schedules list request.

    Returns (region_label, app_base) for the first region that returns a
    valid JSON schedules response, or None if none succeed.
    """
    hdrs = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}
    probe = "/api/scanning/reporting/v2/reports"
    for label, base in REGION_APP_BASE.items():
        try:
            r = requests.get(f"{base}{probe}", headers=hdrs, timeout=10)
            if r.status_code == 200 and r.content:
                try:
                    body = r.json()
                    if isinstance(body, (list, dict)):
                        return label, base
                except Exception:
                    pass   # HTML or non-JSON — skip
        except Exception:
            continue
    return None


def _col(row: dict, *candidates: str, default: str = "") -> str:
    """Return the first matching column value from a CSV row (case-insensitive)."""
    low = {k.lower(): v for k, v in row.items()}
    for c in candidates:
        v = low.get(c.lower())
        if v is not None:
            return v
    return default


def fetch_findings(api_token: str, app_base: str,
                   status_cb=None) -> list[dict]:
    """Fetch vulnerability findings from the Sysdig Reporting v2 API.

    Step 1: GET /api/scanning/reporting/v2/schedules
            Find the schedule named REPORT_SCHEDULE_NAME, extract its ID.

    Step 2: GET /api/scanning/reporting/v2/schedules/{id}/download
            Download the latest generated CSV report.

    Step 3: Parse CSV rows, filter to In Use == true, map to internal format.

    status_cb(msg): optional callable to push live status text to the UI.
    """
    def _status(msg: str):
        if status_cb:
            status_cb(msg)

    hdrs_json = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}

    # ── Step 1: find the report template by name, then get its schedule ────────
    _status(f"Step 1 — searching for report '{REPORT_SCHEDULE_NAME}'…")

    def _parse_list(r):
        body = r.json() if r.content else []
        return body if isinstance(body, list) else body.get("data", [])

    reports = _parse_list(_get(f"{app_base}/api/scanning/reporting/v2/reports", hdrs_json))
    report_id = None
    for rpt in reports:
        if rpt.get("name") == REPORT_SCHEDULE_NAME:
            report_id = rpt.get("id")
            break

    if not report_id:
        names = [r.get("name", "") for r in reports[:20]]
        raise RuntimeError(
            f"Report '{REPORT_SCHEDULE_NAME}' not found in this account. "
            f"Reports visible: {names}"
        )

    _status(f"Step 1 — found report {report_id}, looking up its schedule…")
    schedules = _parse_list(
        _get(f"{app_base}/api/scanning/reporting/v2/reports/{report_id}/schedules", hdrs_json)
    )
    if not schedules:
        raise RuntimeError(
            f"Report '{REPORT_SCHEDULE_NAME}' has no schedules. "
            "Add a schedule in Reports Manager and wait for it to run."
        )
    schedule_id = schedules[0].get("id") or schedules[0].get("scheduleId")

    # ── Step 2: download the CSV ───────────────────────────────────────────────
    _status(f"Step 2 — downloading CSV report (schedule {schedule_id})…")
    hdrs_csv = {**hdrs_json, "Accept": "text/csv,application/octet-stream,*/*"}
    r2 = _get(
        f"{app_base}/api/scanning/reporting/v2/schedules/{schedule_id}/download",
        hdrs_csv,
    )
    if not r2.content:
        raise RuntimeError(
            "CSV download returned an empty file. "
            "The schedule may not have run yet — trigger a run in Reports Manager first."
        )

    # ── Step 3: parse CSV, filter in-use, map columns ─────────────────────────
    _status("Step 3 — parsing CSV and filtering in-use findings…")
    reader = csv.DictReader(io.StringIO(r2.text))
    rows: list[dict] = []

    for raw in reader:
        in_use_val = _col(raw, "In Use", "isRunning", "in_use", "running").lower()
        if in_use_val not in ("true", "yes", "1"):
            continue

        fix_ver  = _col(raw, "Fix Version", "fixVersion", "fix_version")
        cvss_str = _col(raw, "CVSS Score", "cvssScore", "cvss_score", "CVSS")
        try:
            cvss = float(cvss_str)
        except (ValueError, TypeError):
            cvss = 0.0
        sev = _col(raw, "Vulnerability Severity", "Severity", "severity").capitalize()

        rows.append({
            "Severity":    sev,
            "CVE":         _col(raw, "Vulnerability Name", "CVE", "cve", "vulnerability_name"),
            "Fix":         bool(fix_ver),
            "Workload":    _col(raw, "Kubernetes Workload Name", "Workload Name", "workload"),
            "Namespace":   _col(raw, "Kubernetes Namespace Name", "Namespace", "namespace"),
            "Cluster":     _col(raw, "Kubernetes Cluster Name", "Cluster", "cluster"),
            "WorkloadType": _col(raw, "Kubernetes Workload Type", "Workload Type", "workload_type"),
            "CVSS":        cvss,
            "Image":       _col(raw, "Image Name", "Image", "image_name", "image"),
            "Package":     _col(raw, "Package Name", "Package", "package_name"),
            "PkgType":     _col(raw, "Package Type", "package_type", "pkg_type"),
            "FixVersion":  fix_ver,
            "KEV":         _col(raw, "CISA KEV Publish Date", "cisa_kev_publish_date", "KEV"),
            "KEVDue":      _col(raw, "CISA KEV Due Date", "cisa_kev_due_date"),
        })

    _status(f"Done — {len(rows)} in-use findings loaded.")
    return rows


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    sev_count = defaultdict(int)
    img_cves  = defaultdict(dict)                         # img -> {cve: row}
    img_wls   = defaultdict(set)                          # img -> {(wl, ns, cl, type)}
    img_sev   = defaultdict(lambda: defaultdict(int))     # img -> {sev: count}
    cve_meta  = {}                                        # cve -> row
    cve_imgs  = defaultdict(set)                          # cve -> {img}

    for r in rows:
        sev, cve, img = r["Severity"], r["CVE"], r["Image"]
        sev_count[sev] += 1
        if cve not in img_cves[img]:
            img_cves[img][cve] = r
        img_wls[img].add((r["Workload"], r["Namespace"], r["Cluster"], r["WorkloadType"]))
        img_sev[img][sev] += 1
        if cve not in cve_meta:
            cve_meta[cve] = r
        cve_imgs[cve].add(img)

    sorted_imgs = sorted(
        img_cves,
        key=lambda i: (
            SEV_ORD.get(min(img_sev[i], key=lambda s: SEV_ORD.get(s, 9)), 9),
            -len(img_cves[i]),
        ),
    )
    cve_img_count = {cve: len(imgs) for cve, imgs in cve_imgs.items()}
    sorted_cves   = sorted(
        cve_meta.items(),
        key=lambda x: (SEV_ORD.get(x[1]["Severity"], 9), -x[1]["CVSS"]),
    )
    return {
        "sev_count":    sev_count,
        "img_cves":     img_cves,
        "img_wls":      img_wls,
        "img_sev":      img_sev,
        "cve_meta":     cve_meta,
        "cve_img_count": cve_img_count,
        "sorted_imgs":  sorted_imgs,
        "sorted_cves":  sorted_cves,
        "total":        len(rows),
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def _severity_donut(sev_count: dict) -> go.Figure:
    order  = ["Critical", "High", "Medium", "Low"]
    vals   = [sev_count.get(s, 0) for s in order]
    clrs   = [SEV_COL[s] for s in order]
    fig = go.Figure(go.Pie(
        labels=order, values=vals, hole=0.55,
        marker_colors=clrs,
        textinfo="label+value",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        margin=dict(t=30, b=10, l=10, r=10),
        height=320,
    )
    return fig


def _image_risk_chart(sorted_imgs: list, img_sev: dict, img_cves: dict) -> go.Figure:
    labels = [short_image(i) for i in sorted_imgs]
    counts = [len(img_cves[i]) for i in sorted_imgs]
    clrs   = [
        SEV_COL.get(min(img_sev[i], key=lambda s: SEV_ORD.get(s, 9)), "#8C9BAB")
        for i in sorted_imgs
    ]
    fig = go.Figure(go.Bar(
        x=counts, y=labels,
        orientation="h",
        marker_color=clrs,
        text=counts, textposition="outside",
        hovertemplate="%{y}<br>CVEs: %{x}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Unique CVEs per Image",
        yaxis=dict(autorange="reversed"),
        height=max(220, len(sorted_imgs) * 40 + 80),
        margin=dict(t=20, b=50, l=10, r=80),
    )
    return fig


def _blast_radius_chart(sorted_cves: list, cve_img_count: dict) -> go.Figure:
    labels = [cve for cve, _ in sorted_cves]
    counts = [cve_img_count.get(cve, 0) for cve, _ in sorted_cves]
    clrs   = [SEV_COL.get(meta["Severity"], "#8C9BAB") for _, meta in sorted_cves]
    fig = go.Figure(go.Bar(
        x=counts, y=labels,
        orientation="h",
        marker_color=clrs,
        text=counts, textposition="outside",
        hovertemplate="%{y}<br>Images affected: %{x}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Unique Images Affected",
        yaxis=dict(autorange="reversed"),
        height=max(220, len(sorted_cves) * 40 + 80),
        margin=dict(t=20, b=50, l=10, r=80),
    )
    return fig


# ── Page entry point ──────────────────────────────────────────────────────────

def render_page(api_token: str = "", region: str = "Asia Pacific (Sydney)"):
    """Main Streamlit page for Bullish Runtime Vulnerability Findings."""
    st.markdown("## Bullish — Runtime Vulnerability Findings")
    st.caption(
        f"Source: **{REPORT_SCHEDULE_NAME}** · "
        "Filter: **In Use** (package actively running at runtime) · Image-centric grouping"
    )

    if not api_token:
        st.warning("Enter your API token in the sidebar to fetch live data.")
        return

    # Derive app base from the selected region label; fall back to au1
    app_base = REGION_APP_BASE.get(region, "https://app.au1.sysdig.com")
    # If a previous auto-detect cached a working base, honour it
    if "bullish_app_base" in st.session_state:
        app_base = st.session_state["bullish_app_base"]

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        refresh = st.button("🔄 Fetch / Refresh", type="primary", use_container_width=True)
    with col_note:
        if "bullish_data" in st.session_state:
            st.caption("Showing cached data. Click Fetch / Refresh to reload from API.")

    if refresh:
        st.session_state.pop("bullish_data", None)
        st.session_state.pop("bullish_app_base", None)
        app_base = REGION_APP_BASE.get(region, "https://app.au1.sysdig.com")

    if "bullish_data" not in st.session_state:
        status_box = st.empty()
        prog_bar   = st.progress(0, text="Starting…")

        def update_status(msg: str):
            status_box.info(f"⏳ {msg}")
            if "Step 1" in msg and "searching" in msg:
                prog_bar.progress(10, text=msg)
            elif "Step 2" in msg:
                prog_bar.progress(50, text=msg)
            elif "Step 3" in msg:
                prog_bar.progress(80, text=msg)
            elif "Done" in msg:
                prog_bar.progress(100, text=msg)

        def _run_fetch(base: str) -> list[dict]:
            return fetch_findings(api_token, base, status_cb=update_status)

        try:
            update_status(f"Connecting to {app_base}…")
            rows = _run_fetch(app_base)
            prog_bar.progress(100, text="Done!")
            status_box.empty()
            prog_bar.empty()
            st.session_state["bullish_data"] = rows
            st.session_state["bullish_app_base"] = app_base
        except Exception as exc:
            err_str = str(exc)
            if any(code in err_str for code in ("401", "403", "404", "Unauthorized", "Not Found")):
                status_box.warning(
                    f"Cannot reach Reporting API on region **{region}** "
                    f"({err_str[:80]}). Auto-detecting correct region…"
                )
                prog_bar.progress(5, text="Probing regions…")
                detected = auto_detect_region(api_token)
                if detected:
                    det_label, det_base = detected
                    status_box.success(
                        f"Found working region: **{det_label}** (`{det_base}`). "
                        "Retrying fetch…"
                    )
                    prog_bar.progress(15, text=f"Fetching from {det_base}…")
                    st.session_state["bullish_app_base"] = det_base
                    try:
                        rows = _run_fetch(det_base)
                        prog_bar.progress(100, text="Done!")
                        status_box.empty()
                        prog_bar.empty()
                        st.session_state["bullish_data"] = rows
                    except Exception as exc2:
                        status_box.empty()
                        prog_bar.empty()
                        st.error(f"API error after auto-detect: {exc2}")
                        return
                else:
                    status_box.empty()
                    prog_bar.empty()
                    st.error(
                        "Could not find a working region for this token. "
                        "Please verify your API token is correct."
                    )
                    return
            else:
                status_box.empty()
                prog_bar.empty()
                st.error(f"API error: {exc}")
                return

    rows: list[dict] = st.session_state.get("bullish_data", [])
    if not rows:
        st.info("No in-use findings found in the report. The schedule may not have run yet.")
        return

    D = aggregate(rows)

    # ── Metric tiles ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Findings",  D["total"])
    m2.metric("Unique Images",   len(D["img_cves"]))
    m3.metric("Unique CVEs",     len(D["cve_meta"]))
    m4.metric("Clusters",        len({r["Cluster"] for r in rows}))
    m5.metric("CISA KEV CVEs",   sum(1 for m in D["cve_meta"].values() if m.get("KEV")))

    st.divider()

    # ── Severity distribution ─────────────────────────────────────────────────
    st.subheader("Severity Distribution")
    st.plotly_chart(_severity_donut(D["sev_count"]),
                    use_container_width=True, key="bullish_sev_donut")

    st.divider()

    # ── Image risk overview ───────────────────────────────────────────────────
    st.subheader("Image Risk Overview")
    st.caption("Each bar = number of unique CVEs in that image. Color = highest severity present.")
    st.plotly_chart(
        _image_risk_chart(D["sorted_imgs"], D["img_sev"], D["img_cves"]),
        use_container_width=True, key="bullish_img_risk",
    )

    st.divider()

    # ── CVE blast radius ──────────────────────────────────────────────────────
    st.subheader("CVE Blast Radius — Images Affected per CVE")
    st.caption("How many unique images each CVE affects.")
    st.plotly_chart(
        _blast_radius_chart(D["sorted_cves"], D["cve_img_count"]),
        use_container_width=True, key="bullish_blast",
    )

    st.divider()

    # ── CVE reference table ───────────────────────────────────────────────────
    st.subheader(f"CVE Reference — {len(D['cve_meta'])} Unique CVEs")
    cve_rows = [
        {
            "Severity":    meta["Severity"],
            "CVE":         cve,
            "CVSS":        meta["CVSS"],
            "Package":     meta["Package"],
            "Pkg Type":    meta["PkgType"],
            "Fix Version": meta["FixVersion"] or "—",
            "Fix?":        "YES" if meta["Fix"] else "NO",
            "Images":      D["cve_img_count"].get(cve, 0),
            "CISA KEV":    "YES" if meta.get("KEV") else "—",
        }
        for cve, meta in D["sorted_cves"]
    ]
    st.dataframe(pd.DataFrame(cve_rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Per-image tickets ─────────────────────────────────────────────────────
    st.subheader("Per-Image Vulnerability Tickets")
    st.caption(
        "Each ticket = one container image. "
        "Patch the image once and redeploy — all listed workloads are remediated."
    )

    for img in D["sorted_imgs"]:
        cves  = D["img_cves"][img]
        wls   = D["img_wls"][img]
        worst = min(D["img_sev"][img], key=lambda s: SEV_ORD.get(s, 9))
        label = (
            f"{short_image(img)}  ·  {len(cves)} CVE{'s' if len(cves) != 1 else ''}  "
            f"·  worst: {worst}  ·  {len(wls)} workload{'s' if len(wls) != 1 else ''}"
        )
        with st.expander(label):
            st.markdown("**Running workloads:**")
            wl_df = pd.DataFrame([
                {"Workload": w, "Namespace": n, "Cluster": c, "Type": t}
                for w, n, c, t in sorted(wls)
            ])
            st.dataframe(wl_df, hide_index=True, use_container_width=True)

            st.markdown("**CVEs (in-use + exploitable):**")
            cve_df_rows = [
                {
                    "Severity":    meta["Severity"],
                    "CVE":         cve,
                    "CVSS":        meta["CVSS"],
                    "Package":     meta["Package"],
                    "Pkg Type":    meta["PkgType"],
                    "Fix Version": meta["FixVersion"] or "—",
                    "CISA KEV":    "YES" if meta.get("KEV") else "—",
                }
                for cve, meta in sorted(
                    cves.items(),
                    key=lambda x: (SEV_ORD.get(x[1]["Severity"], 9), -x[1]["CVSS"]),
                )
            ]
            st.dataframe(pd.DataFrame(cve_df_rows), hide_index=True, use_container_width=True)

    # ── CSV download ──────────────────────────────────────────────────────────
    st.divider()
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode()
    st.download_button(
        "📥 Download all findings as CSV",
        data=csv_bytes,
        file_name="bullish_runtime_vulns.csv",
        mime="text/csv",
    )
