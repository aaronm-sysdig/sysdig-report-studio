"""Bullish — Runtime Vulnerability Findings Dashboard

Two data source modes:
  1. Upload / Local file  — drop a .csv.gz export or pick from data/reports/
  2. Fetch from API       — triggers an on-demand Sysdig Reporting job using
                            the global API token + region configured in the sidebar

Filter applied: Package In Use == true (actively executing at runtime)
Grouping: image-centric — one expandable ticket per unique container image.
"""
from __future__ import annotations

import csv
import gzip
import io
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config import SYSDIG_REGIONS

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "reports"
REPORT_NAME = "[PG] K8 Workload Vulnerability Findings"
REPORTING_API = "/api/platform/reporting/v1"

SEV_COL = {
    "Critical": "#D72638",
    "High":     "#F4821F",
    "Medium":   "#3B82F6",
    "Low":      "#6B7280",
}
SEV_ORD = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


# ── File helpers ───────────────────────────────────────────────────────────────

def short_image(img: str) -> str:
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


def _col(row: dict, *candidates: str, default: str = "") -> str:
    low = {k.lower(): v for k, v in row.items()}
    for c in candidates:
        v = low.get(c.lower())
        if v is not None:
            return v
    return default


def list_report_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.csv.gz"), key=lambda p: p.stat().st_mtime, reverse=True)


def _parse_findings(fileobj: IO) -> List[dict]:
    """Parse an open file-like object (text mode) of a Sysdig vulnerability CSV."""
    reader = csv.DictReader(fileobj)
    rows: List[dict] = []
    for raw in reader:
        in_use_val = _col(raw, "Package In Use", "In Use", "isRunning", "in_use").lower()
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
            "Severity":     sev,
            "CVE":          _col(raw, "Vulnerability Name", "CVE", "cve"),
            "Fix":          bool(fix_ver),
            "Workload":     _col(raw, "Kubernetes Workload Name", "Workload Name", "workload"),
            "Namespace":    _col(raw, "Kubernetes Namespace Name", "Namespace", "namespace"),
            "Cluster":      _col(raw, "Kubernetes Cluster Name", "Cluster", "cluster"),
            "WorkloadType": _col(raw, "Kubernetes Workload Type", "Workload Type", "workload_type"),
            "CVSS":         cvss,
            "Image":        _col(raw, "Image Name", "Image", "image_name", "image"),
            "Package":      _col(raw, "Package Name", "Package", "package_name"),
            "PkgType":      _col(raw, "Package Type", "package_type", "pkg_type"),
            "FixVersion":   fix_ver,
            "KEV":          _col(raw, "CISA KEV Publish Date", "cisa_kev_publish_date", "KEV"),
            "KEVDue":       _col(raw, "CISA KEV Due Date", "cisa_kev_due_date"),
        })
    return rows


def fetch_findings_from_path(path: Path) -> List[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return _parse_findings(f)


def fetch_findings_from_bytes(data: bytes, filename: str) -> List[dict]:
    if filename.endswith(".gz"):
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as f:
            return _parse_findings(f)
    return _parse_findings(io.StringIO(data.decode("utf-8")))


# ── Sysdig API helpers ─────────────────────────────────────────────────────────

def _auto_detect_base_url(token: str) -> Optional[str]:
    """Cycle through all known Sysdig regions and return the first base_url where the token is valid."""
    cache_key = f"_bullish_base_url_{token[:8]}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for host in SYSDIG_REGIONS.values():
        base_url = f"https://{host}"
        try:
            r = requests.get(f"{base_url}/api/v1/me", headers=headers, timeout=6)
            if r.status_code == 200:
                st.session_state[cache_key] = base_url
                return base_url
        except Exception:
            continue
    return None


def _api_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _find_report_id(base_url: str, token: str) -> Optional[Tuple[int, str]]:
    """Return (reportId, reportName) for the K8 Workload Vulnerability report, or None."""
    r = requests.get(
        f"{base_url}{REPORTING_API}/reports",
        headers=_api_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    reports_list = data if isinstance(data, list) else data.get("reports", [])
    for report in reports_list:
        name = report.get("name", "") or report.get("reportName", "")
        if REPORT_NAME.lower() in name.lower():
            rid = report.get("id") or report.get("reportId")
            return int(rid), name
    return None


def _trigger_job(base_url: str, token: str, report_id: int, report_name: str) -> int:
    """Create an on-demand job and return the job ID."""
    now = int(time.time())
    payload = {
        "reportId":         report_id,
        "isReportTemplate": False,
        "reportFormat":     "csv",
        "jobName":          report_name,
        "fileName":         report_name.replace(" ", "_").replace("[", "").replace("]", ""),
        "jobType":          "ON_DEMAND",
        "zones":            [],
        "timeFrame":        {"from": now - 86400, "to": now},
        "scheduledOn":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = requests.post(
        f"{base_url}{REPORTING_API}/jobs",
        headers=_api_headers(token),
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["id"]


def _poll_job(base_url: str, token: str, job_id: int, status_placeholder) -> Optional[str]:
    """Poll until job is COMPLETED; return fullFilePath or None on failure."""
    for attempt in range(40):
        time.sleep(4)
        r = requests.get(
            f"{base_url}{REPORTING_API}/jobs/{job_id}",
            headers=_api_headers(token),
            timeout=30,
        )
        r.raise_for_status()
        job = r.json()
        status = job.get("status", "UNKNOWN")
        status_placeholder.caption(f"Job status: **{status}** (check {attempt + 1}/40)…")
        if status == "COMPLETED":
            return job.get("fullFilePath")
        if status in ("FAILED", "CANCELLED"):
            return None
    return None


def fetch_findings_from_api(base_url: str, token: str) -> Tuple[List[dict], str]:
    """
    Run full on-demand job flow.
    Returns (rows, label) or raises on error.
    """
    result = _find_report_id(base_url, token)
    if result is None:
        raise RuntimeError(f"Report '{REPORT_NAME}' not found in Reports Manager for this account.")
    report_id, report_name = result

    job_id = _trigger_job(base_url, token, report_id, report_name)
    status_ph = st.empty()
    status_ph.caption(f"Job created (id={job_id}). Waiting for completion…")

    file_url = _poll_job(base_url, token, job_id, status_ph)
    status_ph.empty()

    if not file_url:
        raise RuntimeError("Job did not complete successfully within the timeout.")

    dl = requests.get(
        file_url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=120,
    )
    dl.raise_for_status()

    raw = io.BytesIO()
    shutil.copyfileobj(dl.raw, raw)
    raw.seek(0)
    data = raw.read()

    # Optionally cache to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    cache_path = DATA_DIR / f"{REPORT_NAME}_{ts}.csv.gz"
    cache_path.write_bytes(data)

    rows = fetch_findings_from_bytes(data, "report.csv.gz")
    label = f"API fetch — {report_name} ({ts})"
    return rows, label


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows: List[dict]) -> dict:
    sev_count = defaultdict(int)
    img_cves  = defaultdict(dict)
    img_wls   = defaultdict(set)
    img_sev   = defaultdict(lambda: defaultdict(int))
    cve_meta  = {}
    cve_imgs  = defaultdict(set)

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
    sorted_cves = sorted(
        cve_meta.items(),
        key=lambda x: (SEV_ORD.get(x[1]["Severity"], 9), -x[1]["CVSS"]),
    )
    return {
        "sev_count":     sev_count,
        "img_cves":      img_cves,
        "img_wls":       img_wls,
        "img_sev":       img_sev,
        "cve_meta":      cve_meta,
        "cve_img_count": cve_img_count,
        "sorted_imgs":   sorted_imgs,
        "sorted_cves":   sorted_cves,
        "total":         len(rows),
    }


# ── Charts ─────────────────────────────────────────────────────────────────────

def _severity_donut(sev_count: dict) -> go.Figure:
    order = ["Critical", "High", "Medium", "Low"]
    vals  = [sev_count.get(s, 0) for s in order]
    clrs  = [SEV_COL[s] for s in order]
    fig = go.Figure(go.Pie(
        labels=order, values=vals, hole=0.55,
        marker_colors=clrs,
        textinfo="label+value",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(showlegend=True, margin=dict(t=30, b=10, l=10, r=10), height=320)
    return fig


def _image_risk_chart(sorted_imgs: list, img_sev: dict, img_cves: dict) -> go.Figure:
    labels = [short_image(i) for i in sorted_imgs]
    counts = [len(img_cves[i]) for i in sorted_imgs]
    clrs   = [
        SEV_COL.get(min(img_sev[i], key=lambda s: SEV_ORD.get(s, 9)), "#8C9BAB")
        for i in sorted_imgs
    ]
    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation="h",
        marker_color=clrs, text=counts, textposition="outside",
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
        x=counts, y=labels, orientation="h",
        marker_color=clrs, text=counts, textposition="outside",
        hovertemplate="%{y}<br>Images affected: %{x}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Unique Images Affected",
        yaxis=dict(autorange="reversed"),
        height=max(220, len(sorted_cves) * 40 + 80),
        margin=dict(t=20, b=50, l=10, r=80),
    )
    return fig


# ── Page entry point ───────────────────────────────────────────────────────────

def render_page(*_args, **_kwargs):
    st.markdown("## Bullish — Runtime Vulnerability Findings")
    st.caption(
        "Source: **[PG] K8 Workload Vulnerability Findings** · "
        "Filter: **Package In Use** (actively running at runtime) · Image-centric grouping"
    )

    # ── Data source selection ──────────────────────────────────────────────────
    mode = st.radio(
        "Data source",
        ["Upload / Local file", "Fetch from Sysdig API"],
        horizontal=True,
        label_visibility="collapsed",
    )

    rows: List[dict] = []
    source_label = ""

    # ── Mode 1: Upload / Local file ────────────────────────────────────────────
    if mode == "Upload / Local file":
        uploaded = st.file_uploader(
            "Upload a Sysdig vulnerability CSV export (.csv or .csv.gz)",
            type=["csv", "gz"],
            key="bullish_upload",
        )

        local_files = list_report_files()
        selected_local: Optional[Path] = None
        if local_files:
            file_names   = ["— select —"] + [f.name for f in local_files]
            selected_name = st.selectbox("…or pick a cached file", file_names, index=0)
            if selected_name != "— select —":
                selected_local = DATA_DIR / selected_name

        col_btn, col_note = st.columns([1, 4])
        with col_btn:
            load = st.button("Load / Refresh", type="primary", use_container_width=True)
        with col_note:
            if "bullish_label" in st.session_state:
                st.caption(f"Loaded: {st.session_state['bullish_label']}")

        trigger = load or (
            "bullish_data" not in st.session_state
            and (uploaded is not None or selected_local is not None)
        )

        if trigger:
            if uploaded is not None:
                with st.spinner("Parsing uploaded file…"):
                    try:
                        data = uploaded.read()
                        rows = fetch_findings_from_bytes(data, uploaded.name)
                        source_label = f"Upload: {uploaded.name}"
                        st.session_state["bullish_data"]  = rows
                        st.session_state["bullish_label"] = source_label
                    except Exception as exc:
                        st.error(f"Failed to parse upload: {exc}")
                        return
            elif selected_local is not None:
                with st.spinner("Parsing local file…"):
                    try:
                        rows = fetch_findings_from_path(selected_local)
                        source_label = f"Local: {selected_local.name}"
                        st.session_state["bullish_data"]  = rows
                        st.session_state["bullish_label"] = source_label
                    except Exception as exc:
                        st.error(f"Failed to load file: {exc}")
                        return
            elif "bullish_data" not in st.session_state:
                st.info("Select a file above or upload one to load data.")
                return

        rows = st.session_state.get("bullish_data", [])
        if not rows and "bullish_data" not in st.session_state:
            st.info("Upload a `.csv.gz` export or pick a cached file, then click **Load / Refresh**.")
            return

    # ── Mode 2: Fetch from Sysdig API ─────────────────────────────────────────
    else:
        token = st.session_state.get("global_api_token", "")
        if not token:
            st.warning("No API token configured. Enter your API token in the sidebar.")
            return

        with st.spinner("Detecting region…"):
            base_url = _auto_detect_base_url(token)
        if not base_url:
            st.warning("Could not detect your Sysdig region. Check that your token is valid.")
            return

        st.caption(f"Region auto-detected: `{base_url}` · Report: `{REPORT_NAME}`")

        col_btn, col_note = st.columns([1, 4])
        with col_btn:
            fetch_btn = st.button("Fetch from API", type="primary", use_container_width=True)
        with col_note:
            if "bullish_label" in st.session_state:
                st.caption(f"Last loaded: {st.session_state['bullish_label']}")

        if fetch_btn:
            with st.spinner("Connecting to Sysdig Reports Manager…"):
                try:
                    rows, source_label = fetch_findings_from_api(base_url, token)
                    st.session_state["bullish_data"]  = rows
                    st.session_state["bullish_label"] = source_label
                    st.success(f"Fetched {len(rows):,} in-use findings.")
                except Exception as exc:
                    st.error(f"API fetch failed: {exc}")
                    return

        rows = st.session_state.get("bullish_data", [])
        if not rows and "bullish_data" not in st.session_state:
            st.info("Click **Fetch from API** to pull the latest report on-demand.")
            return

    # ── Guard ──────────────────────────────────────────────────────────────────
    if not rows:
        st.info("No in-use findings found in this report.")
        return

    D = aggregate(rows)

    # ── Metric tiles ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Findings", D["total"])
    m2.metric("Unique Images",  len(D["img_cves"]))
    m3.metric("Unique CVEs",    len(D["cve_meta"]))
    m4.metric("Clusters",       len({r["Cluster"] for r in rows}))
    m5.metric("CISA KEV CVEs",  sum(1 for m in D["cve_meta"].values() if m.get("KEV")))

    st.divider()

    # ── Severity distribution ──────────────────────────────────────────────────
    st.subheader("Severity Distribution")
    st.plotly_chart(_severity_donut(D["sev_count"]),
                    use_container_width=True, key="bullish_sev_donut")

    st.divider()

    # ── Image risk overview ────────────────────────────────────────────────────
    st.subheader("Image Risk Overview")
    st.caption("Each bar = number of unique CVEs in that image. Color = highest severity present.")
    st.plotly_chart(
        _image_risk_chart(D["sorted_imgs"], D["img_sev"], D["img_cves"]),
        use_container_width=True, key="bullish_img_risk",
    )

    st.divider()

    # ── CVE blast radius ───────────────────────────────────────────────────────
    st.subheader("CVE Blast Radius — Images Affected per CVE")
    st.caption("How many unique images each CVE affects.")
    st.plotly_chart(
        _blast_radius_chart(D["sorted_cves"], D["cve_img_count"]),
        use_container_width=True, key="bullish_blast",
    )

    st.divider()

    # ── CVE reference table ────────────────────────────────────────────────────
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

    # ── Per-image tickets ──────────────────────────────────────────────────────
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

    # ── CSV download ───────────────────────────────────────────────────────────
    st.divider()
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode()
    st.download_button(
        "Download all findings as CSV",
        data=csv_bytes,
        file_name="bullish_runtime_vulns.csv",
        mime="text/csv",
    )
