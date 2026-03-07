"""Bullish — Runtime Vulnerability Findings Dashboard

Fetches in-use + exploitable workload vulnerabilities directly from the
Sysdig Vulnerability API (two-step pattern, read-only GET requests only).

Filters applied server-side:
  - isRunning == True   (package actively executing at runtime)
  - exploitable == True (known public exploit exists)

Grouping: image-centric — one expandable ticket per unique container image.
"""
import time
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "https://api.au1.sysdig.com"

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

def fetch_findings(api_token: str) -> list[dict]:
    """Two-step Vulnerability API fetch.

    Step 1: GET /secure/vulnerability/v1/runtime-results
            Paginated, sorted desc by runningVulnTotalBySeverity.
            Early-stop when a batch hits zero-vuln entries.

    Step 2: GET /secure/vulnerability/v1/results/{resultId}
            Per-image CVE details, deduped by resultId.

    Returns a flat list of finding dicts (one row per in-use exploitable CVE
    × workload combination).
    """
    hdrs = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}

    # ── Step 1: collect (image × workload) entries with running vulns ─────────
    entries: list[dict] = []
    cursor = None
    while True:
        params: dict = {"limit": 100, "sort": "runningVulnTotalBySeverity", "order": "desc"}
        if cursor:
            params["cursor"] = cursor
        body  = _get(f"{API_BASE}/secure/vulnerability/v1/runtime-results", hdrs, params=params).json()
        batch = body.get("data", [])
        has_vulns = [
            e for e in batch
            if sum((e.get("runningVulnTotalBySeverity") or {}).values()) > 0
        ]
        entries.extend(has_vulns)
        cursor = body.get("page", {}).get("next")
        if not cursor or not batch or len(has_vulns) < len(batch):
            break   # reached the zero-vuln tail

    # ── Step 2: fetch CVE details per unique resultId ─────────────────────────
    result_cache: dict = {}
    rows: list[dict] = []

    for entry in entries:
        result_id = entry.get("resultId", "")
        img_name  = entry.get("mainAssetName", "")
        scope     = entry.get("scope", {})
        wl        = scope.get("kubernetes.workload.name", "")
        ns        = scope.get("kubernetes.namespace.name", "")
        cl        = scope.get("kubernetes.cluster.name", "")
        wl_type   = scope.get("kubernetes.workload.type", "")

        if not result_id:
            continue
        if result_id not in result_cache:
            result_cache[result_id] = _get(
                f"{API_BASE}/secure/vulnerability/v1/results/{result_id}", hdrs
            ).json()

        result = result_cache[result_id]
        if not result:
            continue

        packages        = result.get("packages", {})
        vulnerabilities = result.get("vulnerabilities", {})

        for v in vulnerabilities.values():
            pkg_id = v.get("packageRef", "")
            pkg    = packages.get(pkg_id, {})
            if not pkg.get("isRunning") or not v.get("exploitable"):
                continue

            sev     = (v.get("severity") or "").capitalize()
            cvss    = (v.get("cvssScore") or {}).get("score", 0.0)
            fix_ver = v.get("fixVersion") or ""
            kev     = (v.get("cisaKev") or {})

            pt_raw = (pkg.get("type") or "").lower()
            if "go" in pt_raw:
                pkg_type = "Golang"
            elif pt_raw in ("java", "maven", "gradle"):
                pkg_type = "Java"
            else:
                pkg_type = "OS"

            rows.append({
                "Severity":    sev,
                "CVE":         v.get("name", ""),
                "Fix":         bool(fix_ver),
                "Workload":    wl,
                "Namespace":   ns,
                "Cluster":     cl,
                "WorkloadType": wl_type,
                "CVSS":        cvss,
                "Image":       img_name,
                "Package":     pkg.get("name", ""),
                "PkgType":     pkg_type,
                "FixVersion":  fix_ver,
                "KEV":         kev.get("dateAdded", ""),
                "KEVDue":      kev.get("dueDate", ""),
            })

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

def render_page(api_token: str = "", region: str = "au1"):
    """Main Streamlit page for Bullish Runtime Vulnerability Findings."""
    st.markdown("## Bullish — Runtime Vulnerability Findings")
    st.caption(
        "Filters: **in-use** (package actively running at runtime) "
        "+ **exploitable** (known public exploit) · Image-centric grouping"
    )

    if not api_token:
        st.warning("Enter your API token in the sidebar to fetch live data.")
        return

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        refresh = st.button("🔄 Fetch / Refresh", type="primary", use_container_width=True)
    with col_note:
        if "bullish_data" in st.session_state:
            st.caption("Showing cached data. Click Fetch / Refresh to reload from API.")

    if refresh:
        st.session_state.pop("bullish_data", None)

    if "bullish_data" not in st.session_state:
        with st.spinner("Fetching runtime vulnerability data from Sysdig API…"):
            try:
                rows = fetch_findings(api_token)
                st.session_state["bullish_data"] = rows
            except Exception as exc:
                st.error(f"API error: {exc}")
                return

    rows: list[dict] = st.session_state.get("bullish_data", [])
    if not rows:
        st.info("No in-use exploitable findings returned by the API.")
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
