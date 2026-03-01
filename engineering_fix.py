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
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0bec5", size=12),
    margin=dict(t=40, b=20, l=20, r=20),
)

EXPECTED_COLS = {
    "clusterName", "findings", "imageReference",
    "imageRegistry", "imageRepository", "imageTag",
    "namespaceName", "resourceName",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _t2_hbar(items: dict, x_label: str, color: str) -> go.Figure:
    df = pd.DataFrame(sorted(items.items(), key=lambda x: x[1]),
                      columns=["Label", "Value"])
    fig = go.Figure(go.Bar(
        x=df["Value"], y=df["Label"], orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        text=df["Value"], textposition="outside",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(200, len(items) * 38 + 60),
                      xaxis=dict(title=x_label, gridcolor="#1e2d3d"),
                      yaxis=dict(showgrid=False))
    return fig


def _eng_load(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    df["findings"] = pd.to_numeric(df["findings"], errors="coerce").fillna(0).astype(int)
    df["imageLabel"] = df["imageRepository"].str.split("/").str[-1] + ":" + df["imageTag"]
    return df


def _eng_image_summary(df) -> pd.DataFrame:
    agg = (
        df.groupby(["imageRegistry", "imageRepository", "imageTag",
                    "imageReference", "imageLabel"])
        .agg(workloads      =("resourceName",  "nunique"),
             clusters       =("clusterName",   "nunique"),
             namespaces     =("namespaceName", "nunique"),
             total_findings =("findings",      "sum"),
             cluster_list   =("clusterName",   lambda x: ", ".join(sorted(x.unique()))),
             ns_list        =("namespaceName", lambda x: ", ".join(sorted(x.unique()))))
        .reset_index()
        .sort_values("total_findings", ascending=False)
        .reset_index(drop=True)
    )
    agg.insert(0, "Priority", range(1, len(agg) + 1))
    return agg


def _eng_repo_summary(df) -> pd.DataFrame:
    return (
        df.groupby("imageRepository")
        .agg(unique_tags    =("imageTag",      "nunique"),
             workloads      =("resourceName",  "nunique"),
             clusters       =("clusterName",   "nunique"),
             total_findings =("findings",      "sum"),
             tags           =("imageTag",      lambda x: ", ".join(sorted(x.unique()))))
        .reset_index()
        .sort_values("total_findings", ascending=False)
        .reset_index(drop=True)
    )


def _eng_top_images_bar(img_df, n: int = 25) -> go.Figure:
    top = img_df.head(n).sort_values("total_findings")
    fig = go.Figure(go.Bar(
        x=top["total_findings"], y=top["imageLabel"], orientation="h",
        marker=dict(color="#E53935", line=dict(width=0)),
        text=top["total_findings"], textposition="outside",
        customdata=top[["workloads", "clusters", "imageReference"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Total findings: %{x}<br>"
            "Workloads: %{customdata[0]}<br>Clusters: %{customdata[1]}<br>"
            "<i>%{customdata[2]}</i><extra></extra>"
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(320, len(top) * 30 + 60),
                      xaxis=dict(title="Total Findings", gridcolor="#1e2d3d"),
                      yaxis=dict(showgrid=False, tickfont=dict(size=10)))
    return fig


def _eng_cluster_bar(df) -> go.Figure:
    counts = (
        df.groupby("clusterName")["resourceName"].nunique()
        .reset_index().rename(columns={"resourceName": "workloads"})
        .sort_values("workloads")
    )
    fig = go.Figure(go.Bar(
        x=counts["workloads"], y=counts["clusterName"], orientation="h",
        marker=dict(color="#9B3FBF", line=dict(width=0)),
        text=counts["workloads"], textposition="outside",
        hovertemplate="%{y}: %{x} affected workloads<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(260, len(counts) * 32 + 60),
                      xaxis=dict(title="Affected Workloads", gridcolor="#1e2d3d"),
                      yaxis=dict(showgrid=False))
    return fig


def _eng_registry_donut(df) -> go.Figure:
    counts = df.groupby("imageRegistry")["imageReference"].nunique().reset_index()
    counts.columns = ["registry", "images"]
    palette = ["#00BFA5", "#E53935", "#9B3FBF", "#FB8C00",
               "#1E88E5", "#00C853", "#FF6F00", "#7C4DFF"]
    fig = go.Figure(go.Pie(
        labels=counts["registry"], values=counts["images"],
        marker=dict(colors=[palette[i % len(palette)] for i in range(len(counts))],
                    line=dict(width=2, color="#12161f")),
        hole=0.55, textinfo="label+value", textfont=dict(size=10),
        hovertemplate="%{label}<br>%{value} unique images (%{percent})<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                      title=dict(text="Unique vulnerable images by registry",
                                 font=dict(size=12, color="#90a4ae"), x=0.5))
    return fig


def _eng_cluster_image_heatmap(df, top_n: int = 30) -> go.Figure:
    top_labels = (
        df.groupby("imageLabel")["findings"].sum()
        .nlargest(top_n).index.tolist()
    )
    sub   = df[df["imageLabel"].isin(top_labels)]
    pivot = (
        sub.groupby(["imageLabel", "clusterName"])["resourceName"].nunique()
        .reset_index()
        .pivot(index="imageLabel", columns="clusterName", values="resourceName")
        .fillna(0).astype(int)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    z     = pivot.values.tolist()
    text  = [[str(int(v)) if v > 0 else "" for v in row] for row in z]
    fig   = go.Figure(go.Heatmap(
        z=z, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#12161f"], [0.3, "#1a2744"], [0.7, "#9B3FBF"], [1, "#E53935"]],
        showscale=True,
        colorbar=dict(title=dict(text="Workloads", font=dict(color="#90a4ae")),
                      tickfont=dict(color="#90a4ae")),
        hovertemplate=(
            "<b>%{y}</b><br>Cluster: %{x}<br>Workloads: %{z}<extra></extra>"
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      height=max(400, len(pivot) * 22 + 120),
                      xaxis=dict(title="Cluster", tickangle=-30,
                                 gridcolor="#1e2d3d", tickfont=dict(size=10)),
                      yaxis=dict(title="Image", autorange="reversed",
                                 gridcolor="#1e2d3d", tickfont=dict(size=10)))
    return fig


def _eng_findings_hist(df) -> go.Figure:
    counts = df["findings"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str), y=counts.values,
        marker=dict(color="#FB8C00", line=dict(width=0)),
        text=counts.values, textposition="outside",
        hovertemplate="Findings per workload: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      xaxis=dict(title="Findings per Workload", gridcolor="#1e2d3d"),
                      yaxis=dict(title="Workload Count", gridcolor="#1e2d3d"))
    return fig


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render_page():
    """Engineering Fix View — image-level action list from ClickHouse CSV."""

    st.markdown("""
<style>
[data-testid="stFileUploader"] {
    border: 2px dashed #37474f; border-radius: 12px;
    background: #12161f; transition: border-color .2s, background .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00C853; background: #0d1117;
}
[data-testid="stFileUploaderDropzone"] { padding: 40px 24px; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #78909c !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { color: #00C853 !important; }
.section-divider { border:none;border-top:1px solid #1e2d3d;margin:36px 0; }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-bottom:22px">
  <h1 style="color:#fff;font-size:1.8rem;font-weight:700;margin:0 0 6px">
    🔧 Engineering Fix View
  </h1>
  <p style="color:#78909c;font-size:.87rem;margin:0">
    Drop the CSV exported from the Sysdig ClickHouse query
    (<code>MATCH Vulnerability … RETURN clusterName, namespaceName, resourceName,
    imageReference, …</code>).
    Shows exactly which images to rebuild and where to redeploy them.
  </p>
</div>
""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        key="eng_file_uploader",
        help="Sysdig ClickHouse vulnerability findings export",
    )

    df_eng = None
    if uploaded is not None:
        try:
            df_eng = _eng_load(uploaded)
            st.success(f"Loaded **{len(df_eng):,}** rows from `{uploaded.name}`")
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    if df_eng is not None:
        img_df  = _eng_image_summary(df_eng)
        repo_df = _eng_repo_summary(df_eng)
        ts = datetime.now().strftime("%Y%m%d_%H%M")

        st.markdown("### Impact Summary")
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("Workloads Affected",  df_eng["resourceName"].nunique())
        e2.metric("Unique Images",        len(img_df), "to rebuild/patch")
        e3.metric("Image Repositories",   df_eng["imageRepository"].nunique())
        e4.metric("Clusters",             df_eng["clusterName"].nunique())
        e5.metric("Namespaces",           df_eng["namespaceName"].nunique())
        e6.metric("Total Findings",       int(df_eng["findings"].sum()))

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:4px'>"
            "🎯 What to Fix — Image Action List</div>"
            "<div style='color:#78909c;font-size:.82rem;margin-bottom:14px'>"
            "Each row is one image that needs to be rebuilt/patched. "
            "Priority is ranked by total findings.</div>",
            unsafe_allow_html=True,
        )

        with st.expander("📂 Group by Repository", expanded=False):
            st.dataframe(repo_df.rename(columns={
                "imageRepository": "Repository",
                "unique_tags":     "Vulnerable Tags",
                "workloads":       "Workloads",
                "clusters":        "Clusters",
                "total_findings":  "Total Findings",
                "tags":            "Tag Versions",
            }), use_container_width=True, hide_index=True)

        action_cols = {
            "Priority": "Priority", "imageLabel": "Image (name:tag)",
            "imageRegistry": "Registry", "workloads": "Workloads",
            "clusters": "Clusters", "namespaces": "Namespaces",
            "total_findings": "Total Findings", "cluster_list": "Cluster Names",
            "imageReference": "Full Image Reference",
        }
        action_df = img_df[list(action_cols.keys())].rename(columns=action_cols)

        def _sty_priority(v):
            if v <= 3:  return "color:#ef5350;font-weight:700"
            if v <= 10: return "color:#ffa726;font-weight:700"
            return "color:#90a4ae"

        st.dataframe(action_df.style.applymap(_sty_priority, subset=["Priority"]),
                     use_container_width=True, hide_index=True,
                     height=min(600, 36 * len(action_df) + 60))

        dl1, _ = st.columns([1, 5])
        with dl1:
            st.download_button("⬇️ Export action list",
                               data=action_df.to_csv(index=False),
                               file_name=f"fix_action_list_{ts}.csv",
                               mime="text/csv", key="dl_eng_action")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:14px'>"
            "📊 Visual Breakdown</div>", unsafe_allow_html=True)

        vt1, vt2, vt3 = st.tabs(["🖼️ Top Images", "🌐 Cluster Spread", "📦 Registry & Findings"])

        with vt1:
            st.plotly_chart(_eng_top_images_bar(img_df), use_container_width=True,
                            config={"displayModeBar": False})
        with vt2:
            vc1, vc2 = st.columns(2)
            with vc1:
                st.plotly_chart(_eng_cluster_bar(df_eng), use_container_width=True,
                                config={"displayModeBar": False})
            with vc2:
                ns_counts = (
                    df_eng.groupby("namespaceName")["resourceName"].nunique()
                    .reset_index().rename(columns={"resourceName": "workloads"})
                    .set_index("namespaceName")["workloads"].to_dict()
                )
                st.plotly_chart(_t2_hbar(ns_counts, "Workloads", "#1E88E5"),
                                use_container_width=True, config={"displayModeBar": False})
        with vt3:
            vc1, vc2 = st.columns(2)
            with vc1:
                st.plotly_chart(_eng_registry_donut(df_eng), use_container_width=True,
                                config={"displayModeBar": False})
            with vc2:
                st.plotly_chart(_eng_findings_hist(df_eng), use_container_width=True,
                                config={"displayModeBar": False})

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:4px'>"
            "🟥 Image × Cluster Heatmap</div>"
            "<div style='color:#78909c;font-size:.82rem;margin-bottom:14px'>"
            "Which images are running in which clusters. Each cell = workload count.</div>",
            unsafe_allow_html=True)
        st.plotly_chart(_eng_cluster_image_heatmap(df_eng), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:4px'>"
            "🔍 Per-Image Workload Detail</div>"
            "<div style='color:#78909c;font-size:.82rem;margin-bottom:14px'>"
            "Select an image to see every workload that needs redeployment.</div>",
            unsafe_allow_html=True)

        sel_image = st.selectbox("Select image", options=img_df["imageLabel"].tolist(),
                                 key="eng_img_sel")
        if sel_image:
            sub_df  = df_eng[df_eng["imageLabel"] == sel_image].copy()
            sel_ref = sub_df["imageReference"].iloc[0]
            st.markdown(
                f"<div style='background:#1a1f2e;border-radius:8px;padding:12px 18px;"
                f"border-left:4px solid #E53935;margin-bottom:14px'>"
                f"<div style='color:#90a4ae;font-size:.78rem;margin-bottom:2px'>Full image reference</div>"
                f"<div style='color:#fff;font-family:monospace;font-size:.9rem'>{sel_ref}</div>"
                f"</div>", unsafe_allow_html=True)
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Workloads",      sub_df["resourceName"].nunique())
            sm2.metric("Clusters",       sub_df["clusterName"].nunique())
            sm3.metric("Namespaces",     sub_df["namespaceName"].nunique())
            sm4.metric("Total Findings", int(sub_df["findings"].sum()))
            detail_df = (
                sub_df[["clusterName", "namespaceName", "resourceName", "findings"]]
                .drop_duplicates()
                .sort_values(["clusterName", "namespaceName", "resourceName"])
                .rename(columns={"clusterName": "Cluster", "namespaceName": "Namespace",
                                 "resourceName": "Workload", "findings": "Findings"})
                .reset_index(drop=True)
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True,
                         height=min(500, 36 * len(detail_df) + 60))
            st.download_button(
                f"⬇️ Export workloads for {sel_image}",
                data=detail_df.to_csv(index=False),
                file_name=f"workloads_{sel_image.replace(':', '_').replace('/', '_')}_{ts}.csv",
                mime="text/csv", key="dl_eng_detail")

    else:
        st.info(
            "👆 Drag and drop your Sysdig ClickHouse CSV export above, or click to browse.\n\n"
            "Expected columns: `clusterName`, `namespaceName`, `resourceName`, "
            "`imageReference`, `imageRegistry`, `imageRepository`, `imageTag`, `findings`."
        )
