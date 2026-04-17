"""
dashboard/app.py
Medicare Part D Drug Spending Dashboard — multi-page
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import fetch_partd_data, split_overall_mftr, apply_outlier_filter, YEARS, _get_repo_root
from src.transforms import build_time_series, top_n_by_metric, melt_metric
from src.metrics import (
    fills_per_beneficiary,
    yoy_spend_change,
    total_program_spend,
    manufacturer_market_share,
    outlier_summary,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medicare Part D Drug Spending",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.stApp { background-color: #0f1117; color: #e8eaf0; }
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a3045;
}
.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #7b8bb2; margin-bottom: 6px;
}
.metric-value { font-size: 28px; font-weight: 600; color: #e8eaf0; line-height: 1.1; }
.metric-delta { font-size: 12px; margin-top: 4px; }
.delta-up   { color: #ff6b6b; }
.delta-down { color: #4ecdc4; }
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px; color: #c8d4f0;
    border-bottom: 1px solid #2a3a5c;
    padding-bottom: 10px; margin: 32px 0 20px 0;
}
.outlier-warning {
    background: linear-gradient(135deg, #2d1f0e, #3d2810);
    border: 1px solid #a0522d; border-radius: 8px;
    padding: 10px 16px; font-size: 13px;
    color: #f0a060; margin-bottom: 16px;
}
.drug-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    line-height: 1.7;
}
.drug-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 20px; color: #4e9af1;
    margin-bottom: 4px;
}
.drug-card-generic { font-size: 13px; color: #7b8bb2; margin-bottom: 12px; }
.drug-card-body    { font-size: 14px; color: #c8d4f0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="#0f1117", plot_bgcolor="#141824",
    font_color="#c8d4f0", font_family="DM Sans",
    colorway=["#4e9af1", "#f1c94e", "#4ecdc4", "#ff6b6b", "#a78bfa", "#f97316"],
    xaxis=dict(gridcolor="#1e2a40", linecolor="#2a3a5c", tickcolor="#2a3a5c"),
    yaxis=dict(gridcolor="#1e2a40", linecolor="#2a3a5c", tickcolor="#2a3a5c"),
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading CMS Part D data...")
def load():
    df = fetch_partd_data()
    df_overall, df_mftr = split_overall_mftr(df)
    return df, df_overall, df_mftr

@st.cache_data(show_spinner="Loading drug information...")
def load_drug_info():
    path = _get_repo_root() / "data" / "drug_information.csv"
    if not path.exists():
        return None
    di = pd.read_csv(path).iloc[:, 0:3]
    di.columns = ["Brand Name", "Generic Name", "Drug Uses"]
    return di

@st.cache_data(show_spinner="Loading therapeutic classifications...")
def load_drug_classes():
    path = _get_repo_root() / "data" / "drug_classes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

df, df_overall, df_mftr = load()
drug_info = load_drug_info()
drug_classes = load_drug_classes()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 Part D Explorer")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Spending Dashboard", "🧬 Browse by Therapy", "💊 Drug Information"],
                    label_visibility="collapsed")
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SPENDING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Spending Dashboard":

    with st.sidebar:
        exclude_outliers = st.toggle(
            "Exclude Outlier Records", value=True,
            help="Removes records where Outlier_Flag = 1 for the selected year"
        )
        st.markdown("### Drug Selection")
        all_drugs = sorted(df_overall["Brnd_Name"].dropna().unique())
        default_drug = "Ozempic" if "Ozempic" in all_drugs else all_drugs[0]
        selected_drug = st.selectbox("Select Drug", all_drugs, index=all_drugs.index(default_drug))

        st.markdown("### Manufacturer View")
        drug_mftrs = sorted(df_mftr[df_mftr["Brnd_Name"] == selected_drug]["Mftr_Name"].dropna().unique())
        selected_mftrs = st.multiselect("Manufacturers", drug_mftrs, default=drug_mftrs[:5])

        st.markdown("---")
        st.markdown("### Market Explorer")
        top_n = st.slider("Top N drugs", 5, 30, 15)
        metric_choice = st.selectbox(
            "Rank by (2023)",
            ["Tot_Spndng_2023", "Tot_Clms_2023", "Tot_Benes_2023"],
            format_func={
                "Tot_Spndng_2023": "Total Spending",
                "Tot_Clms_2023":   "Total Claims",
                "Tot_Benes_2023":  "Total Beneficiaries",
            }.get,
        )
        st.markdown("---")
        st.caption("Data: CMS Medicare Part D · 2019–2023")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='font-family:"DM Serif Display",serif; font-size:38px;
               background:linear-gradient(90deg,#4e9af1,#a78bfa);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:4px;'>
    Medicare Part D Drug Spending Dashboard
    </h1>
    <p style='color:#7b8bb2; font-size:14px; margin-top:0;'>
    CMS Data · 2019–2023 · Medicare Part D Program
    </p>
    """, unsafe_allow_html=True)

    if exclude_outliers:
        st.markdown(
            '<div class="outlier-warning">⚠️ Outlier records (Outlier_Flag = 1) are excluded. '
            "Toggle off in the sidebar to include them.</div>",
            unsafe_allow_html=True,
        )

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    drug_row_23 = apply_outlier_filter(df_overall[df_overall["Brnd_Name"] == selected_drug], 2023, exclude_outliers)
    drug_row_22 = apply_outlier_filter(df_overall[df_overall["Brnd_Name"] == selected_drug], 2022, exclude_outliers)

    def kpi(col, label, value, delta=None, delta_label="", prefix="$", is_positive_bad=True):
        delta_html = ""
        if delta is not None:
            arrow = "▲" if delta > 0 else "▼"
            if delta > 0:
                css = "delta-up" if is_positive_bad else "delta-down"
            else:
                css = "delta-down" if is_positive_bad else "delta-up"
            delta_html = f'<div class="metric-delta {css}">{arrow} {abs(delta):.1f}% {delta_label}</div>'
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{prefix}{value}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    if not drug_row_23.empty:
        spend_23 = drug_row_23["Tot_Spndng_2023"].sum()
        spend_22 = drug_row_22["Tot_Spndng_2022"].sum() if not drug_row_22.empty else None
        spend_delta = ((spend_23 - spend_22) / spend_22 * 100) if spend_22 else None
        kpi(col1, f"{selected_drug} · 2023 Total Spend", f"{spend_23/1e6:,.1f}M", spend_delta, "vs 2022", is_positive_bad=False)
        kpi(col2, "Total Claims 2023",        f"{drug_row_23['Tot_Clms_2023'].sum():,.0f}", prefix="")
        kpi(col3, "Total Beneficiaries 2023", f"{drug_row_23['Tot_Benes_2023'].sum():,.0f}", prefix="")
        kpi(col4, "Avg Spend Per Claim 2023", f"{drug_row_23['Avg_Spnd_Per_Clm_2023'].mean():,.2f}")

    # ── Section 1 — YoY Trend ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📈 Year-over-Year Trend · {selected_drug} (Overall)</div>', unsafe_allow_html=True)
    ts = build_time_series(df_overall, selected_drug,
        metrics=["Tot_Spndng", "Tot_Clms", "Tot_Benes"], exclude_outliers=exclude_outliers)

    if not ts.empty:
        fig1 = make_subplots(rows=1, cols=3, subplot_titles=["Total Spending ($)", "Total Claims", "Beneficiaries"])
        for i, (col_key, color, fill) in enumerate(zip(
            ["Tot_Spndng", "Tot_Clms", "Tot_Benes"],
            ["#4e9af1", "#4ecdc4", "#a78bfa"],
            ["rgba(78,154,241,0.08)", "rgba(78,205,196,0.08)", "rgba(167,139,250,0.08)"],
        ), 1):
            if col_key in ts.columns:
                fig1.add_trace(go.Scatter(
                    x=ts.index, y=ts[col_key], mode="lines+markers",
                    line=dict(width=3, color=color), marker=dict(size=9),
                    fill="tozeroy", fillcolor=fill, name=col_key, showlegend=False,
                ), row=1, col=i)
        fig1.update_layout(height=320, **PLOTLY_THEME, margin=dict(l=20, r=20, t=40, b=20))
        fig1.update_xaxes(tickvals=YEARS)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data available for the selected drug.")

    # ── Section 2 — Manufacturer Trend ───────────────────────────────────────
    st.markdown(f'<div class="section-header">🏭 Spending Trend by Manufacturer · {selected_drug}</div>', unsafe_allow_html=True)

    if selected_mftrs:
        mftr_rows = []
        for yr in YEARS:
            sub = apply_outlier_filter(
                df_mftr[(df_mftr["Brnd_Name"] == selected_drug) & (df_mftr["Mftr_Name"].isin(selected_mftrs))],
                yr, exclude_outliers)
            for _, row in sub.iterrows():
                mftr_rows.append({"Year": yr, "Manufacturer": row["Mftr_Name"],
                                  "Spending": row.get(f"Tot_Spndng_{yr}", np.nan)})
        mftr_df = pd.DataFrame(mftr_rows).dropna(subset=["Spending"])

        if not mftr_df.empty:
            fig2 = px.line(mftr_df, x="Year", y="Spending", color="Manufacturer", markers=True,
                color_discrete_sequence=PLOTLY_THEME["colorway"], labels={"Spending": "Total Spending ($)"})
            fig2.update_traces(line_width=2.5, marker_size=8)
            fig2.update_layout(height=360, **PLOTLY_THEME, margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a3a5c", borderwidth=1, font_size=12))
            fig2.update_xaxes(tickvals=YEARS)
            st.plotly_chart(fig2, use_container_width=True)
            share_df = manufacturer_market_share(df_mftr, selected_drug, year=2023)
            with st.expander("📊 2023 Manufacturer Market Share"):
                st.dataframe(share_df.style.format({"Spending": "${:,.0f}", "Share_Pct": "{:.1f}%"}).hide(axis="index"),
                             use_container_width=True)
        else:
            st.info("No manufacturer data for selected filters.")
    else:
        st.info("Select at least one manufacturer in the sidebar.")

    # ── Section 3 — Claims vs Beneficiaries ──────────────────────────────────
    st.markdown('<div class="section-header">👥 Claims vs Beneficiaries · 2023 Market Overview</div>', unsafe_allow_html=True)
    sub23 = apply_outlier_filter(df_overall.copy(), 2023, exclude_outliers)
    sub23 = sub23.dropna(subset=["Tot_Clms_2023", "Tot_Benes_2023", "Tot_Spndng_2023"])
    sub23 = sub23.nlargest(60, "Tot_Spndng_2023").copy()
    sub23["fills_per_bene"] = fills_per_beneficiary(sub23, year=2023)
    fig3 = px.scatter(sub23, x="Tot_Benes_2023", y="Tot_Clms_2023",
        size="Tot_Spndng_2023", color="fills_per_bene", hover_name="Brnd_Name",
        hover_data={"Tot_Spndng_2023": ":$,.0f", "fills_per_bene": ":.2f",
                    "Tot_Clms_2023": ":,.0f", "Tot_Benes_2023": ":,.0f"},
        labels={"Tot_Benes_2023": "Total Beneficiaries", "Tot_Clms_2023": "Total Claims",
                "fills_per_bene": "Fills per Bene"},
        color_continuous_scale="Viridis", size_max=60)
    fig3.update_layout(height=480, **PLOTLY_THEME, margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_colorbar=dict(title="Fills/Bene", tickfont_size=11))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Bubble size = Total Spending · Color = Fills per Beneficiary · Top 60 drugs by spend")

    # ── Section 4 — Price Change % ────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Price Change 2022 → 2023 · % Change per Dosage Unit</div>', unsafe_allow_html=True)
    chg_df = apply_outlier_filter(df_overall.copy(), 2023, exclude_outliers)
    chg_df = chg_df.dropna(subset=["Chg_Avg_Spnd_Per_Dsg_Unt_22_23", "Tot_Spndng_2023", "Tot_Spndng_2022"])
    chg_df = chg_df[chg_df["Tot_Spndng_2023"] > 1_000_000].copy()
    chg_df["Chg_%"] = (chg_df["Chg_Avg_Spnd_Per_Dsg_Unt_22_23"] * 100).round(1)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 🟢 Largest Price Increases")
        top_inc = chg_df.nlargest(15, "Chg_%").sort_values("Chg_%")
        fig4a = px.bar(top_inc, x="Chg_%", y="Brnd_Name", orientation="h",
            color="Chg_%", color_continuous_scale=[[0, "white"], [1, "#1a7a3a"]],
            labels={"Chg_%": "% Change", "Brnd_Name": ""})
        fig4a.update_layout(height=400, **PLOTLY_THEME, margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False)
        st.plotly_chart(fig4a, use_container_width=True)
    with col_b:
        st.markdown("##### 🔴 Largest Price Decreases")
        top_dec = chg_df.nsmallest(15, "Chg_%").sort_values("Chg_%", ascending=False).copy()
        top_dec["Chg_Abs_%"] = top_dec["Chg_%"].abs()
        fig4b = px.bar(top_dec, x="Chg_%", y="Brnd_Name", orientation="h",
            color="Chg_Abs_%", color_continuous_scale=[[0, "white"], [1, "#cc1111"]],
            labels={"Chg_%": "% Change", "Brnd_Name": "", "Chg_Abs_%": "Magnitude"})
        fig4b.update_layout(height=400, **PLOTLY_THEME, margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False)
        st.plotly_chart(fig4b, use_container_width=True)

    # ── Section 5 — Price Change $ ────────────────────────────────────────────
    st.markdown('<div class="section-header">💵 Spending Change 2022 → 2023 · Total Dollar Difference</div>', unsafe_allow_html=True)
    chg_df["Spend_Diff"] = chg_df["Tot_Spndng_2023"] - chg_df["Tot_Spndng_2022"]
    chg_valid = chg_df.dropna(subset=["Spend_Diff"])

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("##### 🟢 Largest Spending Increases ($)")
        top_inc_usd = chg_valid.nlargest(15, "Spend_Diff").sort_values("Spend_Diff")
        fig5a = px.bar(top_inc_usd, x="Spend_Diff", y="Brnd_Name", orientation="h",
            color="Spend_Diff", color_continuous_scale=[[0, "white"], [1, "#1a7a3a"]],
            hover_data={"Tot_Spndng_2022": ":$,.0f", "Tot_Spndng_2023": ":$,.0f", "Chg_%": ":.1f"},
            labels={"Spend_Diff": "Spending Change ($)", "Brnd_Name": "",
                    "Tot_Spndng_2022": "2022 Spend", "Tot_Spndng_2023": "2023 Spend"})
        fig5a.update_layout(height=400, **PLOTLY_THEME, margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False)
        st.plotly_chart(fig5a, use_container_width=True)
    with col_d:
        st.markdown("##### 🔴 Largest Spending Decreases ($)")
        top_dec_usd = chg_valid.nsmallest(15, "Spend_Diff").sort_values("Spend_Diff", ascending=False).copy()
        top_dec_usd["Spend_Diff_Abs"] = top_dec_usd["Spend_Diff"].abs()
        fig5b = px.bar(top_dec_usd, x="Spend_Diff", y="Brnd_Name", orientation="h",
            color="Spend_Diff_Abs", color_continuous_scale=[[0, "white"], [1, "#cc1111"]],
            hover_data={"Tot_Spndng_2022": ":$,.0f", "Tot_Spndng_2023": ":$,.0f", "Chg_%": ":.1f"},
            labels={"Spend_Diff": "Spending Change ($)", "Brnd_Name": "", "Spend_Diff_Abs": "Magnitude",
                    "Tot_Spndng_2022": "2022 Spend", "Tot_Spndng_2023": "2023 Spend"})
        fig5b.update_layout(height=400, **PLOTLY_THEME, margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False)
        st.plotly_chart(fig5b, use_container_width=True)
    st.caption("Dollar difference = Tot_Spndng_2023 − Tot_Spndng_2022 · Hover to see both years and % change")

    # ── Section 6 — CAGR Quadrant ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 5-Year CAGR vs 2023 Spending · Growth Quadrant</div>', unsafe_allow_html=True)
    cagr_df = apply_outlier_filter(df_overall.copy(), 2023, exclude_outliers)
    cagr_df = cagr_df.dropna(subset=["CAGR_Avg_Spnd_Per_Dsg_Unt_19_23", "Tot_Spndng_2023"])
    cagr_df = cagr_df[cagr_df["Tot_Spndng_2023"] > 10_000_000].copy()
    cagr_df["CAGR_%"] = cagr_df["CAGR_Avg_Spnd_Per_Dsg_Unt_19_23"] * 100
    med_spend = float(cagr_df["Tot_Spndng_2023"].median())
    med_cagr  = float(cagr_df["CAGR_%"].median())
    fig6 = px.scatter(cagr_df, x="Tot_Spndng_2023", y="CAGR_%", hover_name="Brnd_Name",
        size="Tot_Clms_2023", color="CAGR_%", color_continuous_scale="RdYlGn",
        labels={"Tot_Spndng_2023": "2023 Total Spending ($)", "CAGR_%": "5-Year CAGR (%)"}, size_max=40)
    fig6.add_hline(y=med_cagr, line_dash="dot", line_color="#7b8bb2",
        annotation_text=f"Median CAGR: {med_cagr:.1f}%", annotation_font_color="#7b8bb2")
    fig6.add_vline(x=med_spend, line_dash="dot", line_color="#7b8bb2",
        annotation_text="Median Spend", annotation_font_color="#7b8bb2")
    # Quadrant labels
    x_max = float(cagr_df["Tot_Spndng_2023"].max())
    y_max = float(cagr_df["CAGR_%"].max())
    y_min = float(cagr_df["CAGR_%"].min())
    for qx, qy, qtxt, qanchor in [
        (x_max, y_max,  "High Spend · High Growth",  "right"),
        (x_max, y_min,  "High Spend · Low Growth",   "right"),
        (med_spend * 0.05, y_max,  "Low Spend · High Growth",  "left"),
        (med_spend * 0.05, y_min,  "Low Spend · Low Growth",   "left"),
    ]:
        fig6.add_annotation(x=qx, y=qy, text=qtxt, showarrow=False,
            font=dict(size=10, color="#4a5568"), xanchor=qanchor, yanchor="top" if qy == y_max else "bottom")
    fig6.update_layout(height=500, **PLOTLY_THEME, margin=dict(l=20, r=20, t=20, b=20))
    fig6.update_xaxes(type="log")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Bubble size = Total Claims · X-axis log scale · Quadrant lines = median spend and median CAGR")

    # ── Section 7 — Program Spend Over Time ──────────────────────────────────
    st.markdown('<div class="section-header">📉 Total Part D Program Spend · 2019–2023</div>', unsafe_allow_html=True)
    prog_spend = total_program_spend(df_overall, exclude_outliers=exclude_outliers)
    fig7 = px.bar(prog_spend, x="Year", y="Total_Spending",
        color="Total_Spending", color_continuous_scale="Blues",
        labels={"Total_Spending": "Total Spending ($)"},
        text=prog_spend["Total_Spending"].apply(lambda x: f"${x/1e9:.1f}B"))
    fig7.update_traces(textposition="outside")
    fig7.update_layout(height=350, **PLOTLY_THEME, margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False)
    fig7.update_xaxes(tickvals=YEARS)
    st.plotly_chart(fig7, use_container_width=True)

    # ── Section 8 — Top N Ranking ─────────────────────────────────────────────
    metric_labels = {"Tot_Spndng_2023": "Total Spending", "Tot_Clms_2023": "Total Claims", "Tot_Benes_2023": "Total Beneficiaries"}
    st.markdown(f'<div class="section-header">🏆 Top {top_n} Drugs · {metric_labels.get(metric_choice, metric_choice)} (2023)</div>', unsafe_allow_html=True)
    top_df = top_n_by_metric(df_overall, metric_col=metric_choice, n=top_n,
                              exclude_outliers=exclude_outliers, year=2023)
    fig8 = px.bar(top_df.sort_values(metric_choice), x=metric_choice, y="Brnd_Name", orientation="h",
        color=metric_choice, color_continuous_scale="Blues", hover_data={"Gnrc_Name": True},
        labels={metric_choice: metric_labels.get(metric_choice, metric_choice), "Brnd_Name": ""})
    fig8.update_layout(height=max(350, top_n * 28), **PLOTLY_THEME,
        margin=dict(l=20, r=20, t=10, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig8, use_container_width=True)

    with st.expander("🔍 Outlier Record Summary by Year"):
        out_df = outlier_summary(df_overall)
        st.dataframe(out_df.style.format({"Outlier_Pct": "{:.2f}%"}).hide(axis="index"),
                     use_container_width=True)
        st.caption("Shows how many records are excluded when 'Exclude Outlier Records' is toggled on.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DRUG INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💊 Drug Information":

    with st.sidebar:
        st.markdown("### Search")
        search_query = st.text_input("Search drug name or use", placeholder="e.g. Ozempic, diabetes...")
        st.markdown("---")
        st.caption("Source: drug_information.csv")

    st.markdown("""
    <h1 style='font-family:"DM Serif Display",serif; font-size:38px;
               background:linear-gradient(90deg,#4ecdc4,#a78bfa);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:4px;'>
    Drug Information
    </h1>
    <p style='color:#7b8bb2; font-size:14px; margin-top:0;'>
    Drug uses reference · Medicare Part D formulary
    </p>
    """, unsafe_allow_html=True)

    if drug_info is None:
        st.error("Could not load `data/drug_information.csv`. Make sure it exists at the repo root `data/` folder.")
        st.stop()

    # ── Filter ────────────────────────────────────────────────────────────────
    if search_query:
        mask = (
            drug_info["Brand Name"].str.contains(search_query, case=False, na=False) |
            drug_info["Generic Name"].str.contains(search_query, case=False, na=False) |
            drug_info["Drug Uses"].str.contains(search_query, case=False, na=False)
        )
        filtered = drug_info[mask].reset_index(drop=True)
    else:
        filtered = drug_info.reset_index(drop=True)

    st.markdown(
        f"<p style='color:#7b8bb2; font-size:13px;'>Showing {len(filtered):,} of {len(drug_info):,} drugs</p>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No drugs matched your search.")
    else:
        page_size = 20
        total_pages = max(1, (len(filtered) - 1) // page_size + 1)
        if total_pages > 1:
            card_page = st.number_input("Page", min_value=1, max_value=total_pages,
                                        value=1, step=1, label_visibility="collapsed")
        else:
            card_page = 1
        start = (card_page - 1) * page_size
        page_df = filtered.iloc[start: start + page_size]

        for _, row in page_df.iterrows():
            uses = row["Drug Uses"] if pd.notna(row["Drug Uses"]) else "Drug uses not available."
            st.markdown(f"""
            <div class="drug-card">
                <div class="drug-card-title">{row['Brand Name']}</div>
                <div class="drug-card-generic">{row['Generic Name']}</div>
                <div class="drug-card-body">{uses}</div>
            </div>
            """, unsafe_allow_html=True)

        if total_pages > 1:
            st.markdown(
                f"<p style='color:#7b8bb2; font-size:12px; text-align:center;'>Page {card_page} of {total_pages}</p>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BROWSE BY THERAPY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Browse by Therapy":

    with st.sidebar:
        st.markdown("### Classification")
        class_system = st.radio(
            "Browse by",
            ["ATC (Mechanism)", "MeSH (Disease / Condition)"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Source: RxNorm / RxClass · NLM")

    st.markdown("""
    <h1 style='font-family:"DM Serif Display",serif; font-size:38px;
               background:linear-gradient(90deg,#f97316,#f1c94e);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:4px;'>
    Browse by Therapy
    </h1>
    <p style='color:#7b8bb2; font-size:14px; margin-top:0;'>
    Drugs grouped by therapeutic class · RxNorm / RxClass · NLM
    </p>
    """, unsafe_allow_html=True)

    if drug_classes is None:
        st.warning(
            "**`data/drug_classes.csv` not found.** Run the enrichment script first:\n\n"
            "```\npython scripts/enrich_drug_classes.py\n```"
        )
        st.stop()

    # ── Merge spend data ──────────────────────────────────────────────────────
    # Rename before merge to avoid pandas auto-suffixing Gnrc_Name → Gnrc_Name_x
    dc = drug_classes.rename(columns={"Gnrc_Name": "Gnrc_Name_dc"})
    spend_cols = ["Brnd_Name", "Gnrc_Name", "Tot_Spndng_2023", "Tot_Clms_2023", "Tot_Benes_2023"]
    spend_2023 = df_overall[spend_cols].copy()
    merged = dc.merge(spend_2023, on="Brnd_Name", how="left")
    merged["Tot_Spndng_2023"] = pd.to_numeric(merged["Tot_Spndng_2023"], errors="coerce")
    merged["Tot_Clms_2023"]   = pd.to_numeric(merged["Tot_Clms_2023"],   errors="coerce")
    merged["Tot_Benes_2023"]  = pd.to_numeric(merged["Tot_Benes_2023"],  errors="coerce")

    use_atc     = class_system == "ATC (Mechanism)"
    class_col   = "ATC_Class"  if use_atc else "MESH_Class"
    class_label = "ATC Class"  if use_atc else "MeSH Disease Category"

    # ── Sidebar drill-down ────────────────────────────────────────────────────
    with st.sidebar:
        if use_atc:
            groups = sorted(merged["ATC_L1"].dropna().unique())
            selected_group = st.selectbox("Organ system", ["All"] + groups)
            class_options = sorted(
                merged[merged["ATC_L1"] == selected_group][class_col].dropna().unique()
            ) if selected_group != "All" else sorted(merged[class_col].dropna().unique())
        else:
            selected_group = "All"
            class_options = sorted(merged[class_col].dropna().unique())

        selected_class = st.selectbox(class_label, ["All"] + class_options)
        st.markdown("---")
        search_q = st.text_input("Search within results", placeholder="e.g. Ozempic...")

    # ── Filter ────────────────────────────────────────────────────────────────
    view = merged.copy()
    if use_atc and selected_group != "All":
        view = view[view["ATC_L1"] == selected_group]
    if selected_class != "All":
        view = view[view[class_col] == selected_class]
    if search_q:
        view = view[
            view["Brnd_Name"].str.contains(search_q, case=False, na=False) |
            view["Gnrc_Name_dc"].str.contains(search_q, case=False, na=False)
        ]
    view = view.dropna(subset=[class_col]).reset_index(drop=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val in [
        (k1, "Drugs in Selection",  f"{view['Brnd_Name'].nunique():,}"),
        (k2, "2023 Total Spend",    f"${view['Tot_Spndng_2023'].sum()/1e9:,.1f}B"),
        (k3, "2023 Total Claims",   f"{view['Tot_Clms_2023'].sum()/1e6:,.1f}M"),
        (k4, "2023 Beneficiaries",  f"{view['Tot_Benes_2023'].sum()/1e6:,.1f}M"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Top classes bar chart (when no class selected) ────────────────────────
    if selected_class == "All" and not search_q:
        st.markdown('<div class="section-header">💰 Top Classes by 2023 Spend</div>', unsafe_allow_html=True)
        class_spend = (
            view.groupby(class_col)["Tot_Spndng_2023"].sum()
            .reset_index().sort_values("Tot_Spndng_2023", ascending=True).tail(20)
        )
        fig_cls = px.bar(class_spend, x="Tot_Spndng_2023", y=class_col, orientation="h",
            color="Tot_Spndng_2023", color_continuous_scale=[[0, "#1e2a40"], [1, "#4e9af1"]],
            labels={"Tot_Spndng_2023": "2023 Total Spend ($)", class_col: ""})
        fig_cls.update_layout(height=520, **PLOTLY_THEME, margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False)
        st.plotly_chart(fig_cls, use_container_width=True)

        st.markdown('<div class="section-header">🏆 Top 3 Drugs per Class</div>', unsafe_allow_html=True)
        top_per_class = (
            view.sort_values("Tot_Spndng_2023", ascending=False)
            .groupby(class_col).head(3)
            [[class_col, "Brnd_Name", "Gnrc_Name_dc", "Tot_Spndng_2023", "Tot_Clms_2023"]]
            .rename(columns={"Gnrc_Name_dc": "Generic Name", "Tot_Spndng_2023": "2023 Spend ($)",
                              "Tot_Clms_2023": "2023 Claims", class_col: class_label})
        )
        st.dataframe(
            top_per_class.style
                .format({"2023 Spend ($)": "${:,.0f}", "2023 Claims": "{:,.0f}"})
                .hide(axis="index"),
            use_container_width=True, height=500,
        )

    # ── Drug cards (when class or search is active) ───────────────────────────
    else:
        st.markdown('<div class="section-header">💊 Drugs in Selection</div>', unsafe_allow_html=True)
        view_sorted = view.sort_values("Tot_Spndng_2023", ascending=False)

        page_size = 20
        total_pages = max(1, (len(view_sorted) - 1) // page_size + 1)
        card_page = st.number_input("Page", min_value=1, max_value=total_pages,
                                    value=1, step=1, label_visibility="collapsed") if total_pages > 1 else 1
        start = (card_page - 1) * page_size
        page_df = view_sorted.iloc[start: start + page_size]

        st.markdown(
            f"<p style='color:#7b8bb2; font-size:13px;'>Showing {len(view_sorted):,} drugs · sorted by 2023 spend</p>",
            unsafe_allow_html=True)

        for _, row in page_df.iterrows():
            spend_str  = f"${row['Tot_Spndng_2023']/1e6:,.1f}M" if pd.notna(row["Tot_Spndng_2023"]) else "N/A"
            claims_str = f"{row['Tot_Clms_2023']:,.0f}"          if pd.notna(row["Tot_Clms_2023"])   else "N/A"
            benes_str  = f"{row['Tot_Benes_2023']:,.0f}"         if pd.notna(row["Tot_Benes_2023"])  else "N/A"
            class_tag  = row.get(class_col, "") or ""
            atc_l1_tag = f" · {row['ATC_L1']}" if use_atc and pd.notna(row.get("ATC_L1")) else ""

            st.markdown(f"""
            <div class="drug-card">
                <div class="drug-card-title">{row["Brnd_Name"]}</div>
                <div class="drug-card-generic">{row.get("Gnrc_Name_dc", "")} &nbsp;·&nbsp;
                    <span style="color:#4e9af1; font-size:12px;">{class_tag}{atc_l1_tag}</span>
                </div>
                <div style="display:flex; gap:32px; margin-top:10px;">
                    <div>
                        <div style="font-size:10px; color:#7b8bb2; text-transform:uppercase; letter-spacing:0.1em;">2023 Spend</div>
                        <div style="font-size:16px; font-weight:600; color:#e8eaf0;">{spend_str}</div>
                    </div>
                    <div>
                        <div style="font-size:10px; color:#7b8bb2; text-transform:uppercase; letter-spacing:0.1em;">Claims</div>
                        <div style="font-size:16px; font-weight:600; color:#e8eaf0;">{claims_str}</div>
                    </div>
                    <div>
                        <div style="font-size:10px; color:#7b8bb2; text-transform:uppercase; letter-spacing:0.1em;">Beneficiaries</div>
                        <div style="font-size:16px; font-weight:600; color:#e8eaf0;">{benes_str}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if total_pages > 1:
            st.markdown(
                f"<p style='color:#7b8bb2; font-size:12px; text-align:center;'>Page {card_page} of {total_pages}</p>",
                unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:12px;'>"
    "Medicare Part D Drug Spending · CMS Data 2019–2023 · Built with Streamlit & Plotly</p>",
    unsafe_allow_html=True,
)
