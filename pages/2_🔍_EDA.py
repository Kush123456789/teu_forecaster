"""
🔍 EDA — Time Series Plot, Descriptive Stats, Seasonality, Correlation, Stationarity
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from pathlib import Path

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")
css = Path(__file__).parent.parent / "assets" / "style.css"
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

from core.data_pipeline import load_actual_teu, backcast, load_exogenous, build_model_df
from core.config import M_SEASONAL, DATA_DIR

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=12, color="#94A3B8"),
    title_font=dict(family="Inter, sans-serif", size=16, color="#F8FAFC"),
    xaxis=dict(showgrid=True, gridcolor="#1E293B", linecolor="#334155", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#1E293B", linecolor="#334155", zeroline=False),
    hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#F8FAFC", bordercolor="#334155"),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=40, r=20, t=50, b=70),
)


@st.cache_data(show_spinner="Loading data …")
def load():
    actual = load_actual_teu(str(DATA_DIR / "KPA MOMBASA PORT - 5YR Summary.xlsx"))
    full = backcast(actual)
    exo = load_exogenous(str(DATA_DIR / "Exogenous Variables.xlsx"))
    df = build_model_df(full, exo)
    return actual, full, exo, df


actual, full, exo, df = load()

# ── Page Title ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<h1 style='margin:0'>🔍 Exploratory Data Analysis</h1>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Quick Stats ──────────────────────────────────────────────────────────
qs1, qs2, qs3, qs4 = st.columns(4)
qs1.metric("Observations", f"{len(full):,}")
qs2.metric("Mean TEU", f"{full.mean():,.0f}")
qs3.metric("Std Dev", f"{full.std():,.0f}")
qs4.metric("Period", f"{M_SEASONAL}-month cycle", delta="Semi-annual seasonality", delta_color="off")

st.markdown("---")

# ── Time Series Plot ─────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Monthly TEU Throughput</span>"
    "</div>",
    unsafe_allow_html=True,
)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=full.index, y=full.values, name="TEU",
    line=dict(color="#38BDF8", width=2),
    fill="tozeroy", fillcolor="rgba(56,189,248,0.05)",
    hovertemplate="%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>",
))
# rolling mean overlay
rolling = full.rolling(6).mean()
fig.add_trace(go.Scatter(
    x=full.index, y=rolling.values, name="6-month MA",
    line=dict(color="#FBBF24", width=1.5, dash="dot"),
    hovertemplate="%{x|%b %Y}<br>MA: %{y:,.0f}<extra></extra>",
))
fig.update_layout(title="Mombasa Port — Monthly Import TEU (2015–2025)",
                  yaxis_title="TEU", **PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# ── Descriptive Stats ────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Descriptive Statistics</span>"
    "</div>",
    unsafe_allow_html=True,
)
desc = full.describe().to_frame("TEU").T
desc["Skewness"] = full.skew()
desc["Kurtosis"] = full.kurtosis()
st.dataframe(desc.style.format("{:,.2f}"), use_container_width=True)

# ── Boxplots by Month ───────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Seasonal Box Plots</span>"
    "</div>",
    unsafe_allow_html=True,
)
box_df = pd.DataFrame({"TEU": full.values, "Month": full.index.month_name()}, index=full.index)
fig_box = px.box(box_df, x="Month", y="TEU", color_discrete_sequence=["#818CF8"])
fig_box.update_layout(title="TEU Distribution by Calendar Month",
                      yaxis_title="TEU", **PLOTLY_LAYOUT)
fig_box.update_traces(marker=dict(opacity=0.7))
st.plotly_chart(fig_box, use_container_width=True)

# ── Correlation Heatmap ──────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Correlation Matrix</span>"
    "</div>",
    unsafe_allow_html=True,
)
corr_df = df.corr()
fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto")
fig_corr.update_layout(title="Pearson Correlation — TEU & Exogenous", **PLOTLY_LAYOUT)
st.plotly_chart(fig_corr, use_container_width=True)

# ── Seasonal Decomposition ──────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Seasonal Decomposition (Additive, m=6)</span>"
    "</div>",
    unsafe_allow_html=True,
)
decomp = seasonal_decompose(full, model="additive", period=M_SEASONAL)

# Stacked subplots
from plotly.subplots import make_subplots
fig_decomp = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
)
components = [
    (full, "#38BDF8"),
    (decomp.trend, "#34D399"),
    (decomp.seasonal, "#FBBF24"),
    (decomp.resid, "#FB7185"),
]
for i, (comp, color) in enumerate(components, 1):
    fig_decomp.add_trace(
        go.Scatter(x=full.index, y=comp, name=["Observed","Trend","Seasonal","Residual"][i-1],
                   line=dict(color=color, width=1.5), showlegend=False),
        row=i, col=1,
    )
fig_decomp.update_layout(
    height=600,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=11, color="#94A3B8"),
    margin=dict(l=40, r=20, t=40, b=30),
)
for i in range(1, 5):
    fig_decomp.update_xaxes(showgrid=True, gridcolor="#1E293B", linecolor="#334155", row=i, col=1)
    fig_decomp.update_yaxes(showgrid=True, gridcolor="#1E293B", linecolor="#334155", row=i, col=1)
st.plotly_chart(fig_decomp, use_container_width=True)

# ── ADF Test ─────────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Stationarity — Augmented Dickey-Fuller Test</span>"
    "</div>",
    unsafe_allow_html=True,
)
c1, c2 = st.columns(2)
with c1:
    adf_raw = adfuller(full.dropna())
    raw_stationary = adf_raw[1] < 0.05
    badge_raw = "badge-pass" if raw_stationary else "badge-fail"
    verdict_raw = "Stationary" if raw_stationary else "Non-stationary"
    st.markdown(f"**Raw Series** <span class='badge {badge_raw}'>{verdict_raw}</span>",
                unsafe_allow_html=True)
    st.metric("ADF Statistic", f"{adf_raw[0]:.4f}")
    st.metric("p-value", f"{adf_raw[1]:.4f}")
with c2:
    diff1 = full.diff().dropna()
    adf_d = adfuller(diff1)
    d_stationary = adf_d[1] < 0.05
    badge_d = "badge-pass" if d_stationary else "badge-fail"
    verdict_d = "Stationary" if d_stationary else "Non-stationary"
    st.markdown(f"**First-Differenced** <span class='badge {badge_d}'>{verdict_d}</span>",
                unsafe_allow_html=True)
    st.metric("ADF Statistic", f"{adf_d[0]:.4f}")
    st.metric("p-value", f"{adf_d[1]:.6f}")

# ── ACF / PACF ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>ACF & PACF (first-differenced)</span>"
    "</div>",
    unsafe_allow_html=True,
)
nlags = min(40, len(diff1) // 2 - 1)
acf_vals = acf(diff1, nlags=nlags)
pacf_vals = pacf(diff1, nlags=nlags)
ci = 1.96 / np.sqrt(len(diff1))

fig_correl = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"],
                           horizontal_spacing=0.08)
fig_correl.add_trace(
    go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color="#38BDF8",
           width=0.3, showlegend=False),
    row=1, col=1,
)
fig_correl.add_trace(
    go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color="#818CF8",
           width=0.3, showlegend=False),
    row=1, col=2,
)
for col in [1, 2]:
    fig_correl.add_hline(y=ci, line_dash="dot", line_color="#FB7185", row=1, col=col)
    fig_correl.add_hline(y=-ci, line_dash="dot", line_color="#FB7185", row=1, col=col)
fig_correl.update_layout(
    height=350,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=11, color="#94A3B8"),
    margin=dict(l=40, r=20, t=40, b=30),
)
for col in [1, 2]:
    fig_correl.update_xaxes(showgrid=True, gridcolor="#1E293B", linecolor="#334155", row=1, col=col)
    fig_correl.update_yaxes(showgrid=True, gridcolor="#1E293B", linecolor="#334155", row=1, col=col)
st.plotly_chart(fig_correl, use_container_width=True)
st.caption("Red dashed lines = 95% confidence bounds. Significant lags exceed the bounds.")
