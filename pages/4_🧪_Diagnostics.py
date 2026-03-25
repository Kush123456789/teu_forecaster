"""
🧪 Diagnostics — Residual Analysis, Ljung-Box, Jarque-Bera, Diebold-Mariano
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.stats import probplot
from pathlib import Path

st.set_page_config(page_title="Diagnostics", page_icon="🧪", layout="wide")
css = Path(__file__).parent.parent / "assets" / "style.css"
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

from core.data_pipeline import load_actual_teu, backcast, load_exogenous, build_model_df, split
from core.model_engine import (
    fit_arima, forecast_arima,
    fit_sarima, forecast_sarima,
    fit_sarimax, forecast_sarimax,
    fit_egarch,
    calc_metrics, residual_diagnostics, diebold_mariano,
)
from core.config import EXO_LAG_COLS, DATA_DIR, N_FORECAST

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=12, color="#94A3B8"),
    title_font=dict(family="Inter, sans-serif", size=16, color="#F8FAFC"),
    xaxis=dict(showgrid=True, gridcolor="#1E293B", linecolor="#334155", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#1E293B", linecolor="#334155", zeroline=False),
    hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#F8FAFC", bordercolor="#334155"),
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
    margin=dict(l=40, r=20, t=50, b=80),
)

def section(title):
    st.markdown(
        f"<div class='section-header'><div class='accent-dot'></div>"
        f"<span style='font-family:Inter,sans-serif;font-weight:600;"
        f"color:#F8FAFC;font-size:1rem'>{title}</span></div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading …")
def load():
    actual = load_actual_teu(str(DATA_DIR / "KPA MOMBASA PORT - 5YR Summary.xlsx"))
    full = backcast(actual)
    exo = load_exogenous(str(DATA_DIR / "Exogenous Variables.xlsx"))
    df = build_model_df(full, exo)
    train, test = split(df)
    return df, train, test


@st.cache_data(show_spinner="Fitting …")
def fit_all(_train, _test):
    train, test = _train, _test
    arima_m = fit_arima(train["TEU"])
    sarima_m = fit_sarima(train["TEU"])
    sarimax_m = fit_sarimax(train["TEU"], train[EXO_LAG_COLS])

    forecasts = {
        "ARIMA": forecast_arima(arima_m, N_FORECAST),
        "SARIMA": forecast_sarima(sarima_m, N_FORECAST),
        "SARIMAX": forecast_sarimax(sarimax_m, N_FORECAST, test[EXO_LAG_COLS]),
    }
    diag = residual_diagnostics(sarimax_m, train["TEU"], train[EXO_LAG_COLS])
    return forecasts, diag


df, train, test = load()
forecasts, diag = fit_all(train, test)

st.title("🧪 Residual Diagnostics")

# ── Quick Stats Row ──────────────────────────────────────────────────────
resid = diag["residuals"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(resid)}")
c2.metric("Mean Residual", f"{resid.mean():,.0f}")
c3.metric("Std Dev", f"{resid.std():,.0f}")
c4.metric("Range", f"{resid.max() - resid.min():,.0f}")

# ── Residual Time Series ─────────────────────────────────────────────────
section("SARIMAX Residual Plot")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=resid.index, y=resid, name="Residuals",
    line=dict(color="#818CF8", width=1),
    fill="tozeroy", fillcolor="rgba(129,140,248,0.08)",
    hovertemplate="%{x|%b %Y}<br>Residual: %{y:,.0f}<extra></extra>",
))
fig.add_hline(y=0, line_dash="dash", line_color="#FB7185", opacity=0.6)
fig.update_layout(title="SARIMAX In-Sample Residuals", **PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# ── Residual Histogram + Q-Q side by side ────────────────────────────────
section("Residual Distribution & Q-Q Plot")
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Histogram of Residuals", "Normal Q-Q Plot"))

fig2.add_trace(go.Histogram(
    x=resid, nbinsx=30, marker_color="#34D399", opacity=0.7, name="Residuals",
    hovertemplate="Bin: %{x:,.0f}<br>Count: %{y}<extra></extra>",
), row=1, col=1)

qq = probplot(resid.dropna(), dist="norm")
fig2.add_trace(go.Scatter(
    x=qq[0][0], y=qq[0][1], mode="markers", name="Residuals",
    marker=dict(color="#38BDF8", size=5, opacity=0.8),
    hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:,.0f}<extra></extra>",
), row=1, col=2)
fig2.add_trace(go.Scatter(
    x=[qq[0][0].min(), qq[0][0].max()],
    y=[qq[1][1] + qq[1][0] * qq[0][0].min(), qq[1][1] + qq[1][0] * qq[0][0].max()],
    name="Normal Line", line=dict(color="#FB7185", dash="dash"), showlegend=False,
), row=1, col=2)
fig2.update_xaxes(title_text="Residual Value", row=1, col=1)
fig2.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
fig2.update_yaxes(title_text="Frequency", row=1, col=1)
fig2.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
fig2.update_layout(**PLOTLY_LAYOUT, showlegend=True)
for ann in fig2.layout.annotations:
    ann.update(font=dict(family="Inter, sans-serif", size=13, color="#94A3B8"))
st.plotly_chart(fig2, use_container_width=True)
st.caption("Left: distribution of SARIMAX residuals. Right: Q-Q plot against the normal distribution — "
           "deviations at tails indicate heavy tails / non-normality.")

# ── Ljung-Box ───────────────────────────────────────────────────────────
section("Ljung-Box Test — Autocorrelation")
lb = diag["ljung_box"]
lb_display = lb.copy()
lb_display.index.name = "Lag"
lb_display = lb_display.reset_index()
lb_pass = (lb["lb_pvalue"] > 0.05).all()
st.dataframe(lb_display.style.format({"lb_stat": "{:.2f}", "lb_pvalue": "{:.4f}"}),
             use_container_width=True, hide_index=True)
badge_cls = "badge-pass" if lb_pass else "badge-fail"
badge_txt = "PASS — No significant autocorrelation" if lb_pass else "WARN — Autocorrelation detected"
st.markdown(f"<span class='{badge_cls}'>{badge_txt}</span>", unsafe_allow_html=True)

# ── Jarque-Bera ──────────────────────────────────────────────────────────
section("Jarque-Bera Test — Normality")
jb1, jb2, jb3, jb4 = st.columns(4)
jb1.metric("JB Statistic", f"{diag['jb_stat']:.2f}")
jb2.metric("p-value", f"{diag['jb_pval']:.4f}")
jb3.metric("Skewness", f"{diag['skewness']:.2f}")
jb4.metric("Kurtosis", f"{diag['kurtosis']:.2f}")
jb_pass = diag["jb_pval"] >= 0.05
badge_cls = "badge-pass" if jb_pass else "badge-warn"
badge_txt = ("PASS — Residuals approximately normal" if jb_pass
             else "REJECTED — Heavy tails detected → motivates EGARCH volatility overlay")
st.markdown(f"<span class='{badge_cls}'>{badge_txt}</span>", unsafe_allow_html=True)

# ── Diebold-Mariano ─────────────────────────────────────────────────────
section("Diebold-Mariano Test — Forecast Comparison")
names = list(forecasts.keys())
dm_rows = []
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        if i < j:
            p1 = forecasts[n1].values if hasattr(forecasts[n1], "values") else forecasts[n1]
            p2 = forecasts[n2].values if hasattr(forecasts[n2], "values") else forecasts[n2]
            dm = diebold_mariano(test["TEU"].values, p1, p2)
            if abs(dm) > 1.96:
                sig = "✅ Yes"
            elif abs(dm) > 1.65:
                sig = "⚠️ Near"
            else:
                sig = "❌ No"
            dm_rows.append({"Model A": n1, "Model B": n2, "DM Statistic": f"{dm:.3f}",
                            "Significant (5%)": sig})
st.dataframe(pd.DataFrame(dm_rows), use_container_width=True, hide_index=True)
st.caption("DM > 0 → Model B more accurate; |DM| > 1.96 → significant at 5%.")
