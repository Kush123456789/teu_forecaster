"""
📊 Dashboard — KPIs, Forecast Overlay, Prediction Intervals
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
css = Path(__file__).parent.parent / "assets" / "style.css"
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

from core.data_pipeline import load_actual_teu, backcast, load_exogenous, build_model_df, split
from core.model_engine import (
    fit_arima, forecast_arima,
    fit_sarima, forecast_sarima,
    fit_sarimax, forecast_sarimax,
    fit_egarch, forecast_hybrid,
    calc_metrics, coverage,
)
from core.config import EXO_LAG_COLS, DATA_DIR, N_FORECAST

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=12, color="#94A3B8"),
    title_font=dict(family="Inter, sans-serif", size=16, color="#F8FAFC"),
    xaxis=dict(showgrid=True, gridcolor="#1E293B", linecolor="#334155", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#1E293B", linecolor="#334155", zeroline=False),
    hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_color="#F8FAFC", bordercolor="#334155"),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=40, r=20, t=50, b=70),
)
COLORS = ["#38BDF8", "#818CF8", "#34D399", "#FB7185", "#FBBF24", "#A78BFA"]


@st.cache_data(show_spinner="Loading data …")
def load_data():
    teu_path = DATA_DIR / "KPA MOMBASA PORT - 5YR Summary.xlsx"
    exo_path = DATA_DIR / "Exogenous Variables.xlsx"
    actual = load_actual_teu(str(teu_path))
    full = backcast(actual)
    exo = load_exogenous(str(exo_path))
    df = build_model_df(full, exo)
    train, test = split(df)
    return actual, full, exo, df, train, test


@st.cache_data(show_spinner="Fitting models …")
def run_models(_train, _test):
    train, test = _train, _test
    arima_m = fit_arima(train["TEU"])
    arima_fc = forecast_arima(arima_m, N_FORECAST)
    sarima_m = fit_sarima(train["TEU"])
    sarima_fc = forecast_sarima(sarima_m, N_FORECAST)
    sarimax_m = fit_sarimax(train["TEU"], train[EXO_LAG_COLS])
    sarimax_fc = forecast_sarimax(sarimax_m, N_FORECAST, test[EXO_LAG_COLS])
    egarch_res, resid = fit_egarch(sarimax_m, train["TEU"], train[EXO_LAG_COLS])
    hybrid = forecast_hybrid(sarimax_m, egarch_res, N_FORECAST, test[EXO_LAG_COLS])

    forecasts = {
        "ARIMA": arima_fc,
        "SARIMA": sarima_fc,
        "SARIMAX": sarimax_fc,
        "Hybrid": hybrid["point"],
    }
    metrics = {n: calc_metrics(test["TEU"].values, f) for n, f in forecasts.items()}
    return forecasts, metrics, hybrid, egarch_res


actual, full, exo, df, train, test = load_data()
forecasts, metrics, hybrid, egarch_res = run_models(train, test)

# ── Page Title ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<h1 style='margin:0'>📊 Dashboard</h1>"
    "</div>",
    unsafe_allow_html=True,
)

# ── KPI Row ──────────────────────────────────────────────────────────────
best_mape = metrics["Hybrid"]["MAPE"]
cov95 = coverage(test["TEU"], hybrid["lower_95"], hybrid["upper_95"])
cov80 = coverage(test["TEU"], hybrid["lower_80"], hybrid["upper_80"])
arima_mape = metrics["ARIMA"]["MAPE"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best MAPE", f"{best_mape:.2f}%",
           delta=f"-{arima_mape - best_mape:.1f}% vs ARIMA", delta_color="normal")
k2.metric("95% Coverage", f"{cov95:.0f}%",
           delta="Target: ≥95%", delta_color="off")
k3.metric("80% Coverage", f"{cov80:.0f}%",
           delta="Target: ≥80%", delta_color="off")
k4.metric("Observations", f"{len(df):,}",
           delta=f"{N_FORECAST} forecast horizon", delta_color="off")

st.markdown("---")

# ── Forecast Overlay ─────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.1rem'>Forecast Overlay — All Models vs Actuals</span>"
    "</div>",
    unsafe_allow_html=True,
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train["TEU"], name="Train",
                         line=dict(color="#475569", width=1),
                         hovertemplate="Train<br>%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>"))
fig.add_trace(go.Scatter(x=test.index, y=test["TEU"], name="Actual",
                         line=dict(color="#F8FAFC", width=2.5),
                         hovertemplate="Actual<br>%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>"))
for i, (name, fc) in enumerate(forecasts.items()):
    fc_arr = fc.values if hasattr(fc, "values") else fc
    fig.add_trace(go.Scatter(
        x=test.index, y=fc_arr, name=name,
        line=dict(color=COLORS[i], dash="dash", width=2),
        hovertemplate=f"{name}<br>" + "%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>",
    ))
fig.update_layout(title="Multi-Model Forecast vs Actuals (2025)", **PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# ── Prediction Intervals ────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.1rem'>SARIMAX-EGARCH Prediction Intervals</span>"
    "</div>",
    unsafe_allow_html=True,
)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=test.index, y=test["TEU"], name="Actual",
                          line=dict(color="#F8FAFC", width=2.5),
                          hovertemplate="Actual<br>%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>"))
fig2.add_trace(go.Scatter(x=test.index, y=hybrid["point"], name="Point Forecast",
                          line=dict(color="#38BDF8", dash="dash", width=2),
                          hovertemplate="Forecast<br>%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>"))
# 95% band
fig2.add_trace(go.Scatter(x=test.index, y=hybrid["upper_95"], name="95% Upper",
                          line=dict(color="#FB7185", dash="dot", width=1), showlegend=False))
fig2.add_trace(go.Scatter(x=test.index, y=hybrid["lower_95"], name="95% CI",
                          fill="tonexty", fillcolor="rgba(251,113,133,0.08)",
                          line=dict(color="#FB7185", dash="dot", width=1)))
# 80% band
fig2.add_trace(go.Scatter(x=test.index, y=hybrid["upper_80"], name="80% Upper",
                          line=dict(color="#FBBF24", dash="dot", width=1), showlegend=False))
fig2.add_trace(go.Scatter(x=test.index, y=hybrid["lower_80"], name="80% CI",
                          fill="tonexty", fillcolor="rgba(251,191,36,0.06)",
                          line=dict(color="#FBBF24", dash="dot", width=1)))
fig2.update_layout(title="Hybrid SARIMAX-EGARCH: Prediction Intervals", **PLOTLY_LAYOUT)
st.plotly_chart(fig2, use_container_width=True)

# ── Metrics Table ────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.1rem'>Point Forecast Metrics</span>"
    "</div>",
    unsafe_allow_html=True,
)

rows = []
best_model = min(metrics, key=lambda n: metrics[n]["MAPE"])
for name, m in metrics.items():
    rows.append({
        "Model": ("🏆 " + name) if name == best_model else name,
        "MAE": f"{m['MAE']:,.0f}",
        "RMSE": f"{m['RMSE']:,.0f}",
        "MAPE (%)": f"{m['MAPE']:.2f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Conditional Volatility ───────────────────────────────────────────────
with st.expander("⚡ EGARCH Conditional Volatility"):
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=train.index, y=egarch_res.conditional_volatility,
        name="Cond. Volatility",
        line=dict(color="#FBBF24", width=1.5),
        fill="tozeroy", fillcolor="rgba(251,191,36,0.06)",
        hovertemplate="%{x|%b %Y}<br>Vol: %{y:,.0f}<extra></extra>",
    ))
    fig3.update_layout(title="EGARCH(1,1) Conditional Volatility", **PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Conditional volatility captures time-varying uncertainty in the SARIMAX residuals.")
