"""
📈 Models — Model Comparison, Grid Search, Cross-Validation
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Models", page_icon="📈", layout="wide")
css = Path(__file__).parent.parent / "assets" / "style.css"
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

from core.data_pipeline import load_actual_teu, backcast, load_exogenous, build_model_df, split
from core.model_engine import (
    fit_arima, forecast_arima,
    fit_sarima, forecast_sarima,
    fit_sarimax, forecast_sarimax,
    fit_egarch, forecast_hybrid,
    calc_metrics, grid_search_sarimax, time_series_cv,
)
from core.config import (
    EXO_LAG_COLS, DATA_DIR, N_FORECAST,
    ARIMA_ORDER, SARIMA_ORDER, SARIMA_SEASONAL,
    SARIMAX_ORDER, SARIMAX_SEASONAL, M_SEASONAL,
)

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
COLORS = ["#38BDF8", "#818CF8", "#34D399", "#FB7185"]
MODEL_COLORS = {"ARIMA": "#38BDF8", "SARIMA": "#818CF8", "SARIMAX": "#34D399", "Hybrid": "#FB7185"}


@st.cache_data(show_spinner="Loading …")
def load():
    actual = load_actual_teu(str(DATA_DIR / "KPA MOMBASA PORT - 5YR Summary.xlsx"))
    full = backcast(actual)
    exo = load_exogenous(str(DATA_DIR / "Exogenous Variables.xlsx"))
    df = build_model_df(full, exo)
    train, test = split(df)
    return df, train, test


@st.cache_data(show_spinner="Fitting models …")
def fit_all(_train, _test):
    train, test = _train, _test
    arima_m = fit_arima(train["TEU"])
    sarima_m = fit_sarima(train["TEU"])
    sarimax_m = fit_sarimax(train["TEU"], train[EXO_LAG_COLS])
    egarch_res, _ = fit_egarch(sarimax_m, train["TEU"], train[EXO_LAG_COLS])

    forecasts = {
        "ARIMA": forecast_arima(arima_m, N_FORECAST),
        "SARIMA": forecast_sarima(sarima_m, N_FORECAST),
        "SARIMAX": forecast_sarimax(sarimax_m, N_FORECAST, test[EXO_LAG_COLS]),
    }
    hybrid = forecast_hybrid(sarimax_m, egarch_res, N_FORECAST, test[EXO_LAG_COLS])
    forecasts["Hybrid"] = hybrid["point"]
    metrics = {n: calc_metrics(test["TEU"].values, f) for n, f in forecasts.items()}
    return forecasts, metrics


df, train, test = load()
forecasts, metrics = fit_all(train, test)

best_model = min(metrics, key=lambda n: metrics[n]["MAPE"])

# ── Page Title ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<h1 style='margin:0'>📈 Model Comparison</h1>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Winner Banner ────────────────────────────────────────────────────────
st.markdown(
    f"""<div class='winner-card'>
        <span style='font-size:1.3rem'>🏆</span>
        <span style='font-family:Inter,sans-serif; font-weight:700; color:#F8FAFC;
                     font-size:1rem; margin-left:0.5rem'>
            Best Model: {best_model}
        </span>
        <span style='font-family:JetBrains Mono,monospace; color:#34D399;
                     font-size:0.85rem; margin-left:0.8rem'>
            MAPE {metrics[best_model]["MAPE"]:.2f}%
        </span>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown("")

# ── Specification Table ──────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Model Specifications</span>"
    "</div>",
    unsafe_allow_html=True,
)
spec_data = [
    {"Model": "ARIMA", "Order": str(ARIMA_ORDER), "Seasonal": "—", "Exogenous": "No"},
    {"Model": "SARIMA", "Order": str(SARIMA_ORDER), "Seasonal": str(SARIMA_SEASONAL), "Exogenous": "No"},
    {"Model": "SARIMAX", "Order": str(SARIMAX_ORDER), "Seasonal": str(SARIMAX_SEASONAL), "Exogenous": "Yes (lag-1)"},
    {"Model": "Hybrid", "Order": str(SARIMAX_ORDER), "Seasonal": str(SARIMAX_SEASONAL), "Exogenous": "Yes + EGARCH(1,1)"},
]
st.dataframe(pd.DataFrame(spec_data), use_container_width=True, hide_index=True)

# ── Individual Forecast Plots ────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>Individual Model Forecasts</span>"
    "</div>",
    unsafe_allow_html=True,
)
selected = st.selectbox("Select model", list(forecasts.keys()))
fc = forecasts[selected]
fc_arr = fc.values if hasattr(fc, "values") else fc

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=train.index, y=train["TEU"], name="Train",
    line=dict(color="#475569", width=1),
    hovertemplate="Train<br>%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=test.index, y=test["TEU"], name="Actual",
    line=dict(color="#F8FAFC", width=2.5),
    hovertemplate="Actual<br>%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=test.index, y=fc_arr, name=f"{selected} Forecast",
    line=dict(color=MODEL_COLORS.get(selected, "#38BDF8"), dash="dash", width=2),
    hovertemplate=f"{selected}<br>" + "%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>",
))
fig.update_layout(title=f"{selected}: Forecast vs Actuals", **PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

m = metrics[selected]
mc1, mc2, mc3 = st.columns(3)
mc1.metric("MAE", f"{m['MAE']:,.0f} TEU")
mc2.metric("RMSE", f"{m['RMSE']:,.0f} TEU")
mc3.metric("MAPE", f"{m['MAPE']:.2f}%",
           delta="Best" if selected == best_model else f"+{m['MAPE'] - metrics[best_model]['MAPE']:.2f}% vs best",
           delta_color="off" if selected == best_model else "inverse")

# ── MAPE Bar Chart ───────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1.05rem'>MAPE Comparison</span>"
    "</div>",
    unsafe_allow_html=True,
)
mape_df = pd.DataFrame([{"Model": n, "MAPE": m["MAPE"]} for n, m in metrics.items()])
fig_bar = go.Figure()
bar_colors = [MODEL_COLORS.get(n, "#38BDF8") for n in mape_df["Model"]]
fig_bar.add_trace(go.Bar(
    x=mape_df["Model"], y=mape_df["MAPE"],
    marker_color=bar_colors,
    text=[f"{v:.2f}%" for v in mape_df["MAPE"]], textposition="outside",
    textfont=dict(color="#F8FAFC", size=13, family="JetBrains Mono"),
    hovertemplate="%{x}<br>MAPE: %{y:.2f}%<extra></extra>",
))
fig_bar.update_layout(title="Test-Set MAPE by Model", yaxis_title="MAPE (%)", **PLOTLY_LAYOUT)
st.plotly_chart(fig_bar, use_container_width=True)

# ── All Models Overlay ───────────────────────────────────────────────────
with st.expander("📊 All Models — Forecast Overlay"):
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=test.index, y=test["TEU"], name="Actual",
        line=dict(color="#F8FAFC", width=2.5),
        hovertemplate="Actual<br>%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>",
    ))
    for name, fc in forecasts.items():
        fc_arr = fc.values if hasattr(fc, "values") else fc
        fig_all.add_trace(go.Scatter(
            x=test.index, y=fc_arr, name=name,
            line=dict(color=MODEL_COLORS.get(name, "#38BDF8"), dash="dash", width=2),
            hovertemplate=f"{name}<br>" + "%{x|%b %Y}<br>TEU: %{y:,.0f}<extra></extra>",
        ))
    fig_all.update_layout(title="All Models — Forecast Overlay (Test Period)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_all, use_container_width=True)

# ── Grid Search ──────────────────────────────────────────────────────────
with st.expander("⚙️ SARIMAX Grid Search (36 configurations)"):
    with st.spinner("Running grid search …"):
        grid = grid_search_sarimax(
            train["TEU"], train[EXO_LAG_COLS],
            test["TEU"], test[EXO_LAG_COLS],
        )
    st.dataframe(grid.head(10).style.format({"Test_MAPE": "{:.2f}", "AIC": "{:.0f}"}),
                 use_container_width=True, hide_index=True)
    best = grid.iloc[0]

    # Grid search MAPE chart
    top10 = grid.head(10).copy()
    top10["label"] = top10["order"].astype(str) + " × " + top10["seasonal"].astype(str)
    fig_grid = go.Figure()
    fig_grid.add_trace(go.Bar(
        x=top10["label"], y=top10["Test_MAPE"],
        marker_color=["#34D399"] + ["#38BDF8"] * (len(top10) - 1),
        text=[f"{v:.2f}%" for v in top10["Test_MAPE"]], textposition="outside",
        textfont=dict(color="#F8FAFC", size=11, family="JetBrains Mono"),
    ))
    fig_grid.update_layout(title="Top 10 Grid Search Results — MAPE",
                           yaxis_title="MAPE (%)", xaxis_tickangle=-45, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_grid, use_container_width=True)

    st.markdown(
        f"<div class='winner-card'>"
        f"Best: <code>{best['order']} × {best['seasonal']}</code> — "
        f"MAPE = <span style='color:#34D399; font-weight:700'>{best['Test_MAPE']:.2f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Cross-Validation ────────────────────────────────────────────────────
with st.expander("📋 Time-Series Cross-Validation (5 folds)"):
    with st.spinner("Cross-validating …"):
        cv = time_series_cv(df)
    st.dataframe(cv.style.format({"CV_MAPE_mean": "{:.2f}", "CV_MAPE_std": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
