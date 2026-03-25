"""
FastAPI backend for the Mombasa Port TEU Forecaster.

Run:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /api/health            — Health check
    GET  /api/models/info       — Model specifications & constants
    GET  /api/data              — Load TEU + exogenous data, train/test split
    GET  /api/forecast          — Fit all models, return forecasts & metrics
    GET  /api/diagnostics       — Residual diagnostics, Ljung-Box, JB, DM tests
    POST /api/grid-search       — Run SARIMAX grid search
    POST /api/cross-validation  — Run expanding-window cross-validation
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    DataResponse, TEUPoint, ExogenousPoint,
    ForecastResponse, ModelMetrics, ForecastPoint, ForecastInterval,
    DiagnosticsResponse, DMTestResult, LjungBoxRow,
    GridSearchResponse, GridSearchRow,
    CVResponse, CVRow,
    ModelsInfoResponse, ModelInfo,
)
from core.config import (
    DATA_DIR, M_SEASONAL, N_FORECAST, TEST_YEAR,
    ARIMA_ORDER, SARIMA_ORDER, SARIMA_SEASONAL,
    SARIMAX_ORDER, SARIMAX_SEASONAL,
    EGARCH_P, EGARCH_Q, EXO_LAG_COLS,
)
from core.data_pipeline import (
    load_actual_teu, backcast, load_exogenous, build_model_df, split,
)
from core.model_engine import (
    fit_arima, forecast_arima,
    fit_sarima, forecast_sarima,
    fit_sarimax, forecast_sarimax,
    fit_egarch, forecast_hybrid,
    calc_metrics, residual_diagnostics, diebold_mariano,
    grid_search_sarimax, time_series_cv,
)

API_VERSION = "1.0.0"

app = FastAPI(
    title="Mombasa Port TEU Forecaster API",
    version=API_VERSION,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Cached data loaders (module-level singletons) ────────────────────────
@functools.lru_cache(maxsize=1)
def _load_data():
    actual = load_actual_teu(str(DATA_DIR / "KPA MOMBASA PORT - 5YR Summary.xlsx"))
    full = backcast(actual)
    exo = load_exogenous(str(DATA_DIR / "Exogenous Variables.xlsx"))
    df = build_model_df(full, exo)
    train, test = split(df)
    return df, train, test, exo


@functools.lru_cache(maxsize=1)
def _fit_models():
    df, train, test, _ = _load_data()
    arima_m = fit_arima(train["TEU"])
    sarima_m = fit_sarima(train["TEU"])
    sarimax_m = fit_sarimax(train["TEU"], train[EXO_LAG_COLS])
    egarch_res, _ = fit_egarch(sarimax_m, train["TEU"], train[EXO_LAG_COLS])
    return arima_m, sarima_m, sarimax_m, egarch_res


# ── Routes ───────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=API_VERSION)


@app.get("/api/models/info", response_model=ModelsInfoResponse)
def models_info():
    return ModelsInfoResponse(
        models=[
            ModelInfo(name="ARIMA", order=str(ARIMA_ORDER),
                      description="Non-seasonal ARIMA baseline"),
            ModelInfo(name="SARIMA", order=str(SARIMA_ORDER),
                      seasonal_order=str(SARIMA_SEASONAL),
                      description="Seasonal ARIMA with semi-annual cycle"),
            ModelInfo(name="SARIMAX", order=str(SARIMAX_ORDER),
                      seasonal_order=str(SARIMAX_SEASONAL),
                      description="SARIMAX with lag-1 macroeconomic regressors"),
            ModelInfo(name="Hybrid SARIMAX-EGARCH",
                      order=str(SARIMAX_ORDER),
                      seasonal_order=str(SARIMAX_SEASONAL),
                      description=f"SARIMAX + EGARCH({EGARCH_P},{EGARCH_Q}) volatility overlay"),
        ],
        seasonal_period=M_SEASONAL,
        test_year=TEST_YEAR,
        n_forecast=N_FORECAST,
    )


@app.get("/api/data", response_model=DataResponse)
def data():
    df, train, test, exo = _load_data()
    teu_pts = [
        TEUPoint(date=d.strftime("%Y-%m-%d"), teu=float(v))
        for d, v in df["TEU"].items()
    ]
    exo_pts = [
        ExogenousPoint(
            date=d.strftime("%Y-%m-%d"),
            cpi=float(row.get("CPI", np.nan)) if not pd.isna(row.get("CPI", np.nan)) else None,
            inflation=float(row.get("Inflation", np.nan)) if not pd.isna(row.get("Inflation", np.nan)) else None,
            interest_rate=float(row.get("Interest rates", np.nan)) if not pd.isna(row.get("Interest rates", np.nan)) else None,
        )
        for d, row in exo.iterrows()
    ]
    return DataResponse(
        teu_series=teu_pts,
        exogenous=exo_pts,
        train_end=train.index[-1].strftime("%Y-%m-%d"),
        test_start=test.index[0].strftime("%Y-%m-%d"),
        n_train=len(train),
        n_test=len(test),
    )


@app.get("/api/forecast", response_model=ForecastResponse)
def forecast():
    _, train, test, _ = _load_data()
    arima_m, sarima_m, sarimax_m, egarch_res = _fit_models()

    fc = {
        "ARIMA": forecast_arima(arima_m, N_FORECAST),
        "SARIMA": forecast_sarima(sarima_m, N_FORECAST),
        "SARIMAX": forecast_sarimax(sarimax_m, N_FORECAST, test[EXO_LAG_COLS]),
    }
    hyb = forecast_hybrid(sarimax_m, egarch_res, N_FORECAST, test[EXO_LAG_COLS])
    fc["Hybrid"] = hyb["point"]

    actual_vals = test["TEU"].values
    metrics = []
    for name, pred in fc.items():
        m = calc_metrics(actual_vals, pred)
        metrics.append(ModelMetrics(model=name, mae=m["MAE"], rmse=m["RMSE"], mape=m["MAPE"]))

    best = min(metrics, key=lambda m: m.mape).model

    forecasts_out: dict[str, list[ForecastPoint]] = {}
    for name, pred in fc.items():
        pts = []
        pred_arr = pred.values if hasattr(pred, "values") else np.asarray(pred)
        for i, d in enumerate(test.index):
            pts.append(ForecastPoint(
                date=d.strftime("%Y-%m-%d"),
                actual=float(actual_vals[i]),
                predicted=float(pred_arr[i]),
            ))
        forecasts_out[name] = pts

    intervals = []
    for i, d in enumerate(test.index):
        intervals.append(ForecastInterval(
            date=d.strftime("%Y-%m-%d"),
            upper_95=float(hyb["upper_95"][i]),
            lower_95=float(hyb["lower_95"][i]),
            upper_80=float(hyb["upper_80"][i]),
            lower_80=float(hyb["lower_80"][i]),
            volatility=float(hyb["vol"][i]),
        ))

    return ForecastResponse(
        metrics=metrics,
        forecasts=forecasts_out,
        hybrid_intervals=intervals,
        best_model=best,
    )


@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
def diagnostics():
    _, train, test, _ = _load_data()
    arima_m, sarima_m, sarimax_m, _ = _fit_models()

    diag = residual_diagnostics(sarimax_m, train["TEU"], train[EXO_LAG_COLS])
    resid = diag["residuals"]

    resid_pts = [
        TEUPoint(date=d.strftime("%Y-%m-%d"), teu=float(v))
        for d, v in resid.items()
    ]

    lb = diag["ljung_box"]
    lb_rows = [
        LjungBoxRow(lag=int(idx), lb_stat=float(row["lb_stat"]), lb_pvalue=float(row["lb_pvalue"]))
        for idx, row in lb.iterrows()
    ]

    # DM tests
    fc = {
        "ARIMA": forecast_arima(arima_m, N_FORECAST),
        "SARIMA": forecast_sarima(sarima_m, N_FORECAST),
        "SARIMAX": forecast_sarimax(sarimax_m, N_FORECAST, test[EXO_LAG_COLS]),
    }
    names = list(fc.keys())
    dm_tests = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                p1 = fc[n1].values if hasattr(fc[n1], "values") else fc[n1]
                p2 = fc[n2].values if hasattr(fc[n2], "values") else fc[n2]
                dm = diebold_mariano(test["TEU"].values, p1, p2)
                dm_tests.append(DMTestResult(
                    model_a=n1,
                    model_b=n2,
                    dm_statistic=round(float(dm), 3),
                    significant_5pct=abs(dm) > 1.96,
                ))

    return DiagnosticsResponse(
        residual_mean=float(resid.mean()),
        residual_std=float(resid.std()),
        residuals=resid_pts,
        ljung_box=lb_rows,
        ljung_box_pass=bool((lb["lb_pvalue"] > 0.05).all()),
        jb_stat=diag["jb_stat"],
        jb_pval=diag["jb_pval"],
        skewness=diag["skewness"],
        kurtosis=diag["kurtosis"],
        jb_pass=diag["jb_pval"] >= 0.05,
        dm_tests=dm_tests,
    )


@app.post("/api/grid-search", response_model=GridSearchResponse)
def grid_search():
    _, train, test, _ = _load_data()
    gs = grid_search_sarimax(train["TEU"], train[EXO_LAG_COLS],
                             test["TEU"], test[EXO_LAG_COLS])
    rows = [
        GridSearchRow(
            order=str(r["order"]),
            seasonal=str(r["seasonal"]),
            aic=float(r["AIC"]),
            test_mape=float(r["Test_MAPE"]),
        )
        for _, r in gs.iterrows()
    ]
    best = gs.iloc[0]
    return GridSearchResponse(
        results=rows,
        best_order=str(best["order"]),
        best_seasonal=str(best["seasonal"]),
        best_mape=float(best["Test_MAPE"]),
    )


@app.post("/api/cross-validation", response_model=CVResponse)
def cross_validation():
    df, _, _, _ = _load_data()
    cv = time_series_cv(df)
    rows = [
        CVRow(
            model=r["Model"],
            cv_mape_mean=float(r["CV_MAPE_mean"]),
            cv_mape_std=float(r["CV_MAPE_std"]),
            folds=int(r["Folds"]),
        )
        for _, r in cv.iterrows()
    ]
    return CVResponse(results=rows)
