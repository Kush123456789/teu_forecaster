"""
Model engine: fit, forecast, evaluate ARIMA / SARIMA / SARIMAX / Hybrid SARIMAX-EGARCH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pmdarima as pm
from arch import arch_model
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

from core.config import (
    M_SEASONAL,
    ARIMA_ORDER, SARIMA_ORDER, SARIMA_SEASONAL,
    SARIMAX_ORDER, SARIMAX_SEASONAL,
    EGARCH_P, EGARCH_Q, EGARCH_SIMULATIONS,
    GRID_P, GRID_Q, GRID_SP, GRID_SQ,
    EXO_LAG_COLS, CV_FOLDS, CV_MIN_TRAIN,
)


# ── Metrics ──────────────────────────────────────────────────────────────
def calc_metrics(actual, pred) -> dict:
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── ARIMA ────────────────────────────────────────────────────────────────
def fit_arima(train_teu: pd.Series):
    """Fit fixed ARIMA and return (model, forecast_array)."""
    model = pm.ARIMA(order=ARIMA_ORDER, suppress_warnings=True).fit(train_teu)
    return model


def forecast_arima(model, n: int):
    return model.predict(n_periods=n)


# ── SARIMA ───────────────────────────────────────────────────────────────
def fit_sarima(train_teu: pd.Series):
    model = pm.ARIMA(
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL,
        suppress_warnings=True,
    ).fit(train_teu)
    return model


def forecast_sarima(model, n: int):
    return model.predict(n_periods=n)


# ── SARIMAX ──────────────────────────────────────────────────────────────
def fit_sarimax(train_teu: pd.Series, train_exo: pd.DataFrame):
    model = pm.ARIMA(
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL,
        suppress_warnings=True,
    ).fit(train_teu, X=train_exo)
    return model


def forecast_sarimax(model, n: int, test_exo: pd.DataFrame):
    return model.predict(n_periods=n, X=test_exo)


# ── Grid Search ──────────────────────────────────────────────────────────
def grid_search_sarimax(train_teu, train_exo, test_teu, test_exo) -> pd.DataFrame:
    results = []
    for p, q, P, Q in product(GRID_P, GRID_Q, GRID_SP, GRID_SQ):
        try:
            mod = pm.ARIMA(
                order=(p, 1, q),
                seasonal_order=(P, 1, Q, M_SEASONAL),
                suppress_warnings=True,
            ).fit(train_teu, X=train_exo)
            fc = mod.predict(n_periods=len(test_teu), X=test_exo)
            m = calc_metrics(test_teu.values, fc)
            results.append({
                "order": (p, 1, q),
                "seasonal": (P, 1, Q, M_SEASONAL),
                "AIC": mod.aic(),
                "Test_MAPE": m["MAPE"],
            })
        except Exception:
            pass
    return pd.DataFrame(results).sort_values("Test_MAPE")


# ── Hybrid SARIMAX-EGARCH ───────────────────────────────────────────────
def fit_egarch(sarimax_model, train_teu, train_exo):
    """Fit EGARCH(1,1) on SARIMAX residuals. Return fitted result."""
    in_sample = sarimax_model.predict_in_sample(X=train_exo)
    residuals = train_teu - in_sample
    am = arch_model(residuals, mean="Zero", vol="EGARCH", p=EGARCH_P, q=EGARCH_Q)
    return am.fit(disp="off"), residuals


def forecast_hybrid(sarimax_model, egarch_res, n, test_exo):
    """Return point forecast + prediction intervals."""
    point = sarimax_model.predict(n_periods=n, X=test_exo)

    var_fc = egarch_res.forecast(
        horizon=n, method="simulation", simulations=EGARCH_SIMULATIONS, reindex=False
    )
    vol = np.sqrt(var_fc.variance.values[-1])

    upper_95 = point + 1.96 * vol
    lower_95 = np.clip(point - 1.96 * vol, 0, None)
    upper_80 = point + 1.28 * vol
    lower_80 = np.clip(point - 1.28 * vol, 0, None)

    return {
        "point": point,
        "vol": vol,
        "upper_95": upper_95, "lower_95": lower_95,
        "upper_80": upper_80, "lower_80": lower_80,
    }


# ── Coverage ─────────────────────────────────────────────────────────────
def coverage(actual, lower, upper):
    actual = np.asarray(actual)
    return float(np.mean((actual >= lower) & (actual <= upper)) * 100)


# ── Cross-Validation ────────────────────────────────────────────────────
def time_series_cv(df_model: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window CV for ARIMA, SARIMA, SARIMAX."""
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    results: dict[str, list] = {"ARIMA": [], "SARIMA": [], "SARIMAX": []}

    for train_idx, test_idx in tscv.split(df_model):
        if len(train_idx) < CV_MIN_TRAIN:
            continue
        cv_train = df_model.iloc[train_idx]
        cv_test = df_model.iloc[test_idx]

        # ARIMA
        try:
            m = pm.ARIMA(order=ARIMA_ORDER, suppress_warnings=True).fit(cv_train["TEU"])
            fc = m.predict(n_periods=len(cv_test))
            results["ARIMA"].append(calc_metrics(cv_test["TEU"].values, fc)["MAPE"])
        except Exception:
            pass

        # SARIMA
        try:
            m = pm.ARIMA(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                         suppress_warnings=True).fit(cv_train["TEU"])
            fc = m.predict(n_periods=len(cv_test))
            results["SARIMA"].append(calc_metrics(cv_test["TEU"].values, fc)["MAPE"])
        except Exception:
            pass

        # SARIMAX
        try:
            m = pm.ARIMA(order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL,
                         suppress_warnings=True).fit(cv_train["TEU"], X=cv_train[EXO_LAG_COLS])
            fc = m.predict(n_periods=len(cv_test), X=cv_test[EXO_LAG_COLS])
            results["SARIMAX"].append(calc_metrics(cv_test["TEU"].values, fc)["MAPE"])
        except Exception:
            pass

    rows = []
    for name, mapes in results.items():
        if mapes:
            rows.append({
                "Model": name,
                "CV_MAPE_mean": np.mean(mapes),
                "CV_MAPE_std": np.std(mapes),
                "Folds": len(mapes),
            })
    return pd.DataFrame(rows)


# ── Residual Diagnostics ────────────────────────────────────────────────
def residual_diagnostics(sarimax_model, train_teu, train_exo) -> dict:
    in_sample = sarimax_model.predict_in_sample(X=train_exo)
    resid = train_teu - in_sample

    lb = acorr_ljungbox(resid, lags=[6, 12, 18, 24], return_df=True)
    jb_result = jarque_bera(resid)
    jb_stat = float(jb_result.statistic)
    jb_p = float(jb_result.pvalue)
    skew = float(resid.skew())
    kurt = float(resid.kurtosis())

    return {
        "residuals": resid,
        "ljung_box": lb,
        "jb_stat": jb_stat,
        "jb_pval": jb_p,
        "skewness": skew,
        "kurtosis": kurt,
    }


# ── Diebold-Mariano Test ────────────────────────────────────────────────
def diebold_mariano(actual, pred1, pred2) -> float:
    e1 = np.asarray(actual) - np.asarray(pred1)
    e2 = np.asarray(actual) - np.asarray(pred2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dm = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n))) if np.std(d) > 0 else 0.0
    return dm
