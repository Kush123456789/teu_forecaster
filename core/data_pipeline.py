"""
Data pipeline: load KPA Excel, backcast, build modelling dataframe with lag-1 exogenous.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import CubicSpline
from statsmodels.stats.outliers_influence import variance_inflation_factor

from core.config import (
    M_SEASONAL, MONTH_MAP, EXO_RAW_COLS, EXO_LAG_COLS, TEST_YEAR,
)


# ── 1. Load primary KPA TEU data ────────────────────────────────────────
def load_actual_teu(path: str) -> pd.Series:
    """Load KPA 5-year Excel and return monthly TEU series."""
    df = pd.read_excel(path, sheet_name="Sheet2")
    df["MONTH_NUM"] = df["MONTH"].map(MONTH_MAP)
    mask = (df["TYPE 1"] == "IMPORT") & (df["TYPE 2"] == "LOCAL")
    df = df[mask].copy()
    df["Actual_TEU"] = df["20FT"] + 2 * df["40FT"]
    df["Date"] = pd.to_datetime(
        df[["YEAR", "MONTH_NUM"]].assign(DAY=1).rename(columns={"MONTH_NUM": "MONTH"})
    )
    series = df.sort_values("Date").set_index("Date")["Actual_TEU"].asfreq("MS")
    return series


# ── 2. Backcast 2015-2019 ────────────────────────────────────────────────
def backcast(actual: pd.Series, n_back: int = 60) -> pd.Series:
    """Extend series backward using SARIMAX-estimated drift + seasonal."""
    model = sm.tsa.statespace.SARIMAX(
        actual, order=(1, 1, 1), seasonal_order=(0, 1, 0, M_SEASONAL), trend="c"
    )
    res = model.fit(disp=False)
    drift = res.params.get("drift", 60.0)

    t = np.arange(-n_back, 0)
    y0 = actual.iloc[0]
    trend = y0 + drift * t

    seasonal_res = actual - (y0 + drift * np.arange(len(actual)))
    pattern = [seasonal_res[i::M_SEASONAL].mean() for i in range(M_SEASONAL)]
    back = [trend[i] + pattern[i % M_SEASONAL] for i in range(n_back)]

    back_idx = pd.date_range(
        end=actual.index[0] - pd.DateOffset(months=1), periods=n_back, freq="MS"
    )
    return pd.concat([pd.Series(back, index=back_idx), actual])


# ── 3. Load exogenous variables ──────────────────────────────────────────
def load_exogenous(path: str) -> pd.DataFrame:
    """Load CPI / Inflation / Interest-rates Excel."""
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Month"] + "-01")
    df.set_index("Date", inplace=True)
    df.drop(columns=["Month"], inplace=True)
    return df


# ── 4. Build modelling dataframe ─────────────────────────────────────────
def build_model_df(teu_series: pd.Series, exo_df: pd.DataFrame) -> pd.DataFrame:
    """Combine TEU + lag-1 exogenous into a single modelling frame."""
    df = teu_series.to_frame(name="TEU")
    lagged = exo_df.shift(1)
    lagged.columns = EXO_LAG_COLS
    df = df.join(lagged).dropna()
    return df


# ── 5. Train / test split ────────────────────────────────────────────────
def split(df: pd.DataFrame):
    """Return (train, test) split at TEST_YEAR boundary."""
    train = df[df.index < f"{TEST_YEAR}-01-01"]
    test = df[df.index >= f"{TEST_YEAR}-01-01"]
    return train, test


# ── 6. VIF ────────────────────────────────────────────────────────────────
def compute_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    X = df[features].values
    return pd.DataFrame({
        "Feature": features,
        "VIF": [variance_inflation_factor(X, i) for i in range(len(features))],
    })


# ── 7. Daily reconciliation (optional) ──────────────────────────────────
def daily_reconcile(targets: pd.DataFrame) -> pd.DataFrame:
    """Cubic-spline disaggregation with weekday weighting & exact reconciliation."""
    daily_idx = pd.date_range(
        start=targets.index.min(), end=targets.index.max() + pd.offsets.MonthEnd(0), freq="D"
    )
    x = (targets.index - daily_idx[0]).days.values
    y = targets["Target_TEU"].values if "Target_TEU" in targets.columns else targets.iloc[:, 0].values
    cs = CubicSpline(x, y, bc_type="natural")
    curve = cs((daily_idx - daily_idx[0]).days.values)

    df = pd.DataFrame({"Curve": curve}, index=daily_idx)
    df["Month_Start"] = df.index.to_period("M").to_timestamp()
    target_col = targets.columns[0]
    df = df.join(targets[[target_col]], on="Month_Start")
    df["Weights"] = [1.0 if d.weekday() < 6 else 0.5 for d in df.index]
    df["Base"] = df["Curve"] * df["Weights"]
    df["Benchmarked"] = df[target_col] * (
        df["Base"] / df.groupby("Month_Start")["Base"].transform("sum")
    )

    rng = np.random.default_rng(42)
    df["TEU"] = (
        df["Benchmarked"] + rng.normal(0, 1527.53 / np.sqrt(30), len(df))
    ).clip(lower=0).round().astype(int)

    def reconcile(group):
        t = int(group[target_col].iloc[0])
        diff = t - group["TEU"].sum()
        if diff != 0:
            indices = rng.choice(group.index, abs(diff), replace=True)
            for idx in indices:
                group.at[idx, "TEU"] += int(np.sign(diff))
        return group

    return df.groupby("Month_Start", group_keys=False).apply(reconcile)
