"""
Pydantic schemas for the Mombasa Port TEU Forecaster API.

Includes field-level constraints, format validators, and cross-field checks.
"""
from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator

# ── Reusable validated types ─────────────────────────────────────────────
_DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")


def _validate_date(v: str) -> str:
    """Ensure YYYY-MM-DD format."""
    if not _DATE_RE.match(v):
        raise ValueError("date must be in YYYY-MM-DD format")
    return v


DateStr = Annotated[str, AfterValidator(_validate_date)]


# ── Health ───────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")


# ── Data ─────────────────────────────────────────────────────────────────
class TEUPoint(BaseModel):
    date: DateStr
    teu: float


class ExogenousPoint(BaseModel):
    date: DateStr
    cpi: float | None = None
    inflation: float | None = None
    interest_rate: float | None = None


class DataResponse(BaseModel):
    teu_series: list[TEUPoint]
    exogenous: list[ExogenousPoint]
    train_end: DateStr
    test_start: DateStr
    n_train: int = Field(gt=0)
    n_test: int = Field(gt=0)


# ── Forecast ─────────────────────────────────────────────────────────────
class ModelMetrics(BaseModel):
    model: str = Field(min_length=1)
    mae: float = Field(ge=0)
    rmse: float = Field(ge=0)
    mape: float = Field(ge=0)


class ForecastPoint(BaseModel):
    date: DateStr
    actual: float | None = None
    predicted: float


class ForecastInterval(BaseModel):
    date: DateStr
    upper_95: float = Field(ge=0)
    lower_95: float = Field(ge=0)
    upper_80: float = Field(ge=0)
    lower_80: float = Field(ge=0)
    volatility: float = Field(ge=0)

    @model_validator(mode="after")
    def check_interval_bounds(self) -> ForecastInterval:
        if self.lower_95 > self.upper_95:
            raise ValueError("lower_95 must be <= upper_95")
        if self.lower_80 > self.upper_80:
            raise ValueError("lower_80 must be <= upper_80")
        if self.lower_80 < self.lower_95:
            raise ValueError("80% lower bound should be >= 95% lower bound")
        if self.upper_80 > self.upper_95:
            raise ValueError("80% upper bound should be <= 95% upper bound")
        return self


class ForecastResponse(BaseModel):
    metrics: list[ModelMetrics]
    forecasts: dict[str, list[ForecastPoint]]
    hybrid_intervals: list[ForecastInterval]
    best_model: str = Field(min_length=1)


# ── Diagnostics ──────────────────────────────────────────────────────────
class LjungBoxRow(BaseModel):
    lag: int = Field(gt=0)
    lb_stat: float = Field(ge=0)
    lb_pvalue: float = Field(ge=0, le=1)


class DMTestResult(BaseModel):
    model_a: str = Field(min_length=1)
    model_b: str = Field(min_length=1)
    dm_statistic: float
    significant_5pct: bool


class DiagnosticsResponse(BaseModel):
    residual_mean: float
    residual_std: float = Field(ge=0)
    residuals: list[TEUPoint]
    ljung_box: list[LjungBoxRow]
    ljung_box_pass: bool
    jb_stat: float = Field(ge=0)
    jb_pval: float = Field(ge=0, le=1)
    skewness: float
    kurtosis: float
    jb_pass: bool
    dm_tests: list[DMTestResult]


# ── Grid Search ──────────────────────────────────────────────────────────
class GridSearchRow(BaseModel):
    order: str = Field(min_length=1)
    seasonal: str = Field(min_length=1)
    aic: float
    test_mape: float = Field(ge=0)


class GridSearchResponse(BaseModel):
    results: list[GridSearchRow]
    best_order: str = Field(min_length=1)
    best_seasonal: str = Field(min_length=1)
    best_mape: float = Field(ge=0)


# ── Cross-Validation ────────────────────────────────────────────────────
class CVRow(BaseModel):
    model: str = Field(min_length=1)
    cv_mape_mean: float = Field(ge=0)
    cv_mape_std: float = Field(ge=0)
    folds: int = Field(gt=0)


class CVResponse(BaseModel):
    results: list[CVRow]


# ── Model Info ───────────────────────────────────────────────────────────
class ModelInfo(BaseModel):
    name: str = Field(min_length=1)
    order: str = Field(min_length=1)
    seasonal_order: str | None = None
    description: str = Field(min_length=1)


class ModelsInfoResponse(BaseModel):
    models: list[ModelInfo] = Field(min_length=1)
    seasonal_period: int = Field(gt=0)
    test_year: int = Field(ge=2000, le=2100)
    n_forecast: int = Field(gt=0)
