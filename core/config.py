"""
Central configuration for the Mombasa Port TEU Forecasting App.
All model hyperparameters, paths, and constants validated via Pydantic.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppConfig(BaseModel):
    """Validated application configuration (immutable after creation)."""

    model_config = ConfigDict(frozen=True)

    # ── Paths ────────────────────────────────────────────────────────────
    base_dir: Path
    data_dir: Path

    # ── Seasonal Period ──────────────────────────────────────────────────
    seasonal_period: int = Field(gt=0)

    # ── Model Orders ─────────────────────────────────────────────────────
    arima_order: tuple[int, int, int]
    sarima_order: tuple[int, int, int]
    sarima_seasonal: tuple[int, int, int, int]
    sarimax_order: tuple[int, int, int]
    sarimax_seasonal: tuple[int, int, int, int]

    # ── Exogenous columns ────────────────────────────────────────────────
    exo_raw_cols: list[str] = Field(min_length=1)

    # ── Train / Test ─────────────────────────────────────────────────────
    test_year: int = Field(ge=2000, le=2100)
    n_forecast: int = Field(gt=0)

    # ── EGARCH ───────────────────────────────────────────────────────────
    egarch_p: int = Field(gt=0)
    egarch_q: int = Field(gt=0)
    egarch_simulations: int = Field(gt=0)

    # ── Cross-Validation ─────────────────────────────────────────────────
    cv_folds: int = Field(ge=2)
    cv_min_train: int = Field(gt=0)

    # ── Grid Search upper bounds ─────────────────────────────────────────
    grid_max_p: int = Field(ge=0)
    grid_max_q: int = Field(ge=0)
    grid_max_sp: int = Field(ge=0)
    grid_max_sq: int = Field(ge=0)

    @field_validator("arima_order", "sarima_order", "sarimax_order")
    @classmethod
    def _non_negative_order(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(x < 0 for x in v):
            raise ValueError("All (p, d, q) components must be >= 0")
        return v

    @field_validator("sarima_seasonal", "sarimax_seasonal")
    @classmethod
    def _valid_seasonal(
        cls, v: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if any(x < 0 for x in v[:3]):
            raise ValueError("Seasonal P, D, Q must be >= 0")
        if v[3] <= 0:
            raise ValueError("Seasonal period m must be > 0")
        return v

    @property
    def exo_lag_cols(self) -> list[str]:
        return [f"{c}_lag1" for c in self.exo_raw_cols]


# ── Instantiate & validate ───────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent

_cfg = AppConfig(
    base_dir=_BASE_DIR,
    data_dir=_BASE_DIR / "data",
    seasonal_period=6,
    arima_order=(2, 1, 2),
    sarima_order=(0, 1, 0),
    sarima_seasonal=(0, 1, 0, 6),
    sarimax_order=(1, 1, 1),
    sarimax_seasonal=(1, 1, 0, 6),
    exo_raw_cols=["CPI", "Inflation", "Interest rates"],
    test_year=2025,
    n_forecast=12,
    egarch_p=1,
    egarch_q=1,
    egarch_simulations=1000,
    cv_folds=5,
    cv_min_train=60,
    grid_max_p=2,
    grid_max_q=2,
    grid_max_sp=1,
    grid_max_sq=1,
)

# ── Module-level exports (backward-compatible) ──────────────────────────
BASE_DIR = _cfg.base_dir
DATA_DIR = _cfg.data_dir

M_SEASONAL = _cfg.seasonal_period

ARIMA_ORDER = _cfg.arima_order
SARIMA_ORDER = _cfg.sarima_order
SARIMA_SEASONAL = _cfg.sarima_seasonal
SARIMAX_ORDER = _cfg.sarimax_order
SARIMAX_SEASONAL = _cfg.sarimax_seasonal

EXO_RAW_COLS = _cfg.exo_raw_cols
EXO_LAG_COLS = _cfg.exo_lag_cols

TEST_YEAR = _cfg.test_year
N_FORECAST = _cfg.n_forecast

EGARCH_P = _cfg.egarch_p
EGARCH_Q = _cfg.egarch_q
EGARCH_SIMULATIONS = _cfg.egarch_simulations

CV_FOLDS = _cfg.cv_folds
CV_MIN_TRAIN = _cfg.cv_min_train

GRID_P = range(_cfg.grid_max_p + 1)
GRID_Q = range(_cfg.grid_max_q + 1)
GRID_SP = range(_cfg.grid_max_sp + 1)
GRID_SQ = range(_cfg.grid_max_sq + 1)

# ── Month Mapping (KPA data) ─────────────────────────────────────────────
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUNE": 6,
    "JULY": 7, "AUG": 8, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
