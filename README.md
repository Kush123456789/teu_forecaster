# Mombasa Port TEU Forecaster

**Streamlit dashboard for forecasting monthly container import volumes at the Port of Mombasa** using ARIMA, SARIMA, SARIMAX, and a Hybrid SARIMAX-EGARCH model.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT OPTIONS                           │
│                                                                     │
│   ┌─────────────┐  ┌─────────────┐  ┌───────────────────────────┐  │
│   │  Streamlit   │  │   Docker    │  │   Cloud (Azure / AWS)     │  │
│   │  Community   │  │  Compose    │  │                           │  │
│   │  Cloud       │  │  (self-     │  │  Azure App Service        │  │
│   │  (Free)      │  │   hosted)   │  │  + Container Instances    │  │
│   └──────┬───────┘  └──────┬──────┘  └────────────┬──────────────┘  │
│          │                 │                       │                 │
│          └─────────────────┼───────────────────────┘                 │
│                            │                                        │
│                    ┌───────▼───────┐                                 │
│                    │  Streamlit    │  Port 8501                      │
│                    │  App Server   │                                 │
│                    └───────┬───────┘                                 │
│                            │                                        │
│             ┌──────────────┼──────────────┐                         │
│             │              │              │                         │
│      ┌──────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐                  │
│      │   app.py   │ │   pages/   │ │  assets/   │                  │
│      │  (entry)   │ │  4 pages   │ │  style.css │                  │
│      └──────┬─────┘ └─────┬──────┘ └────────────┘                  │
│             │              │                                        │
│             └──────┬───────┘                                        │
│                    │                                                │
│             ┌──────▼──────┐                                         │
│             │    core/    │                                          │
│             │             │                                          │
│             │ config.py   │  Constants, orders, paths               │
│             │ data_pipe.. │  Load Excel, backcast, lag exogenous    │
│             │ model_eng.. │  Fit/forecast 4 models, CV, DM test    │
│             └──────┬──────┘                                         │
│                    │                                                │
│             ┌──────▼──────┐                                         │
│             │    data/    │  Excel files (KPA + Exogenous)          │
│             └─────────────┘                                         │
└──────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mombasa-port-forecaster/
├── .streamlit/
│   └── config.toml              # Dark theme, port, security
├── app.py                       # Entry point — home + sidebar nav
├── core/
│   ├── __init__.py
│   ├── config.py                # All constants (M_SEASONAL=6, orders, etc.)
│   ├── data_pipeline.py         # Load KPA Excel, backcast, exogenous, VIF
│   └── model_engine.py          # ARIMA, SARIMA, SARIMAX, EGARCH, CV, DM
├── pages/
│   ├── 1_📊_Dashboard.py       # KPIs, forecast overlay, prediction intervals
│   ├── 2_🔍_EDA.py             # Time series, stats, seasonality, ACF/PACF
│   ├── 3_📈_Models.py          # Model comparison, grid search, cross-val
│   └── 4_🧪_Diagnostics.py     # Residuals, Ljung-Box, JB, Q-Q, DM test
├── assets/
│   └── style.css                # TradingView-inspired terminal theme
├── data/
│   ├── .gitkeep
│   ├── KPA MOMBASA PORT - 5YR Summary.xlsx   ← place here
│   └── Exogenous Variables.xlsx               ← place here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Local Development

```bash
# Clone / navigate to project
cd mombasa-port-forecaster

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Place data files in data/
#   - KPA MOMBASA PORT - 5YR Summary.xlsx
#   - Exogenous Variables.xlsx

# Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 2. Docker

```bash
# Place data files in data/ first

docker compose up --build
```

### 3. Streamlit Community Cloud (Free)

1. Push to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo → select `app.py`
4. Upload data files via the `data/` folder or use `st.file_uploader`

### 4. Azure App Service

```bash
# Build & push image
az acr build --registry <ACR_NAME> --image mombasa-forecaster:latest .

# Create App Service
az webapp create \
  --resource-group <RG> \
  --plan <PLAN> \
  --name mombasa-forecaster \
  --deployment-container-image-name <ACR_NAME>.azurecr.io/mombasa-forecaster:latest

# Set port
az webapp config appsettings set \
  --name mombasa-forecaster \
  --settings WEBSITES_PORT=8501
```

---

## Models

| # | Model | Order | MAPE |
|---|-------|-------|------|
| 1 | ARIMA | (2,1,2) | 7.74% |
| 2 | SARIMA | (0,1,0)×(0,1,0,6) | 4.30% |
| 3 | SARIMAX | (1,1,1)×(1,1,0,6) + CPI, Inflation, CBR | 3.89% |
| 4 | Hybrid SARIMAX-EGARCH | SARIMAX + EGARCH(1,1) volatility | 3.89% + calibrated intervals |

## Data Requirements

Place two Excel files in the `data/` folder:

1. **KPA MOMBASA PORT - 5YR Summary.xlsx** — Monthly container counts (Sheet2)
   - Columns: `YEAR`, `MONTH`, `TYPE 1`, `TYPE 2`, `20FT`, `40FT`
2. **Exogenous Variables.xlsx** — Macroeconomic indicators
   - Columns: `Month`, `CPI`, `Inflation`, `Interest rates`

## UI Theme

TradingView-inspired terminal aesthetic:
- **Background**: `#0B0F19` (deep navy)
- **Accent**: `#38BDF8` (neon cyan)
- **Typography**: JetBrains Mono (monospaced)
- **Charts**: Transparent Plotly backgrounds with `#1E293B` gridlines
- **Neon palette**: Cyan / Indigo / Emerald / Rose / Amber
