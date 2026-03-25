"""
Mombasa Port TEU Forecaster — Streamlit Entry Point
====================================================
Multi-page app: Dashboard · EDA · Models · Diagnostics
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Mombasa Port TEU Forecaster",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Data file paths ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TEU_PATH  = DATA_DIR / "KPA MOMBASA PORT - 5YR Summary.xlsx"
EXO_PATH  = DATA_DIR / "Exogenous Variables.xlsx"

# ── Sidebar branding ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:1.2rem 0 0.6rem'>
            <span style='font-size:2.4rem'>🚢</span><br>
            <span style='font-family:Inter,sans-serif; font-weight:700;
                         font-size:1.1rem; color:#F8FAFC;
                         letter-spacing:-0.02em'>
                MOMBASA PORT
            </span><br>
            <span style='font-family:JetBrains Mono,monospace;
                         font-size:0.65rem; color:#38BDF8;
                         letter-spacing:0.14em; text-transform:uppercase'>
                TEU FORECASTER
            </span>
            <div style='margin-top:0.5rem'>
                <span class='live-dot'></span>
                <span style='font-family:JetBrains Mono,monospace;
                             font-size:0.6rem; color:#34D399;
                             letter-spacing:0.05em'>SYSTEM ONLINE</span>
            </div>
        </div>
        <hr style='margin:0.5rem 0 1rem'>
        """,
        unsafe_allow_html=True,
    )

    # ── Data upload section ──────────────────────────────────────────────
    st.markdown(
        "<span style='font-family:Inter,sans-serif; font-size:0.8rem; "
        "color:#94A3B8; font-weight:600; letter-spacing:0.05em'>📂 DATA FILES</span>",
        unsafe_allow_html=True,
    )
    teu_upload = st.file_uploader(
        "KPA TEU Summary (.xlsx)",
        type=["xlsx"],
        key="teu_upload",
        help="Upload 'KPA MOMBASA PORT - 5YR Summary.xlsx'",
    )
    exo_upload = st.file_uploader(
        "Exogenous Variables (.xlsx)",
        type=["xlsx"],
        key="exo_upload",
        help="Upload 'Exogenous Variables.xlsx'",
    )

    if teu_upload:
        TEU_PATH.write_bytes(teu_upload.read())
        st.success("✅ TEU data saved")
    if exo_upload:
        EXO_PATH.write_bytes(exo_upload.read())
        st.success("✅ Exogenous data saved")

    st.markdown("<hr style='margin:0.5rem 0 1rem'>", unsafe_allow_html=True)


# ── Hero Banner ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='hero-banner'>
        <h1 style='margin:0 0 0.4rem; font-size:1.8rem'>
            🚢 Mombasa Port TEU Forecaster
        </h1>
        <p style='font-family:Inter,sans-serif; font-size:0.92rem;
                  color:#94A3B8; margin:0; line-height:1.6'>
            Forecasting Monthly Container Import Volumes at the Port of Mombasa
            using <span style='color:#38BDF8'>ARIMA</span>,
            <span style='color:#818CF8'>SARIMA</span>,
            <span style='color:#34D399'>SARIMAX</span>, and a
            <span style='color:#FB7185'>Hybrid SARIMAX-EGARCH</span> model
            with macroeconomic regressors.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Data readiness check ─────────────────────────────────────────────────
_teu_ok = TEU_PATH.exists()
_exo_ok = EXO_PATH.exists()
if not (_teu_ok and _exo_ok):
    missing = []
    if not _teu_ok:
        missing.append("**KPA TEU Summary** (`KPA MOMBASA PORT - 5YR Summary.xlsx`)")
    if not _exo_ok:
        missing.append("**Exogenous Variables** (`Exogenous Variables.xlsx`)")
    st.warning(
        "⚠️ **Data files not found.** Upload the following files using the "
        "📂 DATA FILES uploaders in the sidebar before navigating to any page:\n\n"
        + "\n".join(f"- {m}" for m in missing),
        icon="📂",
    )

# ── Quick Stats Row ──────────────────────────────────────────────────────
st.markdown(
    """
    <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin-bottom:1.5rem'>
        <div class='stat-card'>
            <div class='stat-value'>4</div>
            <div class='stat-label'>Models</div>
            <div class='stat-sub'>ARIMA · SARIMA · SARIMAX · Hybrid</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>3.89%</div>
            <div class='stat-label'>Best MAPE</div>
            <div class='stat-sub'>Hybrid SARIMAX-EGARCH(1,1)</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value' style='color:#34D399'>5 yr</div>
            <div class='stat-label'>Data Span</div>
            <div class='stat-sub'>KPA Mombasa 2020 – 2025</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value' style='color:#818CF8'>3</div>
            <div class='stat-label'>Exogenous Vars</div>
            <div class='stat-sub'>CPI · Inflation · Interest Rate</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Navigation Cards ────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<div class='accent-dot'></div>"
    "<span style='font-family:Inter,sans-serif; font-weight:600; "
    "color:#F8FAFC; font-size:1rem'>Navigation</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin-bottom:1.5rem'>
        <div class='nav-card'>
            <div class='nav-icon'>📊</div>
            <div class='nav-title'>Dashboard</div>
            <div class='nav-desc'>KPI metrics, forecast overlay, prediction intervals, conditional volatility</div>
        </div>
        <div class='nav-card'>
            <div class='nav-icon'>🔍</div>
            <div class='nav-title'>EDA</div>
            <div class='nav-desc'>Time series, seasonality, correlation, stationarity tests, ACF/PACF</div>
        </div>
        <div class='nav-card'>
            <div class='nav-icon'>📈</div>
            <div class='nav-title'>Models</div>
            <div class='nav-desc'>Model comparison, specifications, grid search, cross-validation</div>
        </div>
        <div class='nav-card'>
            <div class='nav-icon'>🧪</div>
            <div class='nav-title'>Diagnostics</div>
            <div class='nav-desc'>Residual analysis, Ljung-Box, Jarque-Bera, Diebold-Mariano test</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div class='footer-text'>
        DISSERTATION · STRATHMORE UNIVERSITY MSc. DATA SCIENCE & ANALYTICS<br>
        DATA: KPA MOMBASA 5-YEAR SUMMARY (2020–2025) + CBS MACROECONOMIC INDICATORS
    </div>
    """,
    unsafe_allow_html=True,
)
