"""
Portfolio Optimization Dashboard — SPO Platform

A working portfolio optimization tool built on the SPO pipeline.
Users configure their universe, risk parameters, and model, then
the dashboard runs live predictions and Markowitz optimization
to produce actionable portfolio weights and analytics.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
import torch

# ── Paths ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

RAW_PRICES_PATH = os.path.join(ROOT_DIR, "data", "raw data", "raw.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "spo")

from spo.prediction_net import ReturnPredictionNet
from spo.portfolio_layer import DifferentiableMarkowitz
from spo.covariance import compute_rolling_covariance

# ── Theme ──
C = {
    "bg": "#0E1117", "card": "#161B22", "border": "#2D3139",
    "text": "#FAFAFA", "muted": "#8B949E",
    "blue": "#6C8EBF", "gold": "#D4A03C", "green": "#82B366",
    "red": "#E06C6C", "teal": "#4EC9B0", "purple": "#9673A6",
}

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "NVDA": "Technology", "META": "Technology",
    "TSLA": "Technology", "NFLX": "Technology", "AMD": "Technology",
    "INTC": "Technology", "CSCO": "Technology", "IBM": "Technology",
    "ORCL": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "AVGO": "Technology", "CRM": "Technology", "ACN": "Technology",
    "JPM": "Financials", "V": "Financials", "BAC": "Financials",
    "MA": "Financials", "WFC": "Financials", "C": "Financials",
    "GS": "Financials", "MS": "Financials", "AXP": "Financials",
    "SCHW": "Financials", "BLK": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "ABBV": "Healthcare", "TMO": "Healthcare",
    "MRK": "Healthcare", "DHR": "Healthcare", "ABT": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare",
    "WMT": "Consumer", "PG": "Consumer", "HD": "Consumer",
    "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
    "TGT": "Consumer", "LOW": "Consumer", "COST": "Consumer",
    "DIS": "Consumer", "TMUS": "Consumer", "VZ": "Consumer",
    "XOM": "Energy/Ind.", "CVX": "Energy/Ind.", "COP": "Energy/Ind.",
    "SLB": "Energy/Ind.", "CAT": "Energy/Ind.", "BA": "Energy/Ind.",
    "HON": "Energy/Ind.", "UNP": "Energy/Ind.", "MMM": "Energy/Ind.",
    "GE": "Energy/Ind.", "RTX": "Energy/Ind.", "LMT": "Energy/Ind.",
}

SECTOR_COLORS = {
    "Technology": "#6C8EBF", "Financials": "#D4A03C",
    "Healthcare": "#82B366", "Consumer": "#9673A6",
    "Energy/Ind.": "#E06C6C",
}

# =====================================================================
#  Data & model loading
# =====================================================================

@st.cache_data
def load_prices():
    return pd.read_csv(RAW_PRICES_PATH, index_col=0, parse_dates=True)


@st.cache_data(ttl=3600)  # cache 1 hour so repeated interactions don't re-download
def fetch_live_prices(tickers: tuple, lookback_days: int = 120):
    """
    Fetch recent Close prices from Yahoo Finance for live predictions.
    Returns a DataFrame with the same wide format as raw.csv.
    """
    import yfinance as yf
    raw = yf.download(
        list(tickers), period=f"{lookback_days}d",
        auto_adjust=True, progress=False, threads=True,
    )["Close"]
    # yfinance may return fewer columns if a ticker has no data
    raw = raw.dropna(axis=1, how="all")
    raw.index = pd.to_datetime(raw.index)
    return raw


@st.cache_data
def compute_features(prices):
    """Compute the 10 alpha features from raw close prices."""
    close = prices
    ret_1d = close.pct_change(1)
    ret_5d = close.pct_change(5)
    ret_10d = close.pct_change(10)
    mom_10 = close / close.shift(10) - 1
    mom_20 = close / close.shift(20) - 1
    vol_5 = ret_1d.rolling(5).std()
    vol_10 = ret_1d.rolling(10).std()
    market_ret = ret_1d.mean(axis=1)
    alpha_1d = ret_1d.sub(market_ret, axis=0)
    rank_mom_10 = mom_10.rank(axis=1)
    anti_mom_10 = -mom_10
    return {
        "RET_1D": ret_1d, "RET_5D": ret_5d, "RET_10D": ret_10d,
        "MOM_10": mom_10, "MOM_20": mom_20,
        "VOL_5": vol_5, "VOL_10": vol_10,
        "ALPHA_1D": alpha_1d, "RANK_MOM_10": rank_mom_10,
        "ANTI_MOM_10": anti_mom_10,
    }


@st.cache_resource
def load_model(mode, n_features=10, n_assets=64):
    """Load a trained prediction network."""
    path = os.path.join(MODEL_DIR, f"pred_net_{mode}.pt")
    if not os.path.exists(path):
        return None
    net = ReturnPredictionNet(n_features=n_features, hidden_dims=[64, 32], dropout=0.2)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def build_feature_matrix(features_dict, tickers, date):
    """Build (n_stocks, n_features) matrix for a given date."""
    feature_names = list(features_dict.keys())
    rows = []
    for ticker in tickers:
        vals = []
        for fname in feature_names:
            df = features_dict[fname]
            if ticker in df.columns and date in df.index:
                v = df.loc[date, ticker]
                vals.append(0.0 if pd.isna(v) else float(v))
            else:
                vals.append(0.0)
        rows.append(vals)
    return np.array(rows, dtype=np.float32)


def run_optimization(net, features_matrix, cov_matrix, gamma, max_weight, n_assets):
    """Run the full predict-then-optimize pipeline."""
    port_layer = DifferentiableMarkowitz(n_assets=n_assets, gamma=gamma, max_weight=max_weight)

    feat_tensor = torch.tensor(features_matrix, dtype=torch.float32).unsqueeze(0)
    cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_returns = net(feat_tensor).squeeze(0)
        weights = port_layer(predicted_returns, cov_tensor.squeeze(0))

    return predicted_returns.numpy(), weights.numpy()


# =====================================================================
#  Chart helpers
# =====================================================================

def _dark_layout(fig, title=None, height=400):
    # FIX #1: Plotly renders `None` title as "undefined" in some versions.
    # Always pass a valid string (empty string if no title) to prevent this.
    safe_title = title if isinstance(title, str) and title else ""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=C["text"]),
        title=dict(text=safe_title, font=dict(size=15, color=C["text"]), x=0),
        height=height,
        margin=dict(l=50, r=20, t=50 if safe_title else 20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    )
    return fig


# =====================================================================
#  CSS
# =====================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

/* ── Base ── */
html, body, [class*="st-"] { font-family: 'Inter', system-ui, sans-serif !important; }
.stApp { background: #0E1117; }

/* Fix Material icon buttons rendering as raw text */
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="collapsedControl"] button span {
    font-family: 'Material Symbols Outlined' !important;
    font-size: 20px !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] * { color: #C9D1D9; }
section[data-testid="stSidebar"] h2 { color: #FAFAFA !important; }
section[data-testid="stSidebar"] hr { border-color: #21262D; margin: 12px 0; }
section[data-testid="stSidebar"] label { color: #8B949E !important; font-size: 12px !important; }

/* ── Sidebar sliders ── */
.stSlider > div > div > div { background: #2D3139 !important; }
.stSlider > div > div > div > div { background: #6C8EBF !important; }

/* ── Multiselect chips — override the jarring default red ── */
[data-baseweb="tag"] {
    background-color: #1C3A5C !important;
    border: 1px solid #3A5A8A !important;
    border-radius: 4px !important;
}
[data-baseweb="tag"] span { color: #90BADF !important; font-size: 12px; }
[data-baseweb="tag"] button svg { fill: #6C8EBF !important; }

/* ── Select dropdowns ── */
div[data-baseweb="select"] > div {
    background: #161B22 !important;
    border-color: #2D3139 !important;
    color: #C9D1D9 !important;
}
div[data-baseweb="select"] li { background: #161B22 !important; color: #C9D1D9 !important; }
div[data-baseweb="select"] li:hover { background: #21262D !important; }

/* ── Checkbox ── */
.stCheckbox label { color: #8B949E !important; }
.stCheckbox [data-testid="stCheckboxWidget"] { accent-color: #6C8EBF; }

/* ── Header ── */
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
header[data-testid="stHeader"] { background: #0E1117; border-bottom: 1px solid #21262D; }

/* Style header & sidebar buttons so icons are visible on dark bg */
header[data-testid="stHeader"] button,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background: #161B22 !important;
    border: 1px solid #2D3139 !important;
    border-radius: 6px !important;
    color: #C9D1D9 !important;
    transition: all 0.15s ease;
}
header[data-testid="stHeader"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: #21262D !important;
    border-color: #6C8EBF !important;
    color: #E6EDF3 !important;
}
/* Ensure SVG icons inside header buttons are visible */
header[data-testid="stHeader"] button svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    fill: #8B949E !important;
    stroke: #8B949E !important;
    color: #8B949E !important;
}
header[data-testid="stHeader"] button:hover svg,
[data-testid="stSidebarCollapseButton"] button:hover svg,
[data-testid="collapsedControl"] button:hover svg {
    fill: #E6EDF3 !important;
    stroke: #E6EDF3 !important;
    color: #E6EDF3 !important;
}

/* ── Page title block ── */
.hdr { padding: 20px 0 10px 0; border-bottom: 1px solid #21262D; margin-bottom: 28px; }
.hdr h1 { font-size: 24px; font-weight: 700; color: #FAFAFA; margin: 0; letter-spacing: -0.5px; }
.hdr p { color: #8B949E; font-size: 13px; margin: 5px 0 0 0; }

/* ── KPI cards ── */
.kpi {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 16px 18px;
    min-height: 96px;
}
.kpi .label {
    font-size: 10px;
    color: #6E7681;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
    font-weight: 500;
}
.kpi .val {
    font-size: 20px;
    font-weight: 700;
    color: #E6EDF3;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}
.kpi .sub {
    font-size: 11px;
    color: #6E7681;
    margin-top: 5px;
    line-height: 1.4;
}

/* ── Charts ── */
.stPlotlyChart {
    border: 1px solid #21262D;
    border-radius: 8px;
    overflow: hidden;
    background: #161B22;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262D;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #161B22 !important;
    color: #8B949E !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid #21262D !important;
}
[data-testid="stDataFrame"] td {
    background: #0D1117 !important;
    color: #C9D1D9 !important;
    font-size: 13px !important;
    border-bottom: 1px solid #161B22 !important;
}
[data-testid="stDataFrame"] tr:hover td { background: #161B22 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #161B22;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 6px;
    color: #6E7681;
    font-weight: 500;
    font-size: 13px;
    padding: 0 22px;
    background: transparent;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #C9D1D9; background: #1C2128; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: #21262D; color: #E6EDF3; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 8px; border: 1px solid #21262D; }

/* ── Section headers ── */
h4 { color: #C9D1D9 !important; font-weight: 600; font-size: 15px !important; margin-bottom: 4px !important; }
.stCaption { color: #6E7681 !important; font-size: 11px !important; }

/* ── Spinner ── */
.stSpinner { color: #6C8EBF !important; }
</style>
"""


# =====================================================================
#  Main
# =====================================================================

def main():
    st.set_page_config(page_title="SPO Portfolio Optimizer", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    prices = load_prices()
    all_tickers = list(prices.columns)
    features_dict = compute_features(prices)

    # ── Sidebar: Configuration ──
    with st.sidebar:
        st.markdown('<div style="padding:12px 0 20px 0"><h2 style="margin:0;font-size:20px;font-weight:700;color:#FAFAFA">SPO Optimizer</h2>'
                    '<p style="color:#8B949E;font-size:12px;margin:4px 0 0 0">Portfolio Construction Platform</p></div>', unsafe_allow_html=True)
        st.markdown("---")

        # ── Prediction Mode ──
        st.markdown('<p style="color:#8B949E;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Prediction Mode</p>', unsafe_allow_html=True)
        live_mode = st.toggle(
            "Live Prediction (fetch today's prices)",
            value=False, key="live_mode",
            help="Fetches real-time prices from Yahoo Finance and optimizes for today. "
                 "Actual 5D return will be unavailable — this is a genuine forward-looking prediction."
        )

        st.markdown("---")
        st.markdown('<p style="color:#8B949E;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Model</p>', unsafe_allow_html=True)
        available_models = []
        for m in ["spo+", "mse", "hybrid"]:
            if os.path.exists(os.path.join(MODEL_DIR, f"pred_net_{m}.pt")):
                available_models.append(m)
        model_labels = {"spo+": "SPO+  (Decision-Focused)", "mse": "MSE  (Prediction-Focused)", "hybrid": "Hybrid  (Blended)"}
        selected_model = st.selectbox("Training regime", available_models,
                                       format_func=lambda m: model_labels.get(m, m), key="model_sel")

        st.markdown("---")
        st.markdown('<p style="color:#8B949E;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Risk Parameters</p>', unsafe_allow_html=True)

        gamma = st.slider("Risk aversion (gamma)", min_value=0.05, max_value=5.0, value=0.5, step=0.05,
                          help="Higher = more conservative, lower risk allocation")
        max_weight = st.slider("Max weight per asset", min_value=0.02, max_value=0.50, value=0.10, step=0.01,
                               help="Concentration limit for any single position")

        st.markdown("---")
        st.markdown('<p style="color:#8B949E;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Universe</p>', unsafe_allow_html=True)

        sectors = sorted(set(SECTOR_MAP.get(t, "Other") for t in all_tickers))
        selected_sectors = st.multiselect("Filter by sector", sectors, default=sectors, key="sector_filter")
        filtered_tickers = [t for t in all_tickers if SECTOR_MAP.get(t, "Other") in selected_sectors]

        selected_tickers = st.multiselect("Select tickers", filtered_tickers, default=filtered_tickers, key="ticker_sel")

        if len(selected_tickers) < 2:
            st.warning("Select at least 2 tickers.")
            st.stop()

        st.markdown("---")
        if not live_mode:
            st.markdown('<p style="color:#8B949E;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Optimization Date</p>', unsafe_allow_html=True)
            available_dates = prices.index[60:]
            date_options = [d.strftime("%Y-%m-%d") for d in available_dates[-252:]]
            default_idx = max(0, len(date_options) - 10)
            opt_date_str = st.selectbox("Date", date_options, index=default_idx, key="opt_date")
            opt_date = pd.Timestamp(opt_date_str)
        else:
            opt_date_str = pd.Timestamp.today().strftime("%Y-%m-%d")
            opt_date = None  # signals live mode downstream

        st.markdown("---")
        st.markdown('<p style="color:#8B949E;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Backtest</p>', unsafe_allow_html=True)
        run_backtest = st.checkbox("Run walk-forward backtest", value=False, key="run_bt",
                                   disabled=live_mode, help="Backtest uses historical data only.")
        bt_days = 60  # default; overridden by slider when backtest is enabled
        if run_backtest and not live_mode:
            bt_days = st.slider("Backtest window (trading days)", 20, 252, 60, key="bt_days")

    # ── Header ──
    if live_mode:
        st.markdown(
            '<div class="hdr"><h1>Portfolio Optimization — Live Prediction</h1>'
            f'<p>Forward-looking optimization using today\'s market data &nbsp;|&nbsp; '
            f'Fetched: <strong>{opt_date_str}</strong> &nbsp;|&nbsp; '
            'Actual 5-day return will be available in 5 trading days</p></div>',
            unsafe_allow_html=True
        )
        st.info(
            "Live mode: prices fetched from Yahoo Finance right now. "
            "The model ranks stocks by its learned signal and solves for the optimal Markowitz portfolio. "
            "No realised return is shown because we are predicting the future."
        )
    else:
        st.markdown('<div class="hdr"><h1>Portfolio Optimization</h1><p>Predict expected returns and solve for optimal portfolio weights via differentiable Markowitz optimization</p></div>', unsafe_allow_html=True)

    # ── Load model ──
    net = load_model(selected_model, n_features=10, n_assets=len(all_tickers))
    if net is None:
        st.error(f"Model file not found for mode '{selected_model}'. Run the SPO training pipeline first.")
        st.stop()

    from sklearn.covariance import LedoitWolf

    # ══════════════════════════════════════════════════════════════════
    #  LIVE MODE  — fetch current prices & compute features on the fly
    # ══════════════════════════════════════════════════════════════════
    if live_mode:
        with st.spinner("Fetching live prices from Yahoo Finance..."):
            live_prices = fetch_live_prices(tuple(all_tickers), lookback_days=120)

        # Keep only tickers that came back successfully
        live_tickers_ok = [t for t in all_tickers if t in live_prices.columns]
        missing = set(all_tickers) - set(live_tickers_ok)
        if missing:
            st.warning(f"Could not fetch data for: {', '.join(sorted(missing))}. These tickers will be excluded.")

        live_features = compute_features(live_prices)
        live_date = live_prices.index[-1]          # most recent trading day
        opt_date_str = live_date.strftime("%Y-%m-%d")

        # Build feature matrix for all (available) tickers on live_date
        full_feat = build_feature_matrix(live_features, live_tickers_ok, live_date)

        # Covariance from last 60 live trading days
        prices_subset_live = live_prices[[t for t in selected_tickers if t in live_prices.columns]]
        selected_tickers_live = list(prices_subset_live.columns)
        window_returns = prices_subset_live.pct_change().iloc[-60:].dropna(how="any")

        if len(window_returns) < 20:
            st.error("Not enough live return history for covariance estimation.")
            st.stop()

        lw = LedoitWolf().fit(window_returns.values)
        cov_matrix = lw.covariance_
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov_matrix = ((cov_matrix + cov_matrix.T) / 2).astype(np.float32)

        # Predict for all live tickers, slice to selected
        live_ticker_to_idx = {t: i for i, t in enumerate(live_tickers_ok)}
        full_feat_tensor = torch.tensor(full_feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # Model expects exactly 64 tickers; pad/slice as needed
            n_live = len(live_tickers_ok)
            n_model = len(all_tickers)
            if n_live < n_model:
                pad = np.zeros((1, n_model - n_live, full_feat.shape[1]), dtype=np.float32)
                feat_padded = np.concatenate([full_feat[np.newaxis], pad], axis=1)
                full_preds = net(torch.tensor(feat_padded)).squeeze(0).numpy()[:n_live]
            else:
                full_preds = net(full_feat_tensor).squeeze(0).numpy()

        sel_live_indices = [live_ticker_to_idx[t] for t in selected_tickers_live]
        predicted_returns = full_preds[sel_live_indices]
        selected_tickers = selected_tickers_live   # use only what was fetched
        n_assets = len(selected_tickers)
        has_actual = False
        actual_5d_returns = None
        actual_port_return = None

    # ══════════════════════════════════════════════════════════════════
    #  HISTORICAL MODE  — use dataset prices
    # ══════════════════════════════════════════════════════════════════
    else:
        prices_subset = prices[selected_tickers]
        cov_window = 60
        loc = prices.index.get_loc(opt_date)
        window_returns = prices_subset.pct_change().iloc[loc - cov_window:loc].dropna(how="any")

        if len(window_returns) < 30:
            st.error("Not enough return history for covariance estimation at this date.")
            st.stop()

        lw = LedoitWolf().fit(window_returns.values)
        cov_matrix = lw.covariance_
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov_matrix = ((cov_matrix + cov_matrix.T) / 2).astype(np.float32)

        full_feat = build_feature_matrix(features_dict, all_tickers, opt_date)
        full_feat_tensor = torch.tensor(full_feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            full_preds = net(full_feat_tensor).squeeze(0).numpy()

    # Map predictions to selected tickers (historical mode only;
    # live mode already set predicted_returns inside its branch)
    if not live_mode:
        ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
        sel_indices = [ticker_to_idx[t] for t in selected_tickers]
        predicted_returns = full_preds[sel_indices]

    # ── FIX #3: Cross-sectional z-score normalisation of predictions ──
    # SPO+ model outputs are rank-based and near-constant in absolute magnitude
    # (~-19%). Only the cross-sectional RANKING matters for Markowitz, not the
    # absolute value.  Feeding near-constant mu into the QP causes numerical
    # instability (all assets look identical to the solver).
    #
    # Solution: normalise predictions cross-sectionally BEFORE optimisation.
    # z = (pred - mean) / (std + eps)  preserves ranking, gives the QP a
    # well-conditioned signal spread, and prevents division by zero.
    pred_mean = predicted_returns.mean()
    pred_std  = predicted_returns.std() + 1e-9                        # eps guards div-by-zero
    pred_zscore = (predicted_returns - pred_mean) / pred_std          # cross-sectional z-score
    pred_rank   = pd.Series(predicted_returns).rank(ascending=True).values  # 1 = weakest signal

    # ── Optimize for the user's selected universe ──
    # FIX #3 cont.: Feed the z-scored predictions (not raw) into the portfolio
    # layer so the solver sees a stable, well-spread signal vector.
    n_assets = len(selected_tickers)
    port_layer = DifferentiableMarkowitz(n_assets=n_assets, gamma=gamma, max_weight=max_weight)
    pred_tensor = torch.tensor(pred_zscore, dtype=torch.float32)      # <-- normalised signal
    cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32)

    with torch.no_grad():
        weights = port_layer(pred_tensor, cov_tensor).numpy()

    # ── Build results dataframe ──
    portfolio_df = pd.DataFrame({
        "Ticker": selected_tickers,
        "Sector": [SECTOR_MAP.get(t, "Other") for t in selected_tickers],
        "Weight": weights,
        "Signal (z-score)": pred_zscore,
        "Rank": pred_rank.astype(int),
        "_raw_pred": predicted_returns,   # kept for reference, hidden from display
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    # ── Portfolio-level stats ──
    # FIX #2: Ensure consistent scaling.  `window_returns` is computed from
    # daily pct_change, so `hist_port_ret` is in DAILY units.  `cov_matrix`
    # is also estimated from daily returns (Ledoit-Wolf on daily pct_change).
    # All annualisation uses 252 trading days per year.
    hist_ret_col = window_returns  # shape (window, n_assets) — DAILY returns
    hist_port_ret = (hist_ret_col * weights).sum(axis=1)   # portfolio DAILY returns

    hist_mean_daily = float(hist_port_ret.mean())           # mean daily return
    hist_vol_daily  = float(hist_port_ret.std())            # daily volatility

    # Annualise: ret * 252, vol * sqrt(252)
    hist_mean_annual = hist_mean_daily * 252
    hist_vol_annual  = hist_vol_daily * np.sqrt(252)
    port_sharpe_est = (hist_mean_annual / hist_vol_annual) if hist_vol_annual > 0 else 0.0

    # Portfolio volatility from covariance matrix (DAILY scale from Ledoit-Wolf)
    port_variance_daily = float(weights @ cov_matrix @ weights)
    port_vol_daily = float(np.sqrt(port_variance_daily))
    port_vol_annual = port_vol_daily * np.sqrt(252)         # annualised volatility

    active_positions = int((weights > 0.001).sum())
    top5_concentration = float(np.sort(weights)[-min(5, n_assets):].sum())
    top_signal_ticker = portfolio_df.iloc[0]["Ticker"] if len(portfolio_df) > 0 else ""

    # FIX #5 (BONUS): Sanity checks for unrealistic values
    if hist_mean_annual > 0.50:
        st.warning(f"⚠️ Estimated annual return ({hist_mean_annual*100:.1f}%) exceeds 50% — verify input data scaling.")
    if hist_vol_annual > 0 and hist_vol_annual < 0.01:
        st.warning(f"⚠️ Estimated annual volatility ({hist_vol_annual*100:.2f}%) is below 1% — unusually low.")

    # \u2500\u2500 Actual (forward) return \u2500\u2500
    # Live mode: already set has_actual=False in the live branch above.
    # Historical mode: check if +5 days exists in the dataset.
    if not live_mode:
        if loc + 5 < len(prices):
            actual_5d_returns = (prices_subset.iloc[loc + 5] / prices_subset.iloc[loc] - 1).values.astype(np.float32)
            actual_port_return = float(np.dot(weights, actual_5d_returns))
            has_actual = True
        else:
            actual_5d_returns = None
            actual_port_return = None
            has_actual = False

    # =====================================================================
    #  TAB LAYOUT
    # =====================================================================

    tabs = st.tabs(["Optimize", "Analysis", "Backtest"])

    # ─────────────────────────────────────────────────────────────────
    #  TAB 1: OPTIMIZE — the core portfolio output
    # ─────────────────────────────────────────────────────────────────
    with tabs[0]:

        # KPI row
        cols = st.columns(6)
        # FIX #2: KPIs now display ANNUALISED volatility for consistency.
        kpi_data = [
            ("Realised 5D Return",
             f"{actual_port_return*100:+.3f}%" if has_actual else "N/A",
             "Actual portfolio return" if has_actual else "Future date — not yet available"),
            ("Annual Volatility", f"{port_vol_annual*100:.2f}%", "Annualised daily vol (√252)"),
            ("Est. Sharpe", f"{port_sharpe_est:.2f}", "Historical 60d annualised"),
            ("Active Positions", f"{active_positions}", f"of {n_assets} assets selected"),
            ("Top-5 Weight", f"{top5_concentration:.1%}", "Concentration"),
            ("Top Signal", top_signal_ticker, "Highest-ranked holding"),
        ]
        for i, (label, val, sub) in enumerate(kpi_data):
            with cols[i]:
                color = ""
                if label == "Realised 5D Return" and val != "N/A":
                    color = f"color:{'#6CCE6C' if '+' in val else '#E06C6C'}"
                st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="val" style="{color}">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Weight allocation chart + table ──
        c1, c2 = st.columns([3, 2])
        with c1:
            active = portfolio_df[portfolio_df["Weight"] > 0.001].copy()
            # Color by sector
            active["Color"] = active["Sector"].map(SECTOR_COLORS).fillna("#808080")

            fig = go.Figure()
            for sector in active["Sector"].unique():
                sec_df = active[active["Sector"] == sector]
                fig.add_trace(go.Bar(
                    x=sec_df["Ticker"], y=sec_df["Weight"],
                    name=sector, marker_color=SECTOR_COLORS.get(sector, "#808080"),
                    hovertemplate="%{x}<br>Weight: %{y:.2%}<extra>" + sector + "</extra>",
                ))
            fig.update_yaxes(tickformat=".1%", title_text="Weight")
            fig.update_layout(barmode="stack", showlegend=True)
            _dark_layout(fig, f"Optimal Portfolio Weights  |  {opt_date_str}", height=400)
            st.plotly_chart(fig, use_container_width=True, key="opt_weights_bar")

        with c2:
            # Allocation pie by sector
            sector_alloc = portfolio_df.groupby("Sector")["Weight"].sum().reset_index()
            sector_alloc = sector_alloc[sector_alloc["Weight"] > 0.001]
            colors_list = [SECTOR_COLORS.get(s, "#808080") for s in sector_alloc["Sector"]]

            fig_pie = go.Figure(data=go.Pie(
                labels=sector_alloc["Sector"], values=sector_alloc["Weight"],
                marker=dict(colors=colors_list, line=dict(color=C["bg"], width=2)),
                textinfo="label+percent", textfont=dict(size=12),
                hovertemplate="%{label}<br>Weight: %{value:.2%}<extra></extra>",
                hole=0.45,
            ))
            _dark_layout(fig_pie, "Sector Allocation", height=400)
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True, key="opt_sector_pie")

        # ── Actual returns chart (when available) ── 
        if has_actual:
            st.markdown("#### Realised 5-Day Returns vs Model Signal")
            st.caption("Bars show actual 5-day return for each active holding. Colour: green = positive, red = negative. Line = portfolio average.")

            active_tickers_list = portfolio_df[portfolio_df["Weight"] > 0.001]["Ticker"].tolist()
            actual_map_full = dict(zip(selected_tickers, actual_5d_returns))
            act_vals = [actual_map_full.get(t, 0.0) for t in active_tickers_list]
            bar_colors = [C["green"] if v >= 0 else C["red"] for v in act_vals]

            fig_act = go.Figure()
            fig_act.add_trace(go.Bar(
                x=active_tickers_list,
                y=[v * 100 for v in act_vals],
                marker_color=bar_colors,
                name="Actual 5D Return",
                hovertemplate="%{x}<br>Return: %{y:+.3f}%<extra></extra>",
            ))
            fig_act.add_hline(
                y=actual_port_return * 100,
                line_dash="dash", line_color=C["gold"], line_width=2,
                annotation_text=f"Portfolio avg: {actual_port_return*100:+.3f}%",
                annotation_font_color=C["gold"],
            )
            fig_act.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig_act.update_yaxes(title_text="5-Day Return (%)", tickformat="+.2f")
            _dark_layout(fig_act, height=320)
            st.plotly_chart(fig_act, use_container_width=True, key="opt_actual_returns")

        # ── Detailed holdings table ──
        st.markdown("#### Holdings Detail")
        st.caption("Signal z-score: cross-sectional standardised model score (higher = stronger buy signal). Rank is out of selected universe.")
        display_df = portfolio_df[portfolio_df["Weight"] > 0.0005].copy()
        display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.2%}")
        display_df["Signal (z-score)"] = display_df["Signal (z-score)"].apply(lambda x: f"{x:+.3f}")
        display_df["Rank"] = display_df["Rank"].apply(lambda x: f"{int(x)} / {n_assets}")
        display_df = display_df.drop(columns=["_raw_pred"])
        if has_actual:
            display_df["Actual 5D Return"] = display_df["Ticker"].map(actual_map_full).apply(
                lambda x: f"{x*100:+.3f}%" if pd.notna(x) else ""
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True, key="opt_holdings_table")

    # ─────────────────────────────────────────────────────────────────
    #  TAB 2: ANALYSIS — risk decomposition & signal diagnostics
    # ─────────────────────────────────────────────────────────────────
    with tabs[1]:

        c1, c2 = st.columns(2)

        with c1:
            # Signal z-score vs weight scatter
            fig_scatter = go.Figure()
            for sector in portfolio_df["Sector"].unique():
                sec = portfolio_df[portfolio_df["Sector"] == sector]
                fig_scatter.add_trace(go.Scatter(
                    x=sec["Signal (z-score)"], y=sec["Weight"] * 100,
                    mode="markers+text",
                    text=sec["Ticker"],
                    textposition="top center",
                    textfont=dict(size=9, color=C["muted"]),
                    marker=dict(size=10, color=SECTOR_COLORS.get(sector, "#808080"),
                                line=dict(width=1, color="#fff")),
                    name=sector,
                    hovertemplate="%{text}<br>Signal z-score: %{x:.3f}<br>Weight: %{y:.2f}%<extra></extra>",
                ))
            fig_scatter.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig_scatter.update_xaxes(title_text="Model Signal (cross-sectional z-score)")
            fig_scatter.update_yaxes(title_text="Portfolio Weight (%)")
            _dark_layout(fig_scatter, "Signal vs Allocation", height=420)
            st.plotly_chart(fig_scatter, use_container_width=True, key="ana_scatter")

        with c2:
            # Marginal risk contribution
            w = weights.copy()
            sigma_w = cov_matrix @ w
            total_risk = np.sqrt(w @ sigma_w)
            mcr = sigma_w / total_risk if total_risk > 0 else sigma_w
            risk_contrib = w * mcr
            risk_pct = risk_contrib / risk_contrib.sum() if risk_contrib.sum() > 0 else risk_contrib

            risk_df = pd.DataFrame({
                "Ticker": selected_tickers, "Weight": w,
                "Risk Contribution": risk_pct,
            }).sort_values("Risk Contribution", ascending=False).head(15)

            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                x=risk_df["Ticker"], y=risk_df["Risk Contribution"],
                marker_color=C["red"], name="Risk Contribution",
                hovertemplate="%{x}: %{y:.2%}<extra></extra>",
            ))
            fig_risk.add_trace(go.Bar(
                x=risk_df["Ticker"], y=risk_df["Weight"],
                marker_color=C["blue"], name="Weight",
                hovertemplate="%{x}: %{y:.2%}<extra></extra>",
            ))
            fig_risk.update_yaxes(tickformat=".1%")
            fig_risk.update_layout(barmode="group")
            _dark_layout(fig_risk, "Risk Contribution vs Weight (Top 15)", height=420)
            st.plotly_chart(fig_risk, use_container_width=True, key="ana_risk_contrib")

        # Covariance heatmap for top positions
        st.markdown("#### Correlation Matrix (Active Holdings)")
        active_idx = np.where(weights > 0.001)[0]
        if len(active_idx) > 1:
            active_tickers = [selected_tickers[i] for i in active_idx]
            sub_cov = cov_matrix[np.ix_(active_idx, active_idx)]
            std_devs = np.sqrt(np.diag(sub_cov))
            corr = sub_cov / np.outer(std_devs, std_devs)
            np.fill_diagonal(corr, 1.0)

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr, x=active_tickers, y=active_tickers,
                colorscale=[[0, C["blue"]], [0.5, C["bg"]], [1, C["red"]]],
                zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title="Corr"),
                hovertemplate="%{x} / %{y}<br>Corr: %{z:.3f}<extra></extra>",
                texttemplate="%{z:.2f}", textfont=dict(size=9),
            ))
            _dark_layout(fig_corr, height=max(350, len(active_tickers) * 22))
            fig_corr.update_yaxes(dtick=1)
            fig_corr.update_xaxes(dtick=1)
            st.plotly_chart(fig_corr, use_container_width=True, key="ana_corr")

        # Efficient frontier sketch & Optimization Analytics
        st.markdown("#### Efficient Frontier & Portfolio Optimization")

        from spo.efficient_frontier import (
            annualize_returns_and_cov,
            shrink_mean_returns,
            simulate_portfolios, 
            optimize_portfolio, 
            compute_efficient_frontier, 
            plot_results,
            compute_portfolio_performance
        )
        
        # Scale pipeline: DAILY → ANNUAL.
        # `window_returns` = daily pct_change, `cov_matrix` = Ledoit-Wolf on daily.
        # Annualise:  ret_annual = ret_daily × 252,  cov_annual = cov_daily × 252.
        hist_mean_rets = window_returns.mean().values.astype(np.float32)
        ann_rets_raw, ann_cov = annualize_returns_and_cov(hist_mean_rets, cov_matrix, trading_days=252)

        # Shrink mean return estimates to reduce estimation noise.
        # With T≈60 daily observations, individual stock annualised return
        # estimates have standard errors of ~60% — they are noise-dominated.
        # Shrinkage toward the cross-sectional mean is standard practice
        # (Jorion 1986, Michaud 1989).
        n_obs = len(window_returns)
        ann_rets, shrink_intensity = shrink_mean_returns(ann_rets_raw, n_obs)

        st.caption(
            f"Mean return shrinkage: {shrink_intensity:.0%} toward grand mean "
            f"(estimation window: {n_obs} days).  "
            f"All portfolios constrained to max {max_weight:.0%} per asset."
        )
        
        with st.spinner("Generating Monte Carlo Portfolios & Efficient Frontier..."):
            # All frontier computations use the SAME max_weight constraint
            # as the SPO portfolio for a fair visual comparison.
            _, sim_rets, sim_vols, sim_sharpes = simulate_portfolios(
                ann_rets, ann_cov, num_portfolios=5000, max_weight=max_weight)

            # Max Sharpe and Min Volatility under same constraints
            _, opt_ret, opt_vol, _ = optimize_portfolio(
                ann_rets, ann_cov, target="sharpe", max_weight=max_weight)
            _, min_vol_ret, min_vol_vol, _ = optimize_portfolio(
                ann_rets, ann_cov, target="volatility", max_weight=max_weight)

            # Equal Weight benchmark (always feasible if max_weight ≥ 1/n)
            eq_w = np.full(n_assets, 1.0 / n_assets)
            eq_ret, eq_vol, _ = compute_portfolio_performance(eq_w, ann_rets, ann_cov)
            
            # SPO weights evaluated on the SAME shrunk/annualised inputs
            spo_ret, spo_vol, _ = compute_portfolio_performance(weights, ann_rets, ann_cov)

            # Efficient Frontier curve (constrained)
            ef_vols, ef_rets = compute_efficient_frontier(
                ann_rets, ann_cov, num_points=25, max_weight=max_weight)

        # Plot complete visual
        fig_ef = plot_results(
            sim_vols, sim_rets, sim_sharpes,
            opt_vol, opt_ret,
            min_vol_vol, min_vol_ret,
            ef_vols, ef_rets,
            spo_vol=spo_vol, spo_ret=spo_ret,
            eq_vol=eq_vol, eq_ret=eq_ret
        )
        
        _dark_layout(fig_ef, "Efficient Frontier — Annualised Risk vs Return", height=500)
        st.plotly_chart(fig_ef, use_container_width=True, key="ana_frontier")

    # ─────────────────────────────────────────────────────────────────
    #  TAB 3: BACKTEST — walk-forward simulation
    # ─────────────────────────────────────────────────────────────────
    with tabs[2]:

        if live_mode:
            st.info("Backtesting is not available in Live Prediction mode. Disable 'Live Prediction' to run historical simulations.")
            st.stop()

        if not run_backtest:
            st.info("Enable 'Run walk-forward backtest' in the sidebar to see historical performance.")
            st.stop()

        st.markdown(f"#### Walk-Forward Backtest  |  {bt_days} trading days ending {opt_date_str}")
        st.caption(f"Model: {model_labels[selected_model]}  |  Gamma: {gamma}  |  Max Weight: {max_weight:.0%}  |  Universe: {len(selected_tickers)} assets")

        with st.spinner("Running backtest..."):
            bt_start = max(0, loc - bt_days)
            bt_dates = prices.index[bt_start:loc + 1]

            bt_returns = []
            bt_oracle = []
            bt_eq = []
            bt_weights_list = []
            bt_date_labels = []
            bt_ic_list = []

            for bi in range(len(bt_dates) - 5):
                d = bt_dates[bi]
                d_str = d.strftime("%Y-%m-%d")

                # Features
                d_loc = prices.index.get_loc(d)
                if d_loc < cov_window or d_loc + 5 >= len(prices):
                    continue

                # Covariance
                w_rets = prices_subset.pct_change().iloc[d_loc - cov_window:d_loc].dropna(how="any")
                if len(w_rets) < 30:
                    continue
                lw_bt = LedoitWolf().fit(w_rets.values)
                cov_bt = lw_bt.covariance_.astype(np.float32)
                eigv, eigvec = np.linalg.eigh(cov_bt)
                eigv = np.maximum(eigv, 1e-8)
                cov_bt = ((eigvec @ np.diag(eigv) @ eigvec.T + (eigvec @ np.diag(eigv) @ eigvec.T).T) / 2).astype(np.float32)

                # Predict (all 64) then slice
                feat_bt = build_feature_matrix(features_dict, all_tickers, d)
                feat_t = torch.tensor(feat_bt, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred_full = net(feat_t).squeeze(0).numpy()
                pred_sel = pred_full[sel_indices]

                # FIX #3 (backtest): Apply cross-sectional z-score normalisation
                # before optimisation, same as in the main branch.
                bt_pred_mean = pred_sel.mean()
                bt_pred_std  = pred_sel.std() + 1e-9
                pred_sel_z   = (pred_sel - bt_pred_mean) / bt_pred_std

                # Optimize using normalised predictions
                pl = DifferentiableMarkowitz(n_assets=n_assets, gamma=gamma, max_weight=max_weight)
                with torch.no_grad():
                    w_bt = pl(torch.tensor(pred_sel_z), torch.tensor(cov_bt)).numpy()

                # Actual returns
                actual_bt = (prices_subset.iloc[d_loc + 5] / prices_subset.iloc[d_loc] - 1).values.astype(np.float32)

                # Portfolio return
                port_ret = float(np.dot(w_bt, actual_bt))

                # Oracle
                with torch.no_grad():
                    w_oracle = pl(torch.tensor(actual_bt), torch.tensor(cov_bt)).numpy()
                oracle_ret = float(np.dot(w_oracle, actual_bt))

                # Equal weight
                eq_ret_bt = float(actual_bt.mean())

                # IC
                try:
                    ic, _ = spearmanr(pred_sel, actual_bt)
                except:
                    ic = 0.0

                bt_returns.append(port_ret)
                bt_oracle.append(oracle_ret)
                bt_eq.append(eq_ret_bt)
                bt_weights_list.append(w_bt)
                bt_date_labels.append(d_str)
                bt_ic_list.append(ic if not np.isnan(ic) else 0.0)

        if len(bt_returns) < 5:
            st.warning("Not enough valid backtest dates. Try a longer window or check data availability.")
            st.stop()

        bt_returns = np.array(bt_returns)
        bt_oracle = np.array(bt_oracle)
        bt_eq = np.array(bt_eq)

        # Backtest KPIs
        # FIX #2 (backtest): Correct annualisation for 5-day returns.
        # Each backtest period is a 5-day (weekly) return, so there are
        # ~252/5 = 50.4 such periods per year.  Annualise accordingly:
        #   vol_annual  = vol_5d * sqrt(252/5)
        #   mean_annual = mean_5d * (252/5)
        #   sharpe      = mean_annual / vol_annual = mean_5d / vol_5d * sqrt(252/5)
        periods_per_year = 252.0 / 5.0  # ~50.4 five-day periods per year
        bt_cum = float(np.prod(1 + bt_returns) - 1)
        bt_vol = float(bt_returns.std() * np.sqrt(periods_per_year))   # annualised vol
        bt_sharpe = float(bt_returns.mean() / bt_returns.std() * np.sqrt(periods_per_year)) if bt_returns.std() > 0 else 0.0
        bt_max_dd = float(((np.maximum.accumulate(np.cumprod(1 + bt_returns)) - np.cumprod(1 + bt_returns)) / np.maximum.accumulate(np.cumprod(1 + bt_returns))).max())
        bt_eq_cum = float(np.prod(1 + bt_eq) - 1)
        bt_mean_ic = float(np.nanmean(bt_ic_list))
        bt_regret = float(np.mean(bt_oracle - bt_returns))

        kc = st.columns(6)
        bt_kpis = [
            ("Cumulative Return", f"{bt_cum*100:+.2f}%", f"{len(bt_returns)} periods"),
            ("Sharpe Ratio", f"{bt_sharpe:.3f}", "Annualised"),
            ("Volatility", f"{bt_vol*100:.2f}%", "Annualised"),
            ("Max Drawdown", f"{bt_max_dd*100:.2f}%", "Peak to trough"),
            ("Mean IC", f"{bt_mean_ic:.4f}", "Spearman rank"),
            ("EW Benchmark", f"{bt_eq_cum*100:+.2f}%", "Equal-weight cum"),
        ]
        for i, (label, val, sub) in enumerate(bt_kpis):
            with kc[i]:
                color = ""
                if "Return" in label or "Benchmark" in label:
                    color = f"color:{'#6CCE6C' if '+' in val else '#E06C6C'}"
                st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="val" style="{color}">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Normalised equity curves (base = 100) so all three lines are visible
        eq_curve_port   = 100 * np.cumprod(1 + bt_returns)
        eq_curve_ew     = 100 * np.cumprod(1 + bt_eq)
        eq_curve_oracle = 100 * np.cumprod(1 + bt_oracle)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_date_labels, y=eq_curve_port, name="SPO Portfolio",
            line=dict(color=C["gold"], width=2.5),
            hovertemplate="%{x}<br>Index: %{y:.2f}<extra>SPO</extra>",
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_date_labels, y=eq_curve_ew, name="Equal Weight",
            line=dict(color=C["muted"], width=1.5, dash="dot"),
            hovertemplate="%{x}<br>Index: %{y:.2f}<extra>EW</extra>",
        ))
        fig_bt.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.15)")
        # Shade area between SPO and EW
        fig_bt.add_trace(go.Scatter(
            x=bt_date_labels, y=eq_curve_port,
            fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False,
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_date_labels, y=eq_curve_ew, mode="lines",
            fill="tonexty",
            fillcolor="rgba(212,160,60,0.08)",
            line_color="rgba(0,0,0,0)", showlegend=False,
        ))
        fig_bt.update_yaxes(title_text="Portfolio Value (base = 100)")
        _dark_layout(fig_bt, "Equity Curve — SPO vs Equal Weight (base 100)", height=400)
        st.plotly_chart(fig_bt, use_container_width=True, key="bt_cum")

        # Period-by-period return bar: SPO vs EW
        c1, c2 = st.columns(2)
        with c1:
            bar_colors_spo = [C["green"] if r >= 0 else C["red"] for r in bt_returns]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bt_date_labels, y=bt_returns * 100,
                marker_color=bar_colors_spo, name="SPO 5D Return",
                hovertemplate="%{x}<br>%{y:+.2f}%<extra>SPO</extra>",
            ))
            fig_bar.add_trace(go.Scatter(
                x=bt_date_labels, y=np.array(bt_eq) * 100,
                mode="lines", name="EW Return",
                line=dict(color=C["muted"], width=1, dash="dot"),
                hovertemplate="%{x}<br>%{y:+.2f}%<extra>EW</extra>",
            ))
            fig_bar.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig_bar.update_yaxes(tickformat="+.1f", title_text="5-Day Return (%)")
            _dark_layout(fig_bar, "Period Returns — SPO vs Equal Weight", height=320)
            st.plotly_chart(fig_bar, use_container_width=True, key="bt_period_bar")

        with c2:
            # Drawdown from equity curve high
            dd = (np.maximum.accumulate(eq_curve_port) - eq_curve_port) / np.maximum.accumulate(eq_curve_port)
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bt_date_labels, y=-dd * 100, mode="lines",
                fill="tozeroy", line=dict(color=C["red"], width=1.5),
                fillcolor="rgba(224,108,108,0.15)", name="Drawdown",
                hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
            ))
            fig_dd.update_yaxes(tickformat=".1f", title_text="Drawdown (%)")
            _dark_layout(fig_dd, "Underwater Equity (Drawdown)", height=320)
            st.plotly_chart(fig_dd, use_container_width=True, key="bt_dd")

        # Rolling IC
        ic_smooth = pd.Series(bt_ic_list).rolling(10, min_periods=3).mean().values
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(
            x=bt_date_labels, y=ic_smooth, mode="lines",
            line=dict(color=C["teal"], width=2), name="IC (10-day MA)",
            hovertemplate="%{x}<br>IC: %{y:.4f}<extra></extra>",
        ))
        fig_ic.add_trace(go.Bar(
            x=bt_date_labels, y=bt_ic_list,
            marker_color=[C["teal"] if v >= 0 else C["red"] for v in bt_ic_list],
            opacity=0.25, name="Raw IC",
        ))
        fig_ic.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig_ic.update_yaxes(title_text="Spearman Rank IC", range=[-0.6, 0.6])
        _dark_layout(fig_ic, "Rolling Information Coefficient (IC)", height=320)
        st.plotly_chart(fig_ic, use_container_width=True, key="bt_ic")

        # Oracle in its own section — separate Y-axis so it doesn't destroy the main chart
        with st.expander("Oracle Benchmark (Perfect Foresight)", expanded=False):
            st.caption(
                "Oracle uses actual 5-day returns as predictions — it has perfect foresight. "
                "Shown separately because its compounded return dwarfs any realistic strategy. "
                "Regret = Oracle − SPO per period."
            )
            c3, c4 = st.columns(2)
            with c3:
                fig_oracle = go.Figure()
                fig_oracle.add_trace(go.Scatter(
                    x=bt_date_labels, y=eq_curve_oracle, name="Oracle",
                    line=dict(color=C["purple"], width=2),
                    hovertemplate="%{x}<br>Index: %{y:.1f}<extra>Oracle</extra>",
                ))
                fig_oracle.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.15)")
                fig_oracle.update_yaxes(title_text="Portfolio Value (base 100)")
                _dark_layout(fig_oracle, "Oracle Equity Curve", height=280)
                st.plotly_chart(fig_oracle, use_container_width=True, key="bt_oracle_curve")
            with c4:
                regret_per_period = (np.array(bt_oracle) - bt_returns) * 100
                fig_regret = go.Figure(go.Bar(
                    x=bt_date_labels, y=regret_per_period,
                    marker_color=[C["red"] if v > 0 else C["green"] for v in regret_per_period],
                    hovertemplate="%{x}<br>Regret: %{y:+.2f}%<extra></extra>",
                ))
                fig_regret.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
                fig_regret.update_yaxes(title_text="Regret per period (%)")
                _dark_layout(fig_regret, "Decision Regret per Period", height=280)
                st.plotly_chart(fig_regret, use_container_width=True, key="bt_regret")

        # Regret note
        st.caption(
            f"Mean per-period regret (Oracle − SPO): {bt_regret*100:+.4f}% &nbsp;|&nbsp; "
            f"This measures how much return is left on the table per period vs perfect foresight. "
            f"Lower regret = model decisions are closer to optimal."
        )


if __name__ == "__main__":
    main()
