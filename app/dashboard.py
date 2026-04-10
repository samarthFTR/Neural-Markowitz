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
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=C["text"]),
        title=dict(text=title, font=dict(size=15, color=C["text"]), x=0) if title else None,
        height=height,
        margin=dict(l=50, r=20, t=40 if title else 10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
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

/* ── Base ── */
html, body, [class*="st-"] { font-family: 'Inter', system-ui, sans-serif !important; }
.stApp { background: #0E1117; }

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

    # ── Cross-sectional z-score & rank of predictions ──
    # NOTE: SPO+ model outputs near-constant absolute values (~-19%) because
    # only the cross-sectional ranking matters for portfolio weights, not the
    # absolute magnitude. We display z-scores and ranks instead of raw values.
    pred_mean = predicted_returns.mean()
    pred_std  = predicted_returns.std() + 1e-9
    pred_zscore = (predicted_returns - pred_mean) / pred_std          # cross-sectional z-score
    pred_rank   = pd.Series(predicted_returns).rank(ascending=True).values  # 1 = weakest signal

    # ── Optimize for the user's selected universe ──
    n_assets = len(selected_tickers)
    port_layer = DifferentiableMarkowitz(n_assets=n_assets, gamma=gamma, max_weight=max_weight)
    pred_tensor = torch.tensor(predicted_returns, dtype=torch.float32)
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
        "_raw_pred": predicted_returns,   # kept for optimizer, hidden from display
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    # ── Portfolio-level stats ──
    # Use historical mean/vol from the covariance window for a meaningful Sharpe estimate
    hist_ret_col = window_returns  # shape (window, n_assets)
    hist_port_ret = (hist_ret_col * weights).sum(axis=1)   # portfolio daily returns
    ann_factor = np.sqrt(252 / 5)
    hist_mean = float(hist_port_ret.mean())
    hist_vol  = float(hist_port_ret.std())
    port_sharpe_est = (hist_mean / hist_vol * np.sqrt(252)) if hist_vol > 0 else 0.0

    port_variance = float(weights @ cov_matrix @ weights)
    port_vol = float(np.sqrt(port_variance))   # 5-day std dev from Ledoit-Wolf
    active_positions = int((weights > 0.001).sum())
    top5_concentration = float(np.sort(weights)[-min(5, n_assets):].sum())
    top_signal_ticker = portfolio_df.iloc[0]["Ticker"] if len(portfolio_df) > 0 else ""

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
        kpi_data = [
            ("Realised 5D Return",
             f"{actual_port_return*100:+.3f}%" if has_actual else "N/A",
             "Actual portfolio return" if has_actual else "Future date — not yet available"),
            ("Portfolio Risk", f"{port_vol*100:.3f}%", "5-day std dev (Ledoit-Wolf)"),
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

        # Efficient frontier sketch
        st.markdown("#### Efficient Frontier (Monte Carlo Simulation)")
        st.caption("Expected returns estimated from 60-day historical mean returns. Risk from Ledoit-Wolf covariance. Colour = Sharpe ratio.")

        # Use historical mean returns (annualised) from the covariance window
        hist_mean_rets = window_returns.mean().values.astype(np.float32)  # daily mean per asset
        hist_mean_rets_ann = hist_mean_rets * 252                          # annualise

        n_sim = 3000
        np.random.seed(42)
        rand_w = np.random.dirichlet(np.ones(n_assets), n_sim)
        for i in range(n_sim):
            rand_w[i] = np.minimum(rand_w[i], max_weight)
            rand_w[i] /= rand_w[i].sum()

        sim_rets = rand_w @ hist_mean_rets_ann
        cov_ann = cov_matrix * 252 / 5  # annualise from 5-day holding
        sim_vols = np.array([np.sqrt(rand_w[i] @ cov_ann @ rand_w[i]) for i in range(n_sim)])
        sim_sharpes = np.where(sim_vols > 0, sim_rets / sim_vols, 0)

        opt_ret_ann = float(weights @ hist_mean_rets_ann)
        opt_vol_ann = float(np.sqrt(weights @ cov_ann @ weights))

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=sim_vols * 100, y=sim_rets * 100, mode="markers",
            marker=dict(size=3, color=sim_sharpes, colorscale="Viridis",
                        colorbar=dict(title="Sharpe"), opacity=0.5),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br><extra></extra>",
        ))
        fig_ef.add_trace(go.Scatter(
            x=[opt_vol_ann * 100], y=[opt_ret_ann * 100],
            mode="markers+text",
            marker=dict(size=16, color=C["gold"], symbol="star",
                        line=dict(width=2, color="#fff")),
            text=["SPO Optimal"], textposition="top center",
            textfont=dict(size=12, color=C["gold"]),
            name="Optimal Portfolio",
        ))
        eq_w = np.full(n_assets, 1.0 / n_assets)
        eq_ret_ann = float(eq_w @ hist_mean_rets_ann)
        eq_vol_ann = float(np.sqrt(eq_w @ cov_ann @ eq_w))
        fig_ef.add_trace(go.Scatter(
            x=[eq_vol_ann * 100], y=[eq_ret_ann * 100], mode="markers+text",
            marker=dict(size=12, color=C["muted"], symbol="diamond",
                        line=dict(width=2, color="#fff")),
            text=["Equal Wt"], textposition="bottom center",
            textfont=dict(size=11, color=C["muted"]),
            name="Equal Weight",
        ))
        fig_ef.update_xaxes(title_text="Annualised Volatility (%)")
        fig_ef.update_yaxes(title_text="Annualised Return (%)")
        _dark_layout(fig_ef, height=450)
        st.plotly_chart(fig_ef, use_container_width=True, key="ana_frontier")

    # ─────────────────────────────────────────────────────────────────
    #  TAB 3: BACKTEST — walk-forward simulation
    # ─────────────────────────────────────────────────────────────────
    with tabs[2]:

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

                # Optimize
                pl = DifferentiableMarkowitz(n_assets=n_assets, gamma=gamma, max_weight=max_weight)
                with torch.no_grad():
                    w_bt = pl(torch.tensor(pred_sel), torch.tensor(cov_bt)).numpy()

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
        bt_cum = float(np.prod(1 + bt_returns) - 1)
        bt_vol = float(bt_returns.std() * ann_factor)
        bt_sharpe = float(bt_returns.mean() / bt_returns.std() * ann_factor) if bt_returns.std() > 0 else 0.0
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

        # Cumulative returns comparison
        cum_port = np.cumprod(1 + bt_returns) - 1
        cum_eq = np.cumprod(1 + bt_eq) - 1
        cum_oracle = np.cumprod(1 + bt_oracle) - 1

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_date_labels, y=cum_port, name="SPO Portfolio",
                                     line=dict(color=C["gold"], width=2.5)))
        fig_bt.add_trace(go.Scatter(x=bt_date_labels, y=cum_eq, name="Equal Weight",
                                     line=dict(color=C["muted"], width=1.5, dash="dot")))
        fig_bt.add_trace(go.Scatter(x=bt_date_labels, y=cum_oracle, name="Oracle",
                                     line=dict(color=C["purple"], width=1.5, dash="dash")))
        fig_bt.update_yaxes(tickformat=".0%")
        _dark_layout(fig_bt, "Cumulative Returns", height=400)
        st.plotly_chart(fig_bt, use_container_width=True, key="bt_cum")

        # Drawdown
        c1, c2 = st.columns(2)
        with c1:
            dd = (np.maximum.accumulate(np.cumprod(1 + bt_returns)) - np.cumprod(1 + bt_returns)) / np.maximum.accumulate(np.cumprod(1 + bt_returns))
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=bt_date_labels, y=-dd, mode="lines",
                                         fill="tozeroy", line=dict(color=C["red"], width=1.5),
                                         fillcolor="rgba(224,108,108,0.15)", name="Drawdown"))
            fig_dd.update_yaxes(tickformat=".1%")
            _dark_layout(fig_dd, "Drawdown", height=300)
            st.plotly_chart(fig_dd, use_container_width=True, key="bt_dd")

        with c2:
            # Rolling IC
            ic_smooth = pd.Series(bt_ic_list).rolling(10, min_periods=3).mean().values
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Scatter(x=bt_date_labels, y=ic_smooth, mode="lines",
                                         line=dict(color=C["teal"], width=1.5), name="IC (10-day MA)"))
            fig_ic.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig_ic.update_yaxes(title_text="Spearman IC")
            _dark_layout(fig_ic, "Rolling Information Coefficient", height=300)
            st.plotly_chart(fig_ic, use_container_width=True, key="bt_ic")


if __name__ == "__main__":
    main()
