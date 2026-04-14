"""
Efficient Frontier & Portfolio Optimisation Utilities.

All functions operate on ANNUALISED inputs unless explicitly noted.
Scaling pipeline: daily returns / covariance → annualise → optimise.

Key design decisions:
  - max_weight is passed through to all optimisation and simulation
    functions so the frontier matches the SPO portfolio's constraint set.
  - Mean return shrinkage is provided to reduce estimation noise from
    short estimation windows (standard in quantitative finance).
"""

import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
from typing import Tuple


# =====================================================================
#  Scaling & Shrinkage
# =====================================================================

def annualize_returns_and_cov(
    daily_mean_returns: np.ndarray,
    daily_cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Annualize daily returns and covariance matrix.

    Formulas (standard, no compounding adjustment):
        ret_annual = ret_daily × 252
        cov_annual = cov_daily × 252
    """
    annual_returns = daily_mean_returns * trading_days
    annual_cov = daily_cov_matrix * trading_days
    return annual_returns, annual_cov


def shrink_mean_returns(
    mean_returns: np.ndarray,
    n_obs: int,
    trading_days: int = 252,
) -> Tuple[np.ndarray, float]:
    """
    Shrink annualised mean return estimates toward the cross-sectional
    grand mean to reduce estimation noise.

    WHY THIS IS NEEDED:
    With T daily observations, the standard error of a stock's
    annualised mean return is approximately:
        SE ≈ σ_annual × √(252 / T)
    For T=60, σ=30%:  SE ≈ 0.30 × √4.2 ≈ 61%.
    Individual annualised mean estimates from 60-day windows are
    essentially noise.  Shrinkage is ESSENTIAL for any mean-variance
    analysis on short windows (Jorion 1986, Michaud 1989).

    Shrinkage intensity is inversely proportional to observation count:
        intensity = clip(1 − T/252,  0.3,  0.9)
    So:
        T=60  → 76% shrinkage (aggressive — very noisy estimates)
        T=126 → 50% shrinkage
        T=252 → 30% shrinkage (floor — always some regularisation)

    Parameters
    ----------
    mean_returns : (n,) annualised mean returns
    n_obs        : int,  number of daily observations used to estimate means
    trading_days : int,  trading days per year (default 252)

    Returns
    -------
    shrunk    : (n,) shrunk annualised mean returns
    intensity : float, shrinkage intensity in [0.3, 0.9]
    """
    n = len(mean_returns)
    if n <= 2:
        return mean_returns.copy(), 0.0

    intensity = float(np.clip(1.0 - n_obs / trading_days, 0.3, 0.9))
    grand_mean = float(mean_returns.mean())
    shrunk = grand_mean + (1.0 - intensity) * (mean_returns - grand_mean)

    return shrunk.astype(np.float32), intensity


# =====================================================================
#  Portfolio Performance
# =====================================================================

def compute_portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
) -> Tuple[float, float, float]:
    """
    Compute annualized return, volatility, and Sharpe ratio for a specific portfolio.
    Weights, mean_returns, and cov_matrix must already be properly scaled (e.g., annualized).
    """
    port_return = float(np.sum(mean_returns * weights))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe_ratio = float((port_return - risk_free_rate) / port_vol) if port_vol > 0 else 0.0
    return port_return, port_vol, sharpe_ratio


# =====================================================================
#  Monte Carlo Simulation
# =====================================================================

def simulate_portfolios(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_portfolios: int = 10000,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
):
    """
    Generate random portfolios to visualize the feasible set.
    All inputs must be annualized.

    Parameters
    ----------
    max_weight : float
        Maximum weight per asset.  Must match the SPO portfolio's constraint
        so the feasible set shown on the chart corresponds to the same
        optimisation problem the model is solving.
        When max_weight < 1, Dirichlet-sampled weights are iteratively
        clipped and renormalised (converges quickly for typical values).
    """
    np.random.seed(42)
    n_assets = len(mean_returns)

    # Generate random weights using Dirichlet distribution (sum = 1)
    weights_array = np.random.dirichlet(np.ones(n_assets), num_portfolios)

    # Enforce max_weight via iterative clip-and-renormalise.
    # For n=64, max_weight=0.10, most Dirichlet(1) draws already satisfy
    # the constraint (E[w_i] = 1/64 ≈ 0.016), so this converges in ≤5 iters.
    if max_weight < 1.0:
        for _ in range(15):
            np.clip(weights_array, 0.0, max_weight, out=weights_array)
            row_sums = weights_array.sum(axis=1, keepdims=True)
            weights_array = weights_array / row_sums

    # Vectorized computation for efficiency
    sim_rets = np.dot(weights_array, mean_returns)

    # Portfolio variance: diag(W @ Cov @ W.T) is memory intensive for N=10000.
    # Optimized: sum_columns((W @ Cov) * W)
    sim_vols = np.sqrt(np.sum((weights_array @ cov_matrix) * weights_array, axis=1))

    sim_sharpes = np.where(sim_vols > 0, (sim_rets - risk_free_rate) / sim_vols, 0)

    return weights_array, sim_rets, sim_vols, sim_sharpes


# =====================================================================
#  Optimisation (Max Sharpe / Min Vol)
# =====================================================================

def optimize_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    target: str = "sharpe",
    max_weight: float = 1.0,
):
    """
    Using scipy.optimize to find Max Sharpe or Min Volatility portfolio.

    Parameters
    ----------
    max_weight : float
        Upper bound on any single asset's weight.  Must match the SPO
        portfolio's constraint for a fair visual comparison on the
        efficient frontier chart.
    """
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    # CRITICAL: use max_weight as upper bound, not 1.0.
    # This ensures the optimised portfolios live in the same feasible set
    # as the SPO portfolio (which is constrained to max_weight).
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    init_guess = np.full(n_assets, 1.0 / n_assets)

    if target == "sharpe":
        # Minimize negative sharpe -> maximize sharpe
        def objective(w, ret, cov, rf):
            _, p_vol, p_sr = compute_portfolio_performance(w, ret, cov, rf)
            return -p_sr
    elif target == "volatility":
        # Minimize volatility
        def objective(w, ret, cov, rf):
            _, p_vol, _ = compute_portfolio_performance(w, ret, cov, rf)
            return p_vol
    else:
        raise ValueError("Target must be 'sharpe' or 'volatility'")

    result = sco.minimize(objective, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = result.x if result.success else init_guess
    opt_ret, opt_vol, opt_sr = compute_portfolio_performance(opt_weights, mean_returns, cov_matrix, risk_free_rate)

    return opt_weights, opt_ret, opt_vol, opt_sr


# =====================================================================
#  Efficient Frontier Curve
# =====================================================================

def compute_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    num_points: int = 50,
    max_weight: float = 1.0,
):
    """
    Generate points for the efficient frontier curve by minimizing
    volatility for a range of target returns.

    Parameters
    ----------
    max_weight : float
        Same constraint as the SPO portfolio.  This bounds the maximum
        achievable return (you can't put 100% in the best stock) and
        produces a frontier that accurately reflects the constrained
        feasible set.
    """
    # Lower bound: return of the Min Volatility portfolio (under same constraints)
    _, min_vol_ret, _, _ = optimize_portfolio(
        mean_returns, cov_matrix, risk_free_rate,
        target="volatility", max_weight=max_weight,
    )

    # Upper bound: maximum achievable return under the constraint set
    # {w: 0 ≤ w_i ≤ max_weight, Σw_i = 1}.   This is a bounded LP;
    # the greedy solution packs max_weight into the highest-return assets.
    sorted_rets = np.sort(mean_returns)[::-1]
    n_full = min(len(sorted_rets), int(np.floor(1.0 / max_weight)))
    max_ret = float(np.sum(sorted_rets[:n_full]) * max_weight)
    remainder = 1.0 - n_full * max_weight
    if remainder > 1e-9 and n_full < len(sorted_rets):
        max_ret += float(sorted_rets[n_full] * remainder)

    if max_ret <= min_vol_ret:
        # Edge case: degenerate frontier
        return np.array([]), np.array([])

    target_returns = np.linspace(min_vol_ret, max_ret, num_points)
    ef_vols = []
    ef_rets = []

    n_assets = len(mean_returns)
    # Use same max_weight bound as elsewhere
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    init_guess = np.full(n_assets, 1.0 / n_assets)

    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    for tr in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            # Bind tr via default arg to avoid late-binding closure issues
            {'type': 'eq', 'fun': lambda x, _tr=tr: np.sum(mean_returns * x) - _tr}
        ]
        res = sco.minimize(port_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            ef_vols.append(res.fun)
            ef_rets.append(tr)

    return np.array(ef_vols), np.array(ef_rets)


# =====================================================================
#  Plotly Chart
# =====================================================================

def plot_results(sim_vols, sim_rets, sim_sharpes,
                 opt_vol, opt_ret,
                 min_vol_vol, min_vol_ret,
                 ef_vols, ef_rets,
                 spo_vol=None, spo_ret=None,
                 eq_vol=None, eq_ret=None,
                 height=450):
    """
    Build the Plotly figure overlaying Random Portfolios, Max Sharpe, Min Volatility, and the Frontier.
    """
    fig = go.Figure()

    # 1. Random portfolios
    fig.add_trace(go.Scatter(
        x=sim_vols * 100, y=sim_rets * 100, mode="markers",
        marker=dict(size=4, color=sim_sharpes, colorscale="Viridis",
                    colorbar=dict(title="Sharpe Ratio"), opacity=0.3),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
    ))

    # 2. Efficient Frontier Line
    if len(ef_vols) > 0:
        fig.add_trace(go.Scatter(
            x=ef_vols * 100, y=ef_rets * 100, mode="lines",
            line=dict(color="white", width=2, dash="dash"),
            name="Efficient Frontier"
        ))

    # 3. Max Sharpe Portfolio
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_ret * 100], mode="markers+text",
        marker=dict(size=14, color="gold", symbol="star", line=dict(width=1, color="black")),
        text=["Max Sharpe"], textposition="top center",
        textfont=dict(size=12, color="gold"),
        name="Max Sharpe Portfolio",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Max Sharpe</extra>",
    ))

    # 4. Min Volatility Portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol_vol * 100], y=[min_vol_ret * 100], mode="markers+text",
        marker=dict(size=12, color="cyan", symbol="triangle-up", line=dict(width=1, color="black")),
        text=["Min Volatility"], textposition="bottom center",
        textfont=dict(size=11, color="cyan"),
        name="Min Vol Portfolio",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Min Volatility</extra>",
    ))

    # 5. Equal Weight Benchmark
    if eq_vol is not None and eq_ret is not None:
        fig.add_trace(go.Scatter(
            x=[eq_vol * 100], y=[eq_ret * 100], mode="markers+text",
            marker=dict(size=12, color="#8B949E", symbol="diamond", line=dict(width=1, color="white")),
            text=["Equal Weight"], textposition="bottom center",
            textfont=dict(size=11, color="#8B949E"),
            name="Equal Weight Benchmark",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Equal Weight</extra>",
        ))

    # 6. SPO Optimal Portfolio (From Model)
    if spo_vol is not None and spo_ret is not None:
        fig.add_trace(go.Scatter(
            x=[spo_vol * 100], y=[spo_ret * 100], mode="markers+text",
            marker=dict(size=16, color="#E06C6C", symbol="cross", line=dict(width=2, color="white")),
            text=["SPO Optimal"], textposition="top center",
            textfont=dict(size=12, color="#E06C6C"),
            name="SPO Optimal (Prediction)",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>SPO Optimal</extra>",
        ))

    fig.update_xaxes(title_text="Annualised Volatility (%)")
    fig.update_yaxes(title_text="Annualised Expected Return (%)")

    # Formatting adjustments handled via _dark_layout primarily, but setup defaults here
    fig.update_layout(height=height, hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    return fig
