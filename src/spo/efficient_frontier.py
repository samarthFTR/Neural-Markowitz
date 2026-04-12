import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
from typing import Tuple

def annualize_returns_and_cov(daily_mean_returns: np.ndarray, daily_cov_matrix: np.ndarray, trading_days: int = 252) -> Tuple[np.ndarray, np.ndarray]:
    """
    Annualize daily returns and covariance matrix.
    Prevents double scaling.
    """
    annual_returns = daily_mean_returns * trading_days
    annual_cov = daily_cov_matrix * trading_days
    return annual_returns, annual_cov

def compute_portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.02) -> Tuple[float, float, float]:
    """
    Compute annualized return, volatility, and Sharpe ratio for a specific portfolio.
    Weights, mean_returns, and cov_matrix must already be properly scaled (e.g., annualized).
    """
    port_return = float(np.sum(mean_returns * weights))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe_ratio = float((port_return - risk_free_rate) / port_vol) if port_vol > 0 else 0.0
    return port_return, port_vol, sharpe_ratio

def simulate_portfolios(mean_returns: np.ndarray, cov_matrix: np.ndarray, num_portfolios: int = 10000, risk_free_rate: float = 0.02):
    """
    Generate random portfolios to visualize the feasible set.
    All inputs must be annualized.
    """
    np.random.seed(42)
    n_assets = len(mean_returns)
    # Generate random weights using Dirichlet distribution to ensure they sum to 1
    weights_array = np.random.dirichlet(np.ones(n_assets), num_portfolios)
    
    # Vectorized computation for efficiency
    sim_rets = np.dot(weights_array, mean_returns)
    
    # Portfolio variance: diag(W @ Cov @ W.T) is memory intensive for N=10000. 
    # Optimized: sum_columns((W @ Cov) * W)
    sim_vols = np.sqrt(np.sum((weights_array @ cov_matrix) * weights_array, axis=1))
    
    sim_sharpes = np.where(sim_vols > 0, (sim_rets - risk_free_rate) / sim_vols, 0)
    
    return weights_array, sim_rets, sim_vols, sim_sharpes

def optimize_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.02, target: str = "sharpe"):
    """
    Using scipy.optimize to find Max Sharpe or Min Volatility portfolio.
    Bounds apply long-only constraints (0, 1) and sum(weights) = 1.
    """
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
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

def compute_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.02, num_points: int = 50):
    """
    Generate points for the efficient frontier curve by minimizing volatility for a range of target returns.
    """
    # Find theoretical boundaries for the frontier
    _, min_vol_ret, _, _ = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, target="volatility")
    max_ret = float(np.max(mean_returns))
    
    if max_ret <= min_vol_ret:
        # Edge case: degenerate frontier
        return np.array([]), np.array([])
        
    target_returns = np.linspace(min_vol_ret, max_ret, num_points)
    ef_vols = []
    ef_rets = []
    
    n_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    init_guess = np.full(n_assets, 1.0 / n_assets)
    
    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    for tr in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - tr}
        ]
        res = sco.minimize(port_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            ef_vols.append(res.fun)
            ef_rets.append(tr)
            
    return np.array(ef_vols), np.array(ef_rets)

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
