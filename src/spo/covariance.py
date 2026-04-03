"""
Rolling Covariance Estimation with Ledoit-Wolf Shrinkage.

Produces a covariance matrix for each trading date using a trailing
window of daily returns.  The output feeds into the differentiable
Markowitz layer as a fixed (non-learnable) parameter.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from utils.logger import logging


def compute_rolling_covariance(
    prices_df: pd.DataFrame,
    window: int = 60,
    shrinkage: str = "ledoit_wolf",
    min_periods: int = 40,
) -> dict:
    """
    Estimate a covariance matrix Σ_t for each date t, using returns
    from [t − window, t).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Wide-format close prices (index=Date, columns=Tickers).
    window : int
        Look-back window in trading days (default 60 ≈ 3 months).
    shrinkage : str
        "ledoit_wolf" — Ledoit-Wolf shrinkage (recommended).
        "empirical"   — raw sample covariance (noisy, for ablation).
    min_periods : int
        Minimum number of non-NaN observations required.

    Returns
    -------
    cov_dict : dict[str, np.ndarray]
        {date_string → Σ_t} where Σ_t has shape (n_assets, n_assets).
    tickers : list[str]
        Ordered list of ticker symbols (column order of Σ).
    """
    returns = prices_df.pct_change().dropna(how="all")
    tickers = list(returns.columns)
    n_assets = len(tickers)

    dates = returns.index.tolist()
    cov_dict = {}

    logging.info(
        f"Computing rolling covariance: window={window}, "
        f"shrinkage={shrinkage}, n_assets={n_assets}"
    )

    for i in range(window, len(dates)):
        date_str = str(dates[i])
        window_returns = returns.iloc[i - window : i].dropna(axis=0, how="any")

        if len(window_returns) < min_periods:
            continue

        if shrinkage == "ledoit_wolf":
            lw = LedoitWolf().fit(window_returns.values)
            cov_matrix = lw.covariance_
        else:
            cov_matrix = window_returns.cov().values

        # Ensure positive-definiteness by clipping tiny eigenvalues
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Symmetrise to kill floating-point asymmetry
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

        cov_dict[date_str] = cov_matrix.astype(np.float32)

    logging.info(f"Covariance matrices computed for {len(cov_dict)} dates.")
    return cov_dict, tickers


def precompute_and_save(
    raw_csv_path: str,
    output_path: str,
    window: int = 60,
):
    """Convenience: load raw prices, compute covariances, save as .npz."""
    prices = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    cov_dict, tickers = compute_rolling_covariance(prices, window=window)

    # Save as compressed numpy archive
    np.savez_compressed(
        output_path,
        dates=np.array(list(cov_dict.keys())),
        covariances=np.stack(list(cov_dict.values())),
        tickers=np.array(tickers),
    )
    logging.info(f"Saved covariance archive to {output_path}")
    return output_path
