"""
Portfolio-Level Backtesting & Evaluation.

Provides walk-forward simulation and financial metrics for comparing
the SPO pipeline against the existing predict-then-rank baseline.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from utils.logger import logging


class PortfolioBacktester:
    """
    Walk-forward backtester.

    For each test date t  (one cross-section):
        1. Predict returns:  ĉ_t = f(X_t; θ)
        2. Solve portfolio:  w_t = argmin Markowitz(ĉ_t, Σ_t)
        3. Record realised return:  r_t = w_t · c_t

    Parameters
    ----------
    prediction_net  : ReturnPredictionNet
    portfolio_layer : DifferentiableMarkowitz
    """

    def __init__(self, prediction_net, portfolio_layer):
        self.pred_net = prediction_net
        self.port_layer = portfolio_layer

    @torch.no_grad()
    def run(self, dataset):
        """
        Run a full walk-forward backtest on the given dataset.

        Parameters
        ----------
        dataset : PortfolioCrossSectionDataset

        Returns
        -------
        results : dict  with keys
            'dates', 'portfolio_returns', 'oracle_returns',
            'weights_history', 'predictions', 'metrics'
        """
        self.pred_net.eval()

        portfolio_returns = []
        oracle_returns = []
        equal_weight_returns = []
        weights_history = []
        predictions_history = []
        ic_list = []
        dates = []

        n_assets = self.port_layer.n_assets

        for idx in range(len(dataset)):
            features, true_returns, cov = dataset[idx]
            date_str = dataset.get_date(idx)

            # Add batch dim
            features_b = features.unsqueeze(0)       # (1, N, F)
            cov_b = cov.unsqueeze(0)                  # (1, N, N)

            # ── predict ──
            predicted = self.pred_net(features_b).squeeze(0)   # (N,)

            # ── SPO portfolio ──
            w_spo = self.port_layer(predicted, cov)            # (N,)
            r_spo = (w_spo * true_returns).sum().item()

            # ── oracle portfolio ──
            w_oracle = self.port_layer(true_returns, cov)      # (N,)
            r_oracle = (w_oracle * true_returns).sum().item()

            # ── equal-weight benchmark ──
            w_eq = torch.full((n_assets,), 1.0 / n_assets)
            r_eq = (w_eq * true_returns).sum().item()

            # ── Spearman IC ──
            ic, _ = spearmanr(predicted.numpy(), true_returns.numpy())

            portfolio_returns.append(r_spo)
            oracle_returns.append(r_oracle)
            equal_weight_returns.append(r_eq)
            weights_history.append(w_spo.numpy())
            predictions_history.append(predicted.numpy())
            ic_list.append(ic)
            dates.append(date_str)

        metrics = self._compute_metrics(
            portfolio_returns, oracle_returns, equal_weight_returns,
            weights_history, ic_list,
        )

        return {
            "dates": dates,
            "portfolio_returns": np.array(portfolio_returns),
            "oracle_returns": np.array(oracle_returns),
            "equal_weight_returns": np.array(equal_weight_returns),
            "weights_history": np.stack(weights_history),
            "predictions": np.stack(predictions_history),
            "ic_list": np.array(ic_list),
            "metrics": metrics,
        }

    @staticmethod
    def _compute_metrics(
        port_rets, oracle_rets, eq_rets, weights_hist, ic_list
    ):
        """Compute portfolio-level financial metrics."""

        port_rets = np.array(port_rets)
        oracle_rets = np.array(oracle_rets)
        eq_rets = np.array(eq_rets)
        weights_hist = np.array(weights_hist)
        ic_arr = np.array(ic_list)

        # ── Realised statistics ──
        ann_factor = np.sqrt(252 / 5)   # 5-day holding period
        mean_ret = port_rets.mean()
        vol = port_rets.std()
        sharpe = (mean_ret / vol * ann_factor) if vol > 0 else 0.0

        # ── Max drawdown ──
        cum = np.cumprod(1 + port_rets)
        peak = np.maximum.accumulate(cum)
        drawdown = (peak - cum) / peak
        max_dd = drawdown.max()

        # ── Decision regret ──
        regret = oracle_rets - port_rets
        mean_regret = regret.mean()

        # ── Turnover ──
        if len(weights_hist) > 1:
            diffs = np.abs(np.diff(weights_hist, axis=0)).sum(axis=1)
            avg_turnover = diffs.mean()
        else:
            avg_turnover = 0.0

        # ── Information Coefficient ──
        mean_ic = np.nanmean(ic_arr)
        ic_hit_rate = (ic_arr > 0).mean()

        # ── vs Equal Weight ──
        eq_mean = eq_rets.mean()
        eq_vol = eq_rets.std()
        eq_sharpe = (eq_mean / eq_vol * ann_factor) if eq_vol > 0 else 0.0

        metrics = {
            "annualised_return":  mean_ret * 252 / 5,
            "annualised_vol":     vol * ann_factor,
            "sharpe_ratio":       sharpe,
            "max_drawdown":       max_dd,
            "mean_decision_regret": mean_regret,
            "avg_turnover":       avg_turnover,
            "mean_ic":            mean_ic,
            "ic_hit_rate":        ic_hit_rate,
            "cumulative_return":  float(np.cumprod(1 + port_rets)[-1] - 1),
            "eq_weight_sharpe":   eq_sharpe,
            "eq_weight_cum_ret":  float(np.cumprod(1 + eq_rets)[-1] - 1),
        }

        return metrics

    @staticmethod
    def print_report(metrics: dict, label: str = "SPO+"):
        """Pretty-print a metrics dictionary."""
        print(f"\n{'=' * 60}")
        print(f"  PORTFOLIO BACKTEST REPORT — {label}")
        print(f"{'=' * 60}")
        print(f"  Annualised Return:     {metrics['annualised_return']:>+.4f}")
        print(f"  Annualised Volatility: {metrics['annualised_vol']:>.4f}")
        print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:>.4f}")
        print(f"  Max Drawdown:          {metrics['max_drawdown']:>.4f}")
        print(f"  Cumulative Return:     {metrics['cumulative_return']:>+.4f}")
        print(f"  Mean Decision Regret:  {metrics['mean_decision_regret']:>.6f}")
        print(f"  Avg Daily Turnover:    {metrics['avg_turnover']:>.4f}")
        print(f"  Mean Spearman IC:      {metrics['mean_ic']:>.4f}")
        print(f"  IC Hit Rate:           {metrics['ic_hit_rate']:>.2%}")
        print(f"  ────────────────────────────────────────")
        print(f"  Equal-Weight Sharpe:   {metrics['eq_weight_sharpe']:>.4f}")
        print(f"  Equal-Weight Cum Ret:  {metrics['eq_weight_cum_ret']:>+.4f}")
        print(f"{'=' * 60}\n")
