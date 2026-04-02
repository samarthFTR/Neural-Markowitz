"""
End-to-End SPO Training Pipeline.

Orchestrates the full workflow:
  1. Load data & covariances
  2. Build PyTorch dataset
  3. Train prediction network with SPO+ / MSE / Hybrid loss
  4. Backtest and report

Supports three training modes for fair comparison:
  - 'spo+'   — decision-focused (minimise portfolio regret)
  - 'mse'    — prediction-focused (minimise return forecast error)
  - 'hybrid' — λ · SPO+ + (1−λ) · MSE
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Local imports ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import logging
from utils.utils import save_object
from spo.prediction_net import ReturnPredictionNet
from spo.portfolio_layer import DifferentiableMarkowitz
from spo.spo_loss import SPOPlusLoss, HybridLoss, DecisionRegretLoss
from spo.evaluation import PortfolioBacktester
from spo.covariance import compute_rolling_covariance
from utils.datasets import PortfolioCrossSectionDataset, build_datasets

import pandas as pd


# =====================================================================
#  Configuration
# =====================================================================

class SPOConfig:
    """All knobs for the SPO pipeline in one place."""

    # Paths
    raw_prices_path:    str = os.path.join("data", "raw data", "raw.csv")
    train_csv_path:     str = os.path.join("data", "train.csv")
    test_csv_path:      str = os.path.join("data", "test.csv")
    model_save_dir:     str = os.path.join("models", "spo")

    # Covariance
    cov_window:         int = 60
    cov_shrinkage:      str = "ledoit_wolf"

    # Network
    hidden_dims:        list = None
    dropout:            float = 0.2

    # Training
    mode:               str = "spo+"        # "spo+" | "mse" | "hybrid"
    hybrid_lambda:      float = 0.5
    lr:                 float = 1e-3
    weight_decay:       float = 1e-4
    n_epochs:           int = 50
    patience:           int = 10            # early-stopping patience
    grad_clip:          float = 1.0
    batch_size:         int = 1             # 1 = one cross-section per step

    # Portfolio
    gamma:              float = 0.5         # risk aversion
    max_weight:         float = 0.10        # concentration limit per asset
    solver:             str = "SCS"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


# =====================================================================
#  Trainer
# =====================================================================

class SPOTrainer:
    """
    End-to-end training for the Predict-then-Optimize pipeline.

    Usage
    -----
    >>> config = SPOConfig()
    >>> trainer = SPOTrainer(config)
    >>> results = trainer.run()
    """

    def __init__(self, config: SPOConfig = None):
        self.config = config or SPOConfig()
        if self.config.hidden_dims is None:
            self.config.hidden_dims = [64, 32]
        os.makedirs(self.config.model_save_dir, exist_ok=True)

    def run(self):
        """Full pipeline: data → train → backtest → report."""
        cfg = self.config

        # ── Step 1: Covariance matrices ──────────────────────
        logging.info("="*60)
        logging.info("STEP 1: Computing rolling covariance matrices")
        logging.info("="*60)
        print("\n[SPO] Step 1: Computing rolling covariance matrices...")

        prices = pd.read_csv(cfg.raw_prices_path, index_col=0, parse_dates=True)
        cov_dict, tickers = compute_rolling_covariance(
            prices, window=cfg.cov_window, shrinkage=cfg.cov_shrinkage,
        )
        n_assets = len(tickers)
        print(f"  {len(cov_dict)} covariance matrices for {n_assets} assets.")

        # ── Step 2: Build datasets ───────────────────────────
        logging.info("="*60)
        logging.info("STEP 2: Building cross-sectional datasets")
        logging.info("="*60)
        print("[SPO] Step 2: Building cross-sectional datasets...")

        train_ds, test_ds = build_datasets(
            cfg.train_csv_path, cfg.test_csv_path,
            cov_dict=cov_dict, tickers_order=tickers,
        )
        n_features = train_ds[0][0].shape[1]
        print(f"  Train: {len(train_ds)} days | Test: {len(test_ds)} days")
        print(f"  Features per stock: {n_features}")

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=False
        )

        # ── Step 3: Build model components ───────────────────
        logging.info("="*60)
        logging.info("STEP 3: Building model components")
        logging.info("="*60)
        print("[SPO] Step 3: Building model components...")

        pred_net = ReturnPredictionNet(
            n_features=n_features,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
        port_layer = DifferentiableMarkowitz(
            n_assets=n_assets,
            gamma=cfg.gamma,
            max_weight=cfg.max_weight,
        )

        # Loss
        if cfg.mode == "spo+":
            criterion = SPOPlusLoss(port_layer)
        elif cfg.mode == "hybrid":
            criterion = HybridLoss(port_layer, lam=cfg.hybrid_lambda)
        else:
            criterion = nn.MSELoss()

        regret_metric = DecisionRegretLoss(port_layer)

        optimiser = torch.optim.Adam(
            pred_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=5, factor=0.5
        )

        print(f"  Mode: {cfg.mode} | LR: {cfg.lr} | Epochs: {cfg.n_epochs}")
        print(f"  Markowitz γ={cfg.gamma}, max_weight={cfg.max_weight}")

        # ── Step 4: Training loop ────────────────────────────
        logging.info("="*60)
        logging.info(f"STEP 4: Training ({cfg.mode})")
        logging.info("="*60)
        print(f"\n[SPO] Step 4: Training ({cfg.mode})...\n")

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, cfg.n_epochs + 1):
            pred_net.train()
            epoch_loss = 0.0
            n_batches = 0

            for features, true_returns, cov in train_loader:
                # features:     (B, N, F)
                # true_returns: (B, N)
                # cov:          (B, N, N)

                optimiser.zero_grad()

                predicted = pred_net(features)    # (B, N)

                if cfg.mode == "mse":
                    loss = criterion(predicted, true_returns)
                else:
                    loss = criterion(predicted, true_returns, cov)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    pred_net.parameters(), max_norm=cfg.grad_clip
                )
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step(avg_loss)

            # ── Validation (decision regret on last 20% of train) ──
            if epoch % 5 == 0 or epoch == 1:
                pred_net.eval()
                val_regret = self._quick_eval(pred_net, train_ds, regret_metric,
                                              start_frac=0.8)
                lr_now = optimiser.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:>3d}/{cfg.n_epochs} | "
                    f"Loss: {avg_loss:.6f} | "
                    f"Val Regret: {val_regret:.6f} | "
                    f"LR: {lr_now:.2e}"
                )
                logging.info(
                    f"Epoch {epoch}: loss={avg_loss:.6f}, "
                    f"val_regret={val_regret:.6f}"
                )

            # ── Early stopping ──
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(pred_net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        # Restore best
        if best_state is not None:
            pred_net.load_state_dict(best_state)
        pred_net.eval()

        # Save
        model_path = os.path.join(cfg.model_save_dir, f"pred_net_{cfg.mode}.pt")
        torch.save(pred_net.state_dict(), model_path)
        logging.info(f"Saved prediction network to {model_path}")
        print(f"\n  Model saved to {model_path}")

        # ── Step 5: Backtest ─────────────────────────────────
        logging.info("="*60)
        logging.info("STEP 5: Backtesting on test set")
        logging.info("="*60)
        print(f"\n[SPO] Step 5: Backtesting on test set...")

        backtester = PortfolioBacktester(pred_net, port_layer)
        results = backtester.run(test_ds)
        PortfolioBacktester.print_report(results["metrics"], label=cfg.mode.upper())

        # Save full results
        results_path = os.path.join(cfg.model_save_dir, f"results_{cfg.mode}.pkl")
        save_object(results_path, results)

        return results

    @staticmethod
    def _quick_eval(pred_net, dataset, regret_fn, start_frac=0.8):
        """Compute mean decision regret on the tail of the dataset."""
        start = int(len(dataset) * start_frac)
        regrets = []
        with torch.no_grad():
            for idx in range(start, len(dataset)):
                feat, ret, cov = dataset[idx]
                pred = pred_net(feat.unsqueeze(0)).squeeze(0)
                r = regret_fn(pred.unsqueeze(0), ret.unsqueeze(0), cov.unsqueeze(0))
                regrets.append(r.item())
        return np.mean(regrets) if regrets else 0.0


# =====================================================================
#  Comparison runner
# =====================================================================

def run_comparison():
    """
    Train all three modes (SPO+, MSE, Hybrid) and print a side-by-side
    comparison of portfolio metrics.
    """
    modes = ["mse", "spo+", "hybrid"]
    all_results = {}

    for mode in modes:
        print(f"\n{'#' * 60}")
        print(f"  TRAINING MODE: {mode.upper()}")
        print(f"{'#' * 60}")

        cfg = SPOConfig()
        cfg.mode = mode
        cfg.n_epochs = 30       # Reduce for comparison runs
        trainer = SPOTrainer(cfg)
        results = trainer.run()
        all_results[mode] = results["metrics"]

    # ── Side-by-side comparison ──
    print(f"\n\n{'=' * 70}")
    print(f"{'COMPARISON TABLE':^70}")
    print(f"{'=' * 70}")

    header = f"{'Metric':<28} {'MSE':>12} {'SPO+':>12} {'Hybrid':>12}"
    print(header)
    print("-" * 70)

    key_metrics = [
        ("Sharpe Ratio",          "sharpe_ratio"),
        ("Annualised Return",     "annualised_return"),
        ("Annualised Vol",        "annualised_vol"),
        ("Max Drawdown",          "max_drawdown"),
        ("Cumulative Return",     "cumulative_return"),
        ("Mean Decision Regret",  "mean_decision_regret"),
        ("Avg Turnover",          "avg_turnover"),
        ("Mean IC",               "mean_ic"),
        ("IC Hit Rate",           "ic_hit_rate"),
    ]

    for label, key in key_metrics:
        vals = []
        for m in modes:
            v = all_results[m].get(key, 0)
            vals.append(f"{v:>+.4f}" if "return" in key or "regret" in key
                        else f"{v:>.4f}")
        print(f"  {label:<26} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print(f"{'=' * 70}\n")

    return all_results


# =====================================================================
#  Main entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPO Portfolio Training")
    parser.add_argument(
        "--mode", choices=["spo+", "mse", "hybrid", "compare"],
        default="spo+", help="Training mode (default: spo+)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison()
    else:
        cfg = SPOConfig()
        cfg.mode = args.mode
        cfg.n_epochs = args.epochs
        cfg.lr = args.lr
        cfg.gamma = args.gamma
        cfg.max_weight = args.max_weight
        cfg.hidden_dims = args.hidden

        trainer = SPOTrainer(cfg)
        trainer.run()
