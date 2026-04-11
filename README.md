# Portfolio Optimization using Smart Predict-then-Optimize (SPO)

> **End-to-end decision-focused learning for equity portfolio construction** — combining cross-sectional return prediction with differentiable convex optimization to directly minimise portfolio decision regret.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CVXPY-Convex_Optimization-0077B5?style=flat-square" />
  <img src="https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [SPO Results](#spo-results)
- [Project Structure](#project-structure)
- [Data & Features](#data--features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Extending the Project](#extending-the-project)
- [References](#references)
- [Disclaimer](#disclaimer)

---

## Overview

Traditional quantitative pipelines follow a **two-stage** approach: first predict returns, then construct a portfolio — optimising each stage independently. This project implements the **Smart Predict-then-Optimize (SPO)** framework, which collapses both stages into a single differentiable pipeline, training the prediction model to produce forecasts whose *downstream portfolio decisions* have minimal regret.

The project includes two complementary pipelines:

| Pipeline | Approach | Objective |
|----------|----------|-----------|
| **Stacked ML** | Classification → Regression (XGBoost, Ridge, RF) | Minimise prediction error (MSE, AUC) |
| **SPO Layer** | Neural Network → Differentiable Markowitz Optimizer | Minimise portfolio decision regret |

Both pipelines share the same data ingestion and feature engineering foundation, enabling direct comparison between prediction-focused and decision-focused learning.

---

## Architecture

```
                              ┌──────────────────────────────────────────┐
                              │        DATA INGESTION (ingestion.py)     │
                              │  yfinance → Feature Engineering → Split  │
                              └──────────────┬───────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
          ┌──────────────────────────┐            ┌───────────────────────────────┐
          │   STACKED ML PIPELINE    │            │       SPO PIPELINE            │
          │                          │            │                               │
          │  ┌────────────────────┐  │            │  ┌─────────────────────────┐  │
          │  │ Stage 1: Classify  │  │            │  │ ReturnPredictionNet     │  │
          │  │ (RF, XGBoost)      │  │            │  │ (Shared-weight MLP)     │  │
          │  └────────┬───────────┘  │            │  └───────────┬─────────────┘  │
          │           │ probabilities│            │              │ μ̂ (predicted)  │
          │  ┌────────▼───────────┐  │            │  ┌───────────▼─────────────┐  │
          │  │ Stage 2: Regress   │  │            │  │ Differentiable          │  │
          │  │ (Ridge, XGB, RF)   │  │            │  │ Markowitz Optimizer     │  │
          │  └────────┬───────────┘  │            │  │ (CVXPY + KKT Backward) │  │
          │           │ rankings     │            │  └───────────┬─────────────┘  │
          │  ┌────────▼───────────┐  │            │              │ w* (weights)   │
          │  │ Top-K Selection    │  │            │  ┌───────────▼─────────────┐  │
          │  │ Spearman IC eval   │  │            │  │ SPO+ Loss / MSE Loss   │  │
          │  └────────────────────┘  │            │  │ (Elmachtoub & Grigas)   │  │
          │                          │            │  └─────────────────────────┘  │
          └──────────────────────────┘            └───────────────────────────────┘
```

---

## SPO Results

Comparison of three training modes on the out-of-sample test set (497 days, 64 assets):

| Metric | MSE | SPO+ | Hybrid (λ=0.5) |
|--------|:---:|:----:|:---------------:|
| **Sharpe Ratio** | 0.77 | **0.99** | 0.88 |
| **Annualised Return** | +8.5% | **+16.6%** | +12.8% |
| **Cumulative Return** | +117.8% | **+345.8%** | +217.3% |
| **Decision Regret** | 0.0588 | **0.0572** | 0.0580 |
| Annualised Volatility | 0.111 | 0.167 | 0.145 |
| Max Drawdown | 0.515 | 0.588 | 0.518 |
| Avg Turnover | 0.107 | 0.506 | 0.592 |
| IC Hit Rate | 0.0% | 44.9% | 43.9% |

**Key takeaway:** SPO+ delivers a **29% higher Sharpe ratio** and **3× cumulative returns** compared to MSE, confirming that optimising for downstream decision quality outperforms optimising for prediction accuracy alone.

> **Note:** All three modes underperform the equal-weight benchmark (Sharpe 1.51) because the test period (2024–2026) features a strong bull market where concentration in predicted winners carries higher volatility.

---

## Project Structure

```
Portfolio-Optimization-using-SPO/
├── src/
│   ├── ingestion.py                 # Data download, feature engineering, train/test split
│   ├── preprocessing.py             # ColumnTransformer pipeline (Imputer + Scaler)
│   ├── training.py                  # Stacked ML pipeline (classifiers → regressors)
│   ├── spo/                         # ── Smart Predict-then-Optimize Module ──
│   │   ├── prediction_net.py        #    Cross-sectional MLP (shared weights)
│   │   ├── portfolio_layer.py       #    Differentiable Markowitz (custom autograd)
│   │   ├── spo_loss.py              #    SPO+ loss, decision regret, hybrid loss
│   │   ├── covariance.py            #    Rolling Ledoit-Wolf covariance estimation
│   │   ├── trainer.py               #    End-to-end training loop (3 modes)
│   │   └── evaluation.py            #    Walk-forward backtester & financial metrics
│   └── utils/
│       ├── datasets.py              #    PyTorch Dataset (cross-sectional samples)
│       ├── logger.py                #    Logging configuration
│       ├── utils.py                 #    Object serialisation helpers
│       └── exception.py             #    Custom exception with traceback
├── data/
│   ├── raw data/raw.csv             # Raw close prices (yfinance)
│   ├── portfolio_dataset.csv        # Engineered features + targets (long format)
│   ├── train.csv                    # Chronological training split (75%)
│   ├── test.csv                     # Chronological test split (25%)
│   └── processor.pkl                # Fitted preprocessing pipeline
├── models/
│   ├── classifier.pkl               # Best base classifier
│   ├── regressor.pkl                # Best ranking regressor
│   ├── model.pkl                    # Full stacked model ensemble
│   └── spo/                         # SPO-trained networks & backtest results
│       ├── pred_net_mse.pt          #    MSE-trained weights
│       ├── pred_net_spo+.pt         #    SPO+-trained weights
│       ├── pred_net_hybrid.pt       #    Hybrid-trained weights
│       └── results_*.pkl            #    Full backtest results per mode
├── logs/                            # Runtime logs
├── notebooks/                       # Exploratory analysis
└── requirements.txt
```

---

## Data & Features

**Universe:** 64 large-cap U.S. equities across sectors — Technology, Financials, Healthcare, Consumer, Industrials, and Energy.

**Period:** Daily close prices from `2018-01-01` to `2026-01-01` (~2,000 trading days).

**Engineered Features (10):**

| Feature | Description |
|---------|-------------|
| `RET_1D`, `RET_5D`, `RET_10D` | Short-horizon return signals |
| `MOM_10`, `MOM_20` | Medium-term momentum |
| `VOL_5`, `VOL_10` | Rolling realised volatility |
| `ALPHA_1D` | Relative strength (stock return − cross-sectional mean) |
| `RANK_MOM_10` | Cross-sectional momentum rank |
| `ANTI_MOM_10` | Short-term mean-reversion signal |

**Dual Targets:**

| Target | Type | Definition |
|--------|------|------------|
| `TARGET_RETURN` | Continuous | Forward 5-day return |
| `TARGET_CLASS` | Binary | 1 if return is in the daily top quintile (top 20%), else 0 |

All features are computed in **long format** (`Date × Ticker`) to ensure correct cross-sectional operations and avoid lookahead bias.

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/samarthFTR/Portfolio-Optimization-using-SPO.git
cd Portfolio-Optimization-using-SPO

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Data Ingestion & Stacked ML Pipeline

Run the full traditional pipeline (downloads data, engineers features, trains stacked models):

```bash
python src/ingestion.py
```

This will:
- Download historical prices via `yfinance`
- Compute alpha features and dual targets
- Perform a chronological 75/25 train-test split
- Fit the preprocessing pipeline (`SimpleImputer` + `StandardScaler`)
- Train and evaluate the stacked classification → regression architecture
- Save all models and artifacts to `models/`

### 2. SPO Pipeline (Decision-Focused Learning)

> **Prerequisite:** Run `ingestion.py` at least once to generate `train.csv`, `test.csv`, and `raw.csv`.

Train with a specific loss function:

```bash
# SPO+ loss — decision-focused (recommended)
python src/spo/trainer.py --mode spo+ --epochs 50

# MSE loss — prediction-focused baseline
python src/spo/trainer.py --mode mse --epochs 50

# Hybrid loss — blended (λ·SPO+ + (1−λ)·MSE)
python src/spo/trainer.py --mode hybrid --epochs 50
```

Run a head-to-head comparison of all three modes:

```bash
python src/spo/trainer.py --mode compare
```

Full configuration options:

```bash
python src/spo/trainer.py \
  --mode spo+ \
  --epochs 30 \
  --lr 0.001 \
  --gamma 0.5 \
  --max-weight 0.10 \
  --hidden 64 32
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `spo+` | Training mode: `spo+`, `mse`, `hybrid`, or `compare` |
| `--epochs` | `50` | Maximum training epochs |
| `--lr` | `0.001` | Learning rate |
| `--gamma` | `0.5` | Risk-aversion parameter (γ) in Markowitz objective |
| `--max-weight` | `0.10` | Maximum allocation per asset (concentration limit) |
| `--hidden` | `64 32` | Hidden layer dimensions for the prediction network |

### 3. Portfolio Optimization Dashboard

> **Prerequisite:** SPO models must be trained first (step 2 above).

```bash
# 4. Create the environment file (one-time setup — .env is gitignored)
echo PYTHONPATH=src > .env

# 5. Launch the interactive dashboard
python -m streamlit run app/dashboard.py
```

The dashboard runs at **http://localhost:8501** and provides:

| Feature | Description |
|---------|-------------|
| **Historical mode** | Backtest any date in the last year of data |
| **Live Prediction mode** | Fetches today's prices from Yahoo Finance and optimises for the *next* 5 trading days |
| **Optimize tab** | Portfolio weights, sector allocation, realised returns, holdings table |
| **Analysis tab** | Signal z-score scatter, risk contribution, correlation heatmap, efficient frontier |
| **Backtest tab** | Walk-forward equity curve, drawdown, rolling IC, decision regret |

> **Note for collaborators:** `.env` is gitignored for security. After cloning, create it manually:
> ```bash
> echo PYTHONPATH=src > .env
> ```
> This ensures `spo.*` and `utils.*` modules resolve correctly from any terminal inside the project.

---

## How It Works

### Stacked ML Pipeline

```
1. Base Classification    →  P(stock ∈ top quintile)  →  RandomForest / XGBoost
2. Meta-Feature Stacking  →  Append classifier probabilities to feature set
3. Regression Ranking     →  Predict continuous returns  →  Ridge / XGB / RF
4. Evaluation             →  Spearman IC · Top-K portfolio lift
```

### SPO Layer — Differentiable Predict-then-Optimize

The SPO module implements an end-to-end differentiable pipeline:

1. **Predict:** A shared-weight MLP scores each stock's expected return:
   $$\hat{\mu} = f_\theta(X_t) \in \mathbb{R}^N$$

2. **Optimize:** A differentiable Markowitz optimizer solves the QP:
   $$w^*(\hat{\mu}) = \arg\min_w \left[ -\hat{\mu}^\top w + \gamma \cdot w^\top \Sigma_t w \right] \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\ w \geq 0,\ w \leq w_{\max}$$

3. **Loss (SPO+):** The SPO+ surrogate loss (Elmachtoub & Grigas, 2021) provides a convex upper bound on true decision regret:
   $$\mathcal{L}_{\text{SPO+}}(\hat{\mu}, \mu) = \text{obj}\!\left(w^*(2\hat{\mu} - \mu),\ \mu\right) - \text{obj}\!\left(w^*(\hat{\mu}),\ \mu\right)$$

4. **Backward:** Gradients flow through the optimizer via implicit differentiation of the KKT conditions (active-set method on the QP's augmented system).

**Why this matters:**  MSE-trained models minimise prediction error but have no incentive to rank stocks correctly for portfolio construction. SPO+ directly optimises the quality of the downstream portfolio decision, producing predictions that may be less accurate but lead to better investment outcomes.

---

## Extending the Project

| Idea | Description |
|------|-------------|
| ~~Portfolio Optimization Layer~~ | ✅ Implemented via differentiable Markowitz |
| **Transaction Cost Penalty** | Add turnover regularisation `λ‖wₜ − wₜ₋₁‖₁` to the QP |
| **Advanced Features** | Volume signals, sector-neutral features, macro indicators |
| **Temporal Models** | Replace MLP with LSTM / Temporal Convolutional Network |
| **Purged CV** | Combinatorial Purged Cross-Validation for overlapping targets |
| **Risk Parity / BL** | Swap Markowitz for alternative portfolio construction |
| **Multi-Period Optimisation** | Extend to dynamic rebalancing with lookahead |

---

## References

1. Elmachtoub, A. N., & Grigas, P. (2021). **Smart "Predict, then Optimize."** *Management Science*, 68(1), 9–26. [doi:10.1287/mnsc.2020.3922](https://doi.org/10.1287/mnsc.2020.3922)

2. Markowitz, H. (1952). **Portfolio Selection.** *The Journal of Finance*, 7(1), 77–91.

3. Ledoit, O., & Wolf, M. (2004). **A well-conditioned estimator for large-dimensional covariance matrices.** *Journal of Multivariate Analysis*, 88(2), 365–411.

4. Amos, B., & Kolter, J. Z. (2017). **OptNet: Differentiable Optimization as a Layer in Neural Networks.** *ICML 2017*.

---

## Disclaimer

This project is for **educational and research purposes only** and does not constitute financial advice. Past performance and model predictions are not guarantees of future returns. Always do your own research and consult a qualified professional before making investment decisions.
