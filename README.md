## Portfolio Optimization using Statistical & ML Models

This project builds an end‑to‑end pipeline for **equity return forecasting and stock ranking** as a core building block for portfolio optimization. It uses a **stacked Machine Learning architecture** combining classification and regression models.

- **Downloads historical prices** for a diverse universe of 40+ large‑cap U.S. stocks across different sectors using `yfinance`.
- **Engineers alpha features** such as momentum, volatility, relative strength, rank features, and explicit mean-reversion signals.
- **Builds a supervised learning dataset** with dual targets:
  - `TARGET_CLASS`: A binary indicator for stocks in the daily top quintile of returns.
  - `TARGET_RETURN`: The continuous forward 5-day return.
- **Scales and preprocesses** data natively in long-format using `scikit-learn` transformers.
- **Trains a Stacked Model Architecture**:
  - **Base Layer (Classification)**: Predicts the probability of a stock being in the top quintile.
  - **Meta-Feature Building**: Uses classifier probabilities as new features.
  - **Top Layer (Regression)**: Predicts continuous returns using original features plus classifier probabilities to correctly rank stocks.
- **Evaluates with Quant Metrics**: Evaluates models using Spearman Rank Correlation (IC) and Top-K portfolio lift.
- **Persists artifacts** (preprocessor and the full stacked model) to disk for later use in downstream portfolio‑construction.

---

## Project Structure

- **`src/`**
  - **`ingestion.py`**: Downloads OHLC data for the stock universe, engineers features and dual targets, melts data into a long format, creates the main `portfolio_dataset.csv`, and performs a time-aware train/test split.
  - **`preprocessing.py`**: Defines `DataTransformation`, which builds a `ColumnTransformer` for numerical features, fits it on the training set, transforms data, stacks both targets back into the output arrays, and saves the preprocessor to `data/processor.pkl`.
  - **`training.py`**: Defines `ModelTraining`, which executes the stacked pipeline:
    - Splits arrays into features, `TARGET_CLASS`, and `TARGET_RETURN`.
    - **Stage 1**: Trains and tunes classifiers (RandomForest, XGBoost) using `TimeSeriesSplit`.
    - **Stage 2**: Generates out-of-fold probability predictions as meta-features.
    - **Stage 3**: Trains and tunes regressors (Ridge, XGBRegressor, RandomForestRegressor) on augmented features.
    - **Stage 4**: Evaluates ranking performance using Spearman IC and Top-K portfolio return.
    - Saves individual models and the stacked ensemble to `models/`.
  - **`spo/`** *(NEW — Smart Predict-then-Optimize Layer)*:
    - **`prediction_net.py`**: PyTorch neural network for cross-sectional return prediction (shared-weight MLP).
    - **`portfolio_layer.py`**: Differentiable Markowitz mean-variance optimizer with custom autograd backward pass (implicit KKT differentiation).
    - **`spo_loss.py`**: SPO+ surrogate loss (Elmachtoub & Grigas, 2021), decision regret metric, and hybrid MSE+SPO+ loss.
    - **`covariance.py`**: Rolling covariance estimation with Ledoit-Wolf shrinkage.
    - **`trainer.py`**: End-to-end SPO training pipeline with three modes (`spo+`, `mse`, `hybrid`) and a comparison runner.
    - **`evaluation.py`**: Walk-forward portfolio backtester with financial metrics (Sharpe, drawdown, turnover, decision regret).
  - **`utils/datasets.py`**: PyTorch `Dataset` for cross-sectional portfolio data (one sample = one trading day of all stocks).
  - **`utils/utils.py`**: Helper functions for saving and loading Python objects.
  - **`utils/logger.py`**: Basic logging configuration writing to `logs/`.
  - **`utils/exception.py`**: Custom exception class for rich error tracing.
- **`data/`**
  - **`raw data/raw.csv`**: Raw close prices downloaded from `yfinance`.
  - **`portfolio_dataset.csv`**: Engineered feature + target dataset in long format.
  - **`train.csv`, `test.csv`**: Train / test splits (split chronologically).
  - **`processor.pkl`**: Persisted preprocessing pipeline.
- **`models/`**
  - **`classifier.pkl`**: Best performing base classifier.
  - **`regressor.pkl`**: Best performing ranking regressor.
  - **`model.pkl`**: Full stacked model dictionary containing all reports and best estimators.
  - **`spo/`**: SPO-trained prediction networks and backtest results per training mode.
- **`logs/`**
  - Timestamped log folders created at runtime.
- **`requirements.txt`**: Python dependencies.

---

## Data & Features

The pipeline uses a diversified universe of large-cap stocks across sectors (e.g., AAPL, MSFT, JPM, WMT, XOM, JNJ, etc.).

Using daily close prices from `2018-01-01` to `2026-01-01`, it computes:

- **Base Alpha Features**: 1D, 5D, 10D returns.
- **Momentum**: 10-day and 20-day momentum.
- **Volatility**: 5-day and 10-day rolling standard deviations.
- **Relative Strength**: Stock return minus cross-sectional mean return (`ALPHA_1D`).
- **Rank & Regime Features**: Cross-sectional momentum rank (`RANK_MOM_10`) and anti-momentum (modeling short-term mean reversion).
- **Dual Targets**:
  - `TARGET_RETURN`: Forward 5-day return (`shift(-5)`).
  - `TARGET_CLASS`: 1 if forward return is in the top 20% cross-sectionally for that day, else 0.

The dataset is reshaped into a **long format** (`Date`, `Ticker`, `Features...`) ensuring that cross-sectional operations are handled correctly.

---

## Model Training Workflow

End‑to‑end training is driven by executing `src/ingestion.py`:

1. **Data Ingestion**
   - Downloads prices via `yfinance`.
   - Computes features, generates targets, and builds the long-format dataset.
   - Performs a chronological 75/25 split to prevent lookahead bias.
2. **Preprocessing**
   - Fits a pipeline (`SimpleImputer` + `StandardScaler` (passed through)) on training features.
   - Outputs Numpy arrays containing `[...features, TARGET_CLASS, TARGET_RETURN]`.
3. **Stacked Model Training**
   - **Base Classification**: Evaluates RandomForest and XGBoost on `TARGET_CLASS` using GridSearch + TimeSeriesSplit. Selects the best by AUC.
   - **Meta-Feature Creation**: Stacks probability outputs from all classifiers alongside the original features.
   - **Regression Ranking Layer**: Evaluates Ridge, XGBRegressor, and RandomForestRegressor on the augmented dataset against `TARGET_RETURN`.
   - **Evaluation**: Ranks test set stocks based on regressor predictions, selects the Top-K (top 20%), and computes the Spearman Information Coefficient (IC) and Top-K average return lift.

---

## Installation

1. **Create and activate a virtual environment** (recommended).

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   # source .venv/bin/activate  # on macOS / Linux
   ```

2. **Install dependencies**.

   From the project root:

   ```bash
   pip install -r requirements.txt
   ```

   > Note: This requires libraries like `xgboost`, `scikit-learn`, `yfinance`, and eventually `cvxpy` / `PyPortfolioOpt`.

---

## Usage

From the project root, run the end‑to‑end pipeline:

```bash
python src/ingestion.py
```

This will:
- Create required directories (`data/raw data/`, `models/`, `logs/`).
- Download data, build features, and split the data.
- Fit and save the scaler/imputer.
- Train the stacked classification-regression architecture.
- Print model performance (AUC for classifiers, Spearman IC and Top-K return for regressors).
- Save the final models to the `models/` directory.

### SPO Layer (Decision-Focused Learning)

After running `ingestion.py` at least once (to generate train/test CSVs and raw prices), run the SPO pipeline:

```bash
# Train with SPO+ loss (decision-focused)
python src/spo/trainer.py --mode spo+ --epochs 50

# Train with MSE loss (prediction-focused baseline)
python src/spo/trainer.py --mode mse --epochs 50

# Train with hybrid loss (blended)
python src/spo/trainer.py --mode hybrid --epochs 50

# Run all three and print a side-by-side comparison
python src/spo/trainer.py --mode compare
```

Additional options:
```bash
python src/spo/trainer.py --mode spo+ --epochs 30 --lr 0.001 --gamma 0.5 --max-weight 0.10 --hidden 64 32
```

---

## Extending the Project

Ideas for further development:

- ~~**Portfolio Optimization Layer**~~: ✅ Implemented via the SPO module using differentiable Markowitz optimization.
- **Advanced Feature Engineering**: Incorporate volume data, sector-neutralize features, or add macroeconomic indicators.
- **Deep Learning**: Introduce LSTMs or Temporal Convolutional Networks (TCNs) into the prediction network.
- **Purged Cross-Validation**: Implement Combinatorial Purged Cross-Validation (CPCV) to better handle time-series overlap in forward returns.
- **Transaction Costs**: Add explicit turnover penalty `λ‖w_t − w_{t−1}‖₁` to the Markowitz objective.
- **Risk Parity / Black-Litterman**: Swap the Markowitz layer for alternative portfolio construction methods.

---

## Disclaimer

This project is for **educational and research purposes only** and **does not constitute financial advice**. Past performance and model predictions are not guarantees of future returns. Always do your own research and consult a qualified professional before making investment decisions.
