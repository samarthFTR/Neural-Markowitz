## Portfolio Optimization using Statistical & ML Models

This project builds a small end‑to‑end pipeline for **equity return forecasting** as a building block for portfolio optimization. It:

- **Downloads historical prices** for a set of large‑cap U.S. stocks using `yfinance`
- **Engineers features** such as moving averages, volatility, momentum, and lagged returns
- **Builds a supervised learning dataset** with **next‑day returns as targets**
- **Scales and preprocesses** the data with `scikit-learn` transformers
- **Trains and tunes multi‑output regression models** (Random Forest and XGBoost)
- **Persists artifacts** (preprocessor and best model) to disk for later use in downstream portfolio‑construction experiments

The current focus is on forecasting; portfolio construction (e.g. using `cvxpy` / `PyPortfolioOpt`) can be layered on top of the saved predictions.

---

## Project Structure

- **`src/`**
  - **`ingestion.py`**: Downloads OHLC data for a fixed stock universe, engineers features / targets, creates the main `portfolio_dataset.csv`, and splits it into `train.csv` and `test.csv`.
  - **`preprocessing.py`**: Defines the `DataTransformation` class, which builds a `ColumnTransformer` pipeline for numerical features, fits it on the training set, transforms train/test, and saves the fitted preprocessor to `data/processor.pkl`.
  - **`training.py`**: Defines `ModelTraining`, which:
    - splits the transformed arrays into features and 5‑dimensional target vectors (one per stock),
    - tunes `RandomForestRegressor` and `MultiOutputRegressor(XGBRegressor)` with `TimeSeriesSplit` + `GridSearchCV`,
    - selects the best model by test R² and saves it to `models/model.pkl`.
  - **`utils/utils.py`**: Helper functions for saving and loading Python objects (models, transformers).
  - **`utils/logger.py`**: Basic logging configuration that writes timestamped logs into the `logs/` directory.
  - **`utils/exception.py`**: Custom exception class with richer error messages (filename + line number).
- **`data/`**
  - **`raw data/raw.csv`**: Raw close prices downloaded from `yfinance`.
  - **`portfolio_dataset.csv`**: Engineered feature + target dataset.
  - **`train.csv`, `test.csv`**: Train / test splits of the dataset.
  - **`processor.pkl`**: Persisted preprocessing pipeline.
- **`models/`**
  - **`model.pkl`**: Persisted best‑performing regression model.
- **`logs/`**
  - Timestamped log folders created at runtime.
- **`requirements.txt`**: Python dependencies.

---

## Data & Features

The pipeline currently uses a hard‑coded universe of 5 stocks:

- `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`

Using daily close prices from `2018-01-01` to `2026-01-01`, it computes:

- **Rolling means**: 7‑day and 30‑day moving averages (suffix `_MA7`, `_MA30`)
- **Rolling volatility**: 30‑day rolling standard deviation of returns (suffix `_VOL`)
- **Momentum**: 20‑day price momentum (suffix `_MOM20`)
- **Lagged returns**: previous‑day returns (suffix `_LAG1`)
- **Targets**: next‑day returns (suffix `_TARGET`)

All features are concatenated into a single DataFrame, rows with missing values are dropped, and the final dataset is split \(75% train / 25% test\).

---

## Model Training Workflow

End‑to‑end training (data ingestion → preprocessing → model training) is driven by `src/ingestion.py` when executed as a script:

1. **Data ingestion**
   - Downloads prices with `yfinance`.
   - Builds engineered dataset and writes:
     - `data/raw data/raw.csv`
     - `data/portfolio_dataset.csv`
     - `data/train.csv`
     - `data/test.csv`
2. **Preprocessing**
   - `DataTransformation.initiate_data_transformation(train_path, test_path)`:
     - Fits a numerical pipeline (`SimpleImputer` + `StandardScaler`) on all engineered features.
     - Transforms train and test features.
     - Saves the fitted preprocessor to `data/processor.pkl`.
3. **Model training**
   - `ModelTraining.initiate_model_train(train_array, test_array)`:
     - Tunes Random Forest and XGBoost via `GridSearchCV` with `TimeSeriesSplit`.
     - Selects the best model by test R².
     - Saves the best model to `models/model.pkl`.

The script prints the training R² and logs intermediate steps to the `logs/` folder.

---

## Installation

1. **Create and activate a virtual environment** (recommended).

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   # source .venv/bin/activate  # on macOS / Linux
   ```

2. **Install dependencies**.

   From the project root (`Portfolio Optimization using SPO`):

   ```bash
   pip install -r requirements.txt
   ```

   > Note: this project uses libraries such as `yfinance`, `scikit-learn`, `xgboost`, `cvxpy`, and `PyPortfolioOpt`. Some of these may require build tools / compilers on your system.

---

## Usage

From the project root, run the end‑to‑end pipeline:

```bash
python -m src.ingestion
```

This will:

- Download and store raw price data.
- Create the engineered dataset and train/test splits.
- Fit and save the preprocessing pipeline.
- Train and tune candidate models, pick the best one, and save it to `models/model.pkl`.

After this completes, you can load the saved preprocessor and model (via `utils.load_object`) to:

- Generate return forecasts on new data.
- Feed predicted returns into portfolio‑construction / optimization logic using `cvxpy` or `PyPortfolioOpt`.

---

## Extending the Project

Ideas for further development:

- **Portfolio optimization layer**: use predicted returns and an estimated covariance matrix to generate allocations (e.g. mean‑variance optimization, risk‑parity, or custom utility).
- **Custom universes**: allow dynamic configuration of tickers and date ranges.
- **Backtesting**: implement a simple backtest loop that converts forecasts into positions and tracks PnL / risk metrics.
- **Model experimentation**: add additional models (e.g. LSTMs, temporal CNNs, gradient‑boosted trees) and richer hyperparameter searches.

---

## Disclaimer

This project is for **educational and research purposes only** and **does not constitute financial advice**. Past performance and model predictions are not guarantees of future returns. Always do your own research and consult a qualified professional before making investment decisions.

