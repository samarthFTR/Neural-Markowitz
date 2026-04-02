"""
PyTorch Dataset for cross-sectional portfolio data.

Each sample represents one *trading day* — a cross-section of all
stocks with their features, forward returns, and covariance matrix.
This is the data contract the SPO training loop expects.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.logger import logging


class PortfolioCrossSectionDataset(Dataset):
    """
    Yields one cross-section per __getitem__ call.

    Shapes
    ------
    features      : (n_stocks, n_features)
    true_returns  : (n_stocks,)
    cov_matrix    : (n_stocks, n_stocks)

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataset (Date, Ticker, feature_cols…, TARGET_RETURN).
    feature_cols : list[str]
        Column names to use as predictive features.
    return_col : str
        Column containing the forward return (TARGET_RETURN).
    cov_dict : dict[str, np.ndarray]
        Mapping date-string → covariance matrix (n_stocks, n_stocks).
    tickers_order : list[str]
        Canonical ordering of tickers matching the covariance matrix columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        return_col: str = "TARGET_RETURN",
        cov_dict: dict = None,
        tickers_order: list = None,
    ):
        self.feature_cols = feature_cols
        self.return_col = return_col
        self.cov_dict = cov_dict
        self.tickers_order = tickers_order

        # ----------------------------------------------------------
        # Pre-group by date and filter to dates where we have a
        # covariance matrix AND a full cross-section of stocks.
        # ----------------------------------------------------------
        required_tickers = set(tickers_order) if tickers_order else None
        usable_dates = []

        grouped = df.groupby("Date")
        self.date_data = {}  # {idx → (features, returns)}

        for date_str, grp in grouped:
            tickers_in_day = set(grp["Ticker"].values)

            # Skip dates that don't have ALL tickers (incomplete cross-section)
            if required_tickers and not required_tickers.issubset(tickers_in_day):
                continue

            # Skip dates without covariance
            if cov_dict is not None and str(date_str) not in cov_dict:
                continue

            # Reorder rows to match canonical ticker order
            grp = grp.set_index("Ticker").loc[tickers_order].reset_index()

            features = grp[feature_cols].values.astype(np.float32)
            returns = grp[return_col].values.astype(np.float32)

            self.date_data[len(usable_dates)] = {
                "date": str(date_str),
                "features": features,
                "returns": returns,
            }
            usable_dates.append(str(date_str))

        self.dates = usable_dates
        logging.info(
            f"PortfolioCrossSectionDataset: {len(self.dates)} usable dates "
            f"(from {len(grouped)} total). "
            f"n_stocks={len(tickers_order)}, n_features={len(feature_cols)}"
        )

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        entry = self.date_data[idx]
        features = torch.tensor(entry["features"], dtype=torch.float32)
        returns = torch.tensor(entry["returns"], dtype=torch.float32)

        if self.cov_dict is not None:
            cov = torch.tensor(
                self.cov_dict[entry["date"]], dtype=torch.float32
            )
        else:
            # Fallback: identity (no risk model)
            n = features.shape[0]
            cov = torch.eye(n, dtype=torch.float32)

        return features, returns, cov

    def get_date(self, idx) -> str:
        """Return the date string for sample idx."""
        return self.date_data[idx]["date"]


def build_datasets(
    train_csv: str,
    test_csv: str,
    cov_dict: dict,
    tickers_order: list,
    feature_cols: list = None,
):
    """
    Convenience function: load CSV splits and return Dataset objects.

    Parameters
    ----------
    train_csv, test_csv : str
        Paths to the long-format CSV files produced by ingestion.py.
    cov_dict : dict
        Pre-computed covariance matrices keyed by date string.
    tickers_order : list
        Canonical ticker ordering.
    feature_cols : list or None
        If None, auto-detect (all columns except Date, Ticker, TARGET_*).

    Returns
    -------
    train_ds, test_ds : PortfolioCrossSectionDataset
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if feature_cols is None:
        exclude = {"Date", "Ticker", "TARGET_CLASS", "TARGET_RETURN"}
        feature_cols = [c for c in train_df.columns if c not in exclude]

    train_ds = PortfolioCrossSectionDataset(
        train_df, feature_cols,
        cov_dict=cov_dict, tickers_order=tickers_order,
    )
    test_ds = PortfolioCrossSectionDataset(
        test_df, feature_cols,
        cov_dict=cov_dict, tickers_order=tickers_order,
    )
    return train_ds, test_ds
