import pandas as pd
import yfinance as yf

stocks = ["AAPL","MSFT","GOOGL","AMZN","NVDA"]
data = yf.download(stocks, start="2018-01-01", end="2026-01-01")

close = data["Close"]

close.to_csv("data/raw data/stock_prices.csv")
data = pd.read_csv("data/raw data/stock_prices.csv")
close = data.iloc[:, 1:]
returns = close.pct_change()
future_returns = returns.shift(-1)
ma7 = close.rolling(7).mean()
ma30 = close.rolling(30).mean()
volatility = returns.rolling(30).std()
momentum = close / close.shift(7) - 1
dataset = pd.concat(
    [
        ma7.add_suffix("_MA7"),
        ma30.add_suffix("_MA30"),
        volatility.add_suffix("_VOL"),
        momentum.add_suffix("_MOM"),
        future_returns.add_suffix("_TARGET"),
    ],
    axis=1,
)
dataset = dataset.dropna()
dataset.to_csv("data/processed data/portfolio_dataset.csv")
