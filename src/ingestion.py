import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import yfinance as yf
from utils.exception import CustomException
from utils.logger import logging
from preprocessing import DataTransformation
from training import ModelTraining
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("data","train.csv")
    test_data_path: str = os.path.join("data","test.csv")
    raw_data_path: str = os.path.join("data","raw data","raw.csv")
    portfolio_dataset_path: str = os.path.join("data","portfolio_dataset.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            stocks = ["AAPL","MSFT","GOOGL","AMZN","NVDA"]
            data = yf.download(stocks, start="2018-01-01", end="2026-01-01")
            close = data["Close"]
            close.to_csv(self.ingestion_config.raw_data_path)
            logging.info("Saved the raw data to csv file")
            
            returns = close.pct_change()
            future_returns = (close.shift(-1) / close) - 1

            # Fix non-stationarity by dividing by current price
            ma7_dev = (close.rolling(7).mean() / close) - 1
            ma30_dev = (close.rolling(30).mean() / close) - 1
            
            volatility = returns.rolling(30).std()
            momentum20 = close / close.shift(20) - 1
            lag1 = returns.shift(1)
            lag0 = returns.copy() # Current day return

            # Function to compute daily cross-sectional Z-scores
            def cross_sectional_zscore(df):
                return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)

            ma7_cs = cross_sectional_zscore(ma7_dev)
            ma30_cs = cross_sectional_zscore(ma30_dev)
            vol_cs = cross_sectional_zscore(volatility)
            mom20_cs = cross_sectional_zscore(momentum20)
            lag1_cs = cross_sectional_zscore(lag1)
            lag0_cs = cross_sectional_zscore(lag0)

            # Convert wide formats to long format
            dataset = pd.concat([
                ma7_cs.stack(),
                ma30_cs.stack(),
                vol_cs.stack(),
                mom20_cs.stack(),
                lag1_cs.stack(),
                lag0_cs.stack(),
                future_returns.stack()
            ], axis=1, keys=['MA7', 'MA30', 'VOL', 'MOM20', 'LAG1', 'LAG0', 'TARGET'])
            
            dataset = dataset.reset_index()
            dataset.rename(columns={'level_0': 'Date', 'level_1': 'Ticker'}, inplace=True)
            
            # Drop NaNs after stacking
            dataset = dataset.dropna()
            
            # Sort chronologically
            dataset = dataset.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)

            dataset.to_csv(self.ingestion_config.portfolio_dataset_path, index=False)
            logging.info("Saved the processed long format data to csv file")
            
            # Time-Series split needs to respect exact Date boundaries in long format
            dates = dataset['Date'].unique()
            split_date = dates[int(len(dates) * 0.75)]
            
            train_set = dataset[dataset['Date'] < split_date]
            test_set = dataset[dataset['Date'] >= split_date]
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    Data_Transformation=DataTransformation()
    train_arr,test_arr,_=Data_Transformation.initiate_data_transformation(train_data,test_data)
    modeltrain = ModelTraining()
    print(modeltrain.initiate_model_train(test_array=test_arr,train_array=train_arr))