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

            # 1. Base Alpha Features
            ret_1d  = close.pct_change(1)
            ret_5d  = close.pct_change(5)
            ret_10d = close.pct_change(10)

            mom_10 = close / close.shift(10) - 1
            mom_20 = close / close.shift(20) - 1

            vol_5  = ret_1d.rolling(5).std()
            vol_10 = ret_1d.rolling(10).std()

            # 2. Relative Strength
            market_ret = ret_1d.mean(axis=1)
            alpha_1d = ret_1d.sub(market_ret, axis=0)

            # 3. Rank & Regime Features
            rank_mom_10 = mom_10.rank(axis=1)
            anti_mom_10 = -mom_10  # Explicitly model short-term mean reversion

            # Convert wide formats to long format natively using raw signals
            dataset = pd.concat([
                ret_1d.stack(),
                ret_5d.stack(),
                ret_10d.stack(),
                mom_10.stack(),
                mom_20.stack(),
                vol_5.stack(),
                vol_10.stack(),
                alpha_1d.stack(),
                rank_mom_10.stack(),
                anti_mom_10.stack(),
                future_returns.stack()
            ], axis=1, keys=[
                'RET_1D', 'RET_5D', 'RET_10D', 
                'MOM_10', 'MOM_20', 
                'VOL_5', 'VOL_10', 
                'ALPHA_1D', 'RANK_MOM_10', 
                'ANTI_MOM_10',
                'TARGET'
            ])
            
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