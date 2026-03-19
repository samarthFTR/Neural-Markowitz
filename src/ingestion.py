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
            stocks = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","NFLX","TSLA","ADBE","INTC","AMD"]
            data = yf.download(stocks, start="2018-01-01", end="2026-01-01")
            close = data["Close"]
            close.to_csv(self.ingestion_config.raw_data_path)
            logging.info("Saved the raw data to csv file")
            returns = close.pct_change()
            future_returns = (close.shift(-1) / close) - 1

            ma7 = close.rolling(7).mean()
            ma30 = close.rolling(30).mean()

            volatility = returns.rolling(30).std()

            momentum20 = close / close.shift(20) - 1

            lag1 = returns.shift(1)

            dataset = pd.concat(
                [
                    ma7.add_suffix("_MA7"),
                    ma30.add_suffix("_MA30"),
                    volatility.add_suffix("_VOL"),
                    momentum20.add_suffix("_MOM20"),
                    lag1.add_suffix("_LAG1"),
                    future_returns.add_suffix("_TARGET"),
                ],
                axis=1,
            )
            dataset = dataset.dropna()
            dataset.to_csv(self.ingestion_config.portfolio_dataset_path)
            logging.info("Saved the processed data to csv file")
            split_index = int(len(dataset) * 0.75)
            train_set = dataset.iloc[:split_index]
            test_set = dataset.iloc[split_index:]
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
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