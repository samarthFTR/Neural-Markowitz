from dataclasses import dataclass 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #to create pipeline
from sklearn.pipeline import Pipeline # for making pipeling
from sklearn.impute import SimpleImputer # to handle missing values
from sklearn.preprocessing import StandardScaler # to standardize the data

import os

import sys
from utils.exception import CustomException
from utils.utils import save_object 
from utils.logger import logging

@dataclass
class DataTransformationConfig:
    processed_data_path: str = os.path.join("data","processor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "AAPL_MA7","AMZN_MA7","GOOGL_MA7","MSFT_MA7","NVDA_MA7",
                "AAPL_MA30","AMZN_MA30","GOOGL_MA30","MSFT_MA30","NVDA_MA30",
                "AAPL_VOL","AMZN_VOL","GOOGL_VOL","MSFT_VOL","NVDA_VOL",
                "AAPL_MOM","AMZN_MOM","GOOGL_MOM","MSFT_MOM","NVDA_MOM"
                ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                ]
            )
            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Preprocessor: {preprocessor}')
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading test and train data completed')
            logging.info('Obtaining Preprocessing Object')  

            preprocessor = self.get_data_transformer_object()

            target_columns = ["AAPL_TARGET","AMZN_TARGET","GOOGL_TARGET","MSFT_TARGET","NVDA_TARGET"]
            numerical_columns = [
                "AAPL_MA7","AMZN_MA7","GOOGL_MA7","MSFT_MA7","NVDA_MA7",
                "AAPL_MA30","AMZN_MA30","GOOGL_MA30","MSFT_MA30","NVDA_MA30",
                "AAPL_VOL","AMZN_VOL","GOOGL_VOL","MSFT_VOL","NVDA_VOL",
                "AAPL_MOM","AMZN_MOM","GOOGL_MOM","MSFT_MOM","NVDA_MOM"
                ]

            input_feature_train_df = train_df.drop(columns=target_columns,axis=1)
            input_feature_test_df = test_df.drop(columns=target_columns,axis=1)

            logging.info(f"Applying preprocessor object on training and test dataframe")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            target_feature_train_arr = train_df[target_columns]
            target_feature_test_arr = test_df[target_columns]

            train_arr = np.concatenate((input_feature_train_arr,np.array(target_feature_train_arr)),axis=1)
            test_arr = np.concatenate((input_feature_test_arr,np.array(target_feature_test_arr)),axis=1)

            logging.info(f"Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.processed_data_path,
                obj=preprocessor
            )
            return(
               train_arr,
               test_arr,
               self.data_transformation_config.processed_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)