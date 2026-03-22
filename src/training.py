import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import sys
from utils.exception import CustomException
from utils.utils import save_object
from utils.logger import logging

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("models","model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
    def initiate_model_train(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1:],
                test_array[:,:-1],
                test_array[:,-1:],
                )
            
            models = {
                "RandomForest": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", RandomForestRegressor())
                ]),
                "XGBoost": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", XGBRegressor())
                ])
            }
            params = {
                "RandomForest": {
                    "regressor__n_estimators":[100,200],
                    "regressor__max_depth":[10,20]
                },
                "XGBoost":{
                    "regressor__n_estimators":[100,200],
                    "regressor__max_depth":[3,5],
                    "regressor__learning_rate":[0.01,0.1],
                    "regressor__subsample": [0.7,0.9],
                    "regressor__colsample_bytree": [0.7,0.9]
                }
            }
            model_report, trained_models= ModelTraining.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score) ]
            best_model = trained_models[best_model_name]
            logging.info(f"Best found mode on dataset {best_model_name}")
            print("Test Accuracy",model_report)
            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )
            y_pred = best_model.predict(X_train)
            train_score = r2_score(y_pred=y_pred,y_true=y_train)
            return train_score
        except Exception as e:
            raise CustomException(e,sys)
    def evaluate_models(X_train,y_train,X_test,y_test,models,params):
        tscv = TimeSeriesSplit(n_splits=5)
        try:
            report = {}
            best_models = {}

            for model_name, model in models.items():

                print(f"Training {model_name}")

                param = params[model_name]
                gs = GridSearchCV(model, param,cv=tscv, n_jobs=-1)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)

                report[model_name] = score
                best_models[model_name] = best_model

                corr,_ = spearmanr(y_test.flatten(), y_pred.flatten())
                print("Spearman Rank Correlation:",corr)

                logging.info(f"{model_name} training completed")

            return report, best_models

        except Exception as e:
            raise CustomException(e,sys)
