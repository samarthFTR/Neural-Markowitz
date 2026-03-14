import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass

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
                train_array[:,:-5],
                train_array[:,-5:],
                test_array[:,:-5],
                test_array[:,-5:],
                )
            models = {
                "RandomForest": RandomForestRegressor(),
                "XGBoost": XGBRegressor()
            }
            params = {
                "RandomForest": {
                    "n_estimators": [8,16,32,64,128,256],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False]
                },
                "XGBoost":{
                    "n_estimators": [100, 200, 500],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.7, 0.8, 1],
                    "colsample_bytree": [0.7, 0.8, 1],
                    "gamma": [0, 0.1, 0.3],
                    "min_child_weight": [1, 3, 5]           
                }
            }
            model_report:dict= ModelTraining.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            if best_model_score<0.75:
                raise CustomException('No Suitable Model found')
            logging.info(f"Best found mode on dataset {best_model_name}")

            best_model = models[best_model_name]
            print("Test Accuracy",model_report)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            y_pred = best_model.predict(X_train)
            train_score = accuracy_score(y_pred=y_pred,y_true=y_train)
            return train_score
        except Exception as e:
            raise CustomException(e,sys)
    def evaluate_models(X_train,y_train,X_test,y_test,models,params):
        try:
            report = {}
            for i in range(len(list(models))):
                print("Starting Training")
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=8)
                gs.fit(X=X_train,y=y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_pred = model.predict(X_test)
                model_score = accuracy_score(y_true=y_test,y_pred=y_pred)

                report[list(model.keys())[i]]=model_score
                logging.info("Training of the data is completed")
                return report

        except Exception as e:
            raise CustomException(e,sys)
