import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from dataclasses import dataclass
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
    classifier_model_path: str = os.path.join("models", "classifier.pkl")
    regressor_model_path: str = os.path.join("models", "regressor.pkl")
    stacked_model_path: str = os.path.join("models", "model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_train(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Array layout: [...features, TARGET_CLASS, TARGET_RETURN]
            X_train = train_array[:, :-2]
            y_train_class = train_array[:, -2]
            y_train_return = train_array[:, -1]

            X_test = test_array[:, :-2]
            y_test_class = test_array[:, -2]
            y_test_return = test_array[:, -1]

            # ============================================================
            # STAGE 1: Train Classification Models (existing base layer)
            # ============================================================
            logging.info("=== STAGE 1: Training Classification Base Models ===")

            classifiers = {
                "RandomForest": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", RandomForestClassifier(random_state=42))
                ]),
                "XGBoost": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", XGBClassifier(eval_metric='logloss', random_state=42))
                ])
            }

            clf_params = {
                "RandomForest": {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [10, 20]
                },
                "XGBoost": {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [3, 5],
                    "classifier__learning_rate": [0.01, 0.1],
                    "classifier__subsample": [0.7, 0.9],
                    "classifier__colsample_bytree": [0.7, 0.9]
                }
            }

            clf_report, trained_classifiers = ModelTraining.evaluate_classifiers(
                X_train=X_train, y_train=y_train_class,
                X_test=X_test, y_test=y_test_class,
                models=classifiers, params=clf_params
            )

            # Pick best classifier by AUC
            best_clf_name = max(clf_report, key=clf_report.get)
            best_classifier = trained_classifiers[best_clf_name]
            logging.info(f"Best classifier: {best_clf_name} (AUC: {clf_report[best_clf_name]:.4f})")
            print(f"\n--- Best Classifier: {best_clf_name} | AUC: {clf_report[best_clf_name]:.4f} ---")

            save_object(
                file_path=self.model_training_config.classifier_model_path,
                obj=best_classifier
            )

            # ============================================================
            # STAGE 2: Build Meta-Features from Classifier Probabilities
            # ============================================================
            logging.info("=== STAGE 2: Building Meta-Features for Regression Layer ===")

            # Generate probability predictions from ALL classifiers as meta-features
            meta_train_features = []
            meta_test_features = []

            for clf_name, clf_model in trained_classifiers.items():
                if hasattr(clf_model, "predict_proba"):
                    train_proba = clf_model.predict_proba(X_train)[:, 1]
                    test_proba = clf_model.predict_proba(X_test)[:, 1]
                else:
                    train_proba = clf_model.decision_function(X_train)
                    test_proba = clf_model.decision_function(X_test)

                meta_train_features.append(train_proba.reshape(-1, 1))
                meta_test_features.append(test_proba.reshape(-1, 1))

            meta_train = np.hstack(meta_train_features)
            meta_test = np.hstack(meta_test_features)

            # Augmented features = original features + classifier probabilities
            X_train_stacked = np.hstack([X_train, meta_train])
            X_test_stacked = np.hstack([X_test, meta_test])

            logging.info(f"Stacked feature shape: {X_train_stacked.shape} "
                         f"(original: {X_train.shape[1]} + meta: {meta_train.shape[1]})")

            # ============================================================
            # STAGE 3: Train Regression Models on Continuous Returns
            # ============================================================
            logging.info("=== STAGE 3: Training Regression Ranking Layer ===")

            regressors = {
                "Ridge": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge())
                ]),
                "XGBRegressor": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", XGBRegressor(random_state=42))
                ]),
                "RandomForestRegressor": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", RandomForestRegressor(random_state=42))
                ])
            }

            reg_params = {
                "Ridge": {
                    "regressor__alpha": [0.01, 0.1, 1.0, 10.0]
                },
                "XGBRegressor": {
                    "regressor__n_estimators": [100, 200],
                    "regressor__max_depth": [3, 5],
                    "regressor__learning_rate": [0.01, 0.1],
                    "regressor__subsample": [0.7, 0.9]
                },
                "RandomForestRegressor": {
                    "regressor__n_estimators": [100, 200],
                    "regressor__max_depth": [10, 20]
                }
            }

            reg_report, trained_regressors = ModelTraining.evaluate_regressors(
                X_train=X_train_stacked, y_train=y_train_return,
                X_test=X_test_stacked, y_test=y_test_return,
                models=regressors, params=reg_params
            )

            # Pick best regressor by Spearman IC
            best_reg_name = max(reg_report, key=lambda k: reg_report[k]["spearman_ic"])
            best_regressor = trained_regressors[best_reg_name]
            best_ic = reg_report[best_reg_name]["spearman_ic"]
            logging.info(f"Best regressor: {best_reg_name} (Spearman IC: {best_ic:.4f})")
            print(f"\n--- Best Regressor: {best_reg_name} | Spearman IC: {best_ic:.4f} ---")

            save_object(
                file_path=self.model_training_config.regressor_model_path,
                obj=best_regressor
            )

            # ============================================================
            # STAGE 4: Rank Stocks & Evaluate Top-K Portfolio
            # ============================================================
            logging.info("=== STAGE 4: Stock Ranking & Top-K Evaluation ===")

            y_pred_scores = best_regressor.predict(X_test_stacked)

            # Rank and pick top-K stocks (top 20% each day-equivalent slice)
            n_stocks = len(y_pred_scores)
            top_k = max(1, int(n_stocks * 0.2))

            ranked_indices = np.argsort(y_pred_scores)[::-1]  # descending
            top_k_indices = ranked_indices[:top_k]

            top_k_actual_returns = y_test_return[top_k_indices]
            overall_avg_return = np.mean(y_test_return)
            top_k_avg_return = np.mean(top_k_actual_returns)

            print(f"\n{'='*60}")
            print(f"STACKED MODEL RESULTS")
            print(f"{'='*60}")
            print(f"Classifier (base):  {best_clf_name}")
            print(f"Regressor (top):    {best_reg_name}")
            print(f"Spearman IC:        {best_ic:.4f}")
            print(f"Overall avg return: {overall_avg_return:.4f}")
            print(f"Top-{top_k} avg return:  {top_k_avg_return:.4f}")
            print(f"Top-K lift:         {top_k_avg_return - overall_avg_return:.4f}")
            print(f"{'='*60}")

            # Save the full stacked model info
            stacked_model = {
                "classifiers": trained_classifiers,
                "best_classifier_name": best_clf_name,
                "best_classifier": best_classifier,
                "best_regressor_name": best_reg_name,
                "best_regressor": best_regressor,
                "clf_report": clf_report,
                "reg_report": reg_report
            }

            save_object(
                file_path=self.model_training_config.stacked_model_path,
                obj=stacked_model
            )

            return {
                "best_classifier": best_clf_name,
                "clf_auc": clf_report[best_clf_name],
                "best_regressor": best_reg_name,
                "spearman_ic": best_ic,
                "top_k_avg_return": top_k_avg_return,
                "overall_avg_return": overall_avg_return
            }

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def evaluate_classifiers(X_train, y_train, X_test, y_test, models, params):
        """Stage 1: Train and evaluate classification models."""
        tscv = TimeSeriesSplit(n_splits=5)
        try:
            report = {}
            best_models = {}

            for model_name, model in models.items():
                print(f"\n[CLF] Training {model_name}...")

                param = params[model_name]
                gs = GridSearchCV(model, param, cv=tscv, n_jobs=-1)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                y_pred_class = best_model.predict(X_test)

                if hasattr(best_model, "predict_proba"):
                    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = best_model.decision_function(X_test)

                acc = accuracy_score(y_test, y_pred_class)
                auc = roc_auc_score(y_test, y_pred_proba)
                corr, _ = spearmanr(y_test.flatten(), y_pred_proba.flatten())

                report[model_name] = auc
                best_models[model_name] = best_model

                print(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f} | Spearman IC (proba): {corr:.4f}")
                logging.info(f"[CLF] {model_name} - Acc: {acc:.4f}, AUC: {auc:.4f}, IC: {corr:.4f}")

            return report, best_models

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def evaluate_regressors(X_train, y_train, X_test, y_test, models, params):
        """Stage 3: Train and evaluate regression models for ranking."""
        tscv = TimeSeriesSplit(n_splits=5)
        try:
            report = {}
            best_models = {}

            for model_name, model in models.items():
                print(f"\n[REG] Training {model_name}...")

                param = params[model_name]
                gs = GridSearchCV(model, param, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                corr, p_value = spearmanr(y_test.flatten(), y_pred.flatten())

                report[model_name] = {
                    "mse": mse,
                    "r2": r2,
                    "spearman_ic": corr,
                    "p_value": p_value
                }
                best_models[model_name] = best_model

                print(f"  MSE: {mse:.6f} | R²: {r2:.4f} | Spearman IC: {corr:.4f} (p={p_value:.4e})")
                logging.info(f"[REG] {model_name} - MSE: {mse:.6f}, R²: {r2:.4f}, IC: {corr:.4f}")

            return report, best_models

        except Exception as e:
            raise CustomException(e, sys)