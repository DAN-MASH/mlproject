import pandas as pd
import numpy as np

import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from dataclasses import dataclass


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    #
    trained_model_file_path:str =os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting the data training process")
            logging.info("splitting the data into X_train,y_test and X_test,y_test")

            X_train,y_train=train_array[:, :-1], train_array[:, -1]
            X_test,y_test=test_array[:,:-1], test_array[:,-1]

            #define the model to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            #model evaluation
            model_report:dict =evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            logging.info(f"Model report {model_report}")
            #select the best model based on r2_score
            best_model_score=max(sorted(model_report.values()))
            #best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            #train the data on the best model
            best_model.fit(X_train,y_train)
            #save the model
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)

            #evaluate the model
            y_pred=best_model.predict(X_test)
            r2_square=r2_score(y_test,y_pred)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)


