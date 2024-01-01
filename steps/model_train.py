import logging

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV 
from xgboost import XGBRegressor

from src.model_dev import LinearRegression_Model, ElasticNet_Model, RandomForestRegressor_Model
from src.model_dev import GradientBoostingRegressor_Model, XGBoostRegressor_Model, GridSearchCV_Model
from config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    try:
        model = None
        if config.model_name == 'LinearRegression':
            model = LinearRegression_Model()
            trained_model = model.train(X_train, y_train)
            return trained_model
        
        elif config.model_name == 'ElasticNet':
            model = ElasticNet_Model()
            trained_model = model.train(X_train, y_train, 
                                        alpha=0.1, l1_ratio=0.3)
            return trained_model
        
        elif config.model_name == 'RandomForestRegressor':
            model = RandomForestRegressor_Model()
            trained_model = model.train(X_train, y_train, n_jobs=-1,
                                        criterion='squared_error', max_depth=10,
                                        max_features='sqrt', verbose=2, n_estimators=100)
            return trained_model
        
        elif config.model_name == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor_Model()
            trained_model = model.train(X_train, y_train, learning_rate=0.1,
                                        max_depth=5, n_estimators=300, min_samples_leaf=2,
                                        min_samples_split=2, verbose=2)
            return trained_model
        
        elif config.model_name == 'XGBoostRegressor':
            model = XGBoostRegressor_Model()
            trained_model = model.train(X_train, y_train, learning_rate=0.2,
                                        max_depth=4, n_estimators=300, reg_lambda=0.2)
            return trained_model
        
        elif config.model_name == 'GridSearchCV':
            model = GridSearchCV_Model()
            estimator = XGBRegressor()
            trained_model = model.train(X_train, y_train, estimator=estimator,
                                        scoring='r2', cv=5, n_jobs=-1, verbose=3)
            return trained_model.best_estimator_
        
        else:
            raise ValueError(f"Model not supported {config.model_name}")
    except Exception as e:
        logging.error(f"Error in training model {e}")
        raise e


