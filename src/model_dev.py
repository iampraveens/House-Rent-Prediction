from abc import ABC, abstractmethod
import logging

import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV 
from xgboost import XGBRegressor

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
class LinearRegression_Model(Model):
    
    def train(self, X_train, y_train, **kwargs):
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            logging.info(f"Model training completed")
            return lr
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class ElasticNet_Model(Model):
    
    def train(self, X_train, y_train, alpha: float, l1_ratio: float, **kwargs):
        try:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            en.fit(X_train, y_train)
            logging.info(f"Model training completed")
            mlflow.log_param('alpha', alpha)
            mlflow.log_param('l1_ratio', l1_ratio)
            return en
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class RandomForestRegressor_Model(Model):
    
    def train(self, X_train, y_train, criterion: str, max_depth: int, 
              max_features: str, verbose: int, n_estimators: int, n_jobs: int, **kwargs):
        try:
            rf = RandomForestRegressor(criterion=criterion, max_depth=max_depth, 
                                       max_features=max_features, verbose=verbose, 
                                       n_estimators=n_estimators, n_jobs=n_jobs)
            rf.fit(X_train, y_train)
            logging.info(f"Model training completed")
            mlflow.log_param('criterion', criterion)
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_param('n_estimators', n_estimators)
            return rf
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class GridSearchCV_Model(Model):
    
    def train(self, X_train, y_train, estimator,
              scoring: str, cv: int, verbose: int, 
              n_jobs: int, **kwargs):
        try:
            param_grid_dict = {
                    "RandomForestRegressor":{
                        'n_estimators': [100, 150, 300],                
                        'criterion': ['absolute_error', 'squared_error'],               
                        'max_depth': [None, 10, 20, 30],               
                        'max_features': ['sqrt', 'log2', None]
                    },
                    "GradientBoostingRegressor":{
                        'n_estimators': [100, 200, 300], 
                        'learning_rate': [0.01, 0.1, 0.2],  
                        'max_depth': [None, 10, 20, 30]
                    },
                    "XGBRegressor":{
                        'n_estimators': [100, 150, 300],  
                        'learning_rate': [0.2], 
                        'max_depth': [None, 10, 20, 30],
                        'reg_alpha': [0, 0.1, 0.01], 
                        'reg_lambda': [1, 2, 0.5] 
                    }
                }
            estimator_name = estimator.__class__.__name__
            param_grid = param_grid_dict.get(estimator_name, {})
            gs = GridSearchCV(estimator=estimator, scoring=scoring, param_grid=param_grid,
                              cv=cv, verbose=verbose, n_jobs=n_jobs)
            gs.fit(X_train, y_train)
            logging.info(f"Model training completed")
            return gs
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class GradientBoostingRegressor_Model(Model):
    
    def train(self, X_train, y_train, learning_rate: float, max_depth: int, 
              n_estimators: int, min_samples_leaf: int,
              min_samples_split: int, verbose: int, **kwargs):
        try:
            gb = GradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth, 
                                           n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split, verbose=verbose)
            gb.fit(X_train, y_train)
            logging.info(f"Model training completed")
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_param('n_estimators', n_estimators)
            return gb
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e
        
class XGBoostRegressor_Model(Model):
    
    def train(self, X_train, y_train, learning_rate: float, max_depth: int, 
              n_estimators: int, reg_lambda: float, **kwargs):
        try:
            xgb = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, 
                               n_estimators=n_estimators, reg_lambda=reg_lambda)
            xgb.fit(X_train, y_train)
            logging.info(f"Model training completed")
            # fim = pd.Series(xgb.feature_importances_, index= X_train.columns)
            # print(fim.sort_values(ascending=False))
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('max_depth', max_depth)
            mlflow.log_param('n_estimators', n_estimators)
            return xgb
        except Exception as e:
            logging.error(f"Error in training model {e}")
            raise e