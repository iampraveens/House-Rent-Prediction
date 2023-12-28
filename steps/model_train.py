import logging

import pandas as pd
# from zenml import step
# from zenml.client import Client
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.model_dev import DecisionTreeClassifier_Model, RandomForestClassifier_Model
from src.model_dev import GridSearchCV_Model, GradientBoostingClassifier_Model
from src.model_dev import XGBoost_Model
from config import ModelNameConfig

# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> ClassifierMixin:
    """
    Trains a machine learning model based on the given configuration.
    Args:
        X_train (pd.DataFrame): Training data features.
        X_test (pd.DataFrame): Testing data features.
        y_train (pd.DataFrame): Training data labels.
        y_test (pd.DataFrame): Testing data labels.
        config (ModelNameConfig): Configuration object specifying the model name.
    Returns:
        ClassifierMixin: Trained machine learning model.
    Raises:
        ValueError: If the specified model name is not supported.
        Exception: If any error occurs during training.
    """
    try:
        model = None
        if config.model_name == 'DecisionTree':
            # mlflow.sklearn.autolog()
            model = DecisionTreeClassifier_Model()
            trained_model = model.train(X_train, y_train)
            return trained_model

        elif config.model_name == 'RandomForest':
            # mlflow.sklearn.autolog()
            model = RandomForestClassifier_Model()
            trained_model = model.train(X_train, y_train, 
                                        criterion='entropy', max_features='sqrt',
                                        max_depth=10, n_estimators=150, 
                                        n_jobs=-1, verbose=3)
            return trained_model
        
        elif config.model_name == 'GridSearchCV':
            # mlflow.sklearn.autolog()
            model = GridSearchCV_Model()
            
            trained_model = model.train(X_train, y_train,
                                        estimator=XGBClassifier(), scoring='f1',
                                        cv=5, n_jobs=-1, verbose=3)
            return trained_model
        
        elif config.model_name == 'GradientBoosting':
            # mlflow.sklearn.autolog()
            model = GradientBoostingClassifier_Model()
            trained_model = model.train(X_train, y_train, criterion='squared_error',
                                        learning_rate=0.3, max_depth=19, max_leaf_nodes=24,
                                        min_samples_leaf=9, min_samples_split=7, n_estimators=100,
                                        verbose=3, max_features='sqrt')
            return trained_model
        
        elif config.model_name == 'XGBoost':
            # mlflow.sklearn.autolog()
            model = XGBoost_Model()
            trained_model = model.train(X_train, y_train)
            return trained_model
        
        else:
            raise ValueError(f"Model not supported {config.model_name}")
        
    except Exception as e:
        logging.error(f"Error in  training mdoel {e}")
        raise e