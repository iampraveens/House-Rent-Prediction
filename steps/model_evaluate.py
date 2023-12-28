import logging

import pandas as pd
import mlflow
# from zenml import step
# from zenml.client import Client
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin

from src.evaluation import MSE, R2_Score, MAE

# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "r2_score"],
    Annotated[float, "mae"],
]:
    """
    Evaluate the performance of a model by calculating accuracy, F1 score, and recall.
    Args:
        model (ClassifierMixin): The trained model.
        X_test (pd.DataFrame): The test dataset.
        y_test (pd.DataFrame): The true labels for the test dataset.
    Returns:
        Tuple[float, float, float]: A tuple containing the accuracy, F1 score, and recall.
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric('MSE', mse)
        
        r2_class = R2_Score()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric('r2_score', r2)
        
        mae_class = MAE()
        mae = mae_class.calculate_scores(y_test, prediction)
        mlflow.log_metric('MAE', mae)
        
        return mse, r2, mae
    
    except Exception as e:
        logging.error(f"Error in evaluating the model {e}")
        raise e