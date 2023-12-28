import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Evaluation(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates evaluation scores based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        pass
    
class MSE(Evaluation):
    """
    Class for calculating accuracy score as an evaluation metric.
    Inherits from the Evaluation abstract class.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates accuracy score based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        try:
            logging.info("Calculating Accuracy Score")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Accuracy Score: {mse}")
            return mse
        
        except Exception as e:
            logging.error(f"Error in calculating Accuracy Score {e}")
            raise e
        
class R2_Score(Evaluation):
    """
    Class for calculating F1 score as an evaluation metric.
    Inherits from the Evaluation abstract class.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates F1 score based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: F1 score.
        """
        try:
            logging.info("Calculating F1 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"F1 Score: {r2}")
            return r2
        
        except Exception as e:
            logging.error(f"Error in calculating F1 Score {e}")
            raise e
        
class MAE(Evaluation):
    """
    Class for calculating recall score as an evaluation metric.
    Inherits from the Evaluation abstract class.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates recall score based on true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Recall score.
        """
        try:
            logging.info("Calculating Recall Score")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE Score: {mae}")
            return mae
        
        except Exception as e:
            logging.error(f"Error in calculating Recall Score {e}")
            raise e