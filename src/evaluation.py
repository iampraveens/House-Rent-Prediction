import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
    
class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE Score")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE Score: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE {e}")
            raise e
         
class R2_Score(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 {e}")
            raise e
        
class MAE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MAE Score")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE Score: {mae}")
            return mae
        except Exception as e:
            logging.error(f"Error in calculating MAE {e}")
            raise e