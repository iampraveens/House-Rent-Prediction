import os
import logging

from steps.load_data import load_df
from steps.model_train import (XGBoostRegressor_Model, GradientBoostingRegressor_Model, 
                               RandomForestRegressor_Model)
from src.data_cleaning import ( DataCleaning, DataPreProcessStrategy, 
                               DataSplitStrategy, DataScaleStrategy )


def predict_model():

    try:
        data_path = os.path.join('data', 'house_rent.csv')
        df = load_df(data_path=data_path)
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        proccessed_data = data_cleaning.handle_data()
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(proccessed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        # scale_strategy = DataScaleStrategy(X_train, y_train)
        # data_cleaning = DataCleaning(X_train, scale_strategy)
        # X_train, y_train = data_cleaning.handle_data()
        
        model = XGBoostRegressor_Model()
        trained_model = model.train(X_train, y_train, learning_rate=0.3,
                                        max_depth=4, n_estimators=300, reg_lambda=0.2)
        
        return trained_model
    except Exception as e:
        logging.error(f"Error in training model {e}")
        raise e