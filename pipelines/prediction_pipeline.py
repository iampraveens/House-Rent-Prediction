
import logging
import pandas as pd

from steps.load_data import load_df
from steps.clean_data import clean_df
from steps.model_train import RandomForestClassifier_Model
from src.data_cleaning import ( DataCleaning, DataPreProcessStrategy, 
                               DataSplitStrategy, DataBalanceStrategy )


def predict_model():
    """
    Reads the telcoChurn.csv file and trains a random forest classifier model.
    
    Returns:
        trained_model (object): Trained random forest classifier model.
    """
    try:
        df = pd.read_csv(r"C:\Users\sprav\Pictures\Customer Churn Prediction\data\telcoChurn.csv")
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        proccessed_data = data_cleaning.handle_data()
        proccessed_data.drop(columns=['InternetService_No', 'gender', 'PaperlessBilling', 'StreamingTV',
                         'StreamingMovies', 'PhoneService', 'MultipleLines', 'SeniorCitizen'], 
                axis=1, inplace=True)
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(proccessed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        balance_strategy = DataBalanceStrategy(X_train, y_train)
        data_cleaning = DataCleaning(X_train, balance_strategy)
        X_train, y_train = data_cleaning.handle_data()
        
        model = RandomForestClassifier_Model()
        trained_model = model.train(X_train, y_train, 
                                        criterion='entropy', max_features='sqrt',
                                        max_depth=10, n_estimators=150, 
                                        n_jobs=-1, verbose=3)
        return trained_model
    except Exception as e:
        logging.error(f"Error in training model {e}")
        raise e