import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy, DataScaleStrategy

@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        scale_strategy = DataScaleStrategy(X_train, X_test)
        data_cleaning = DataCleaning(X_train, scale_strategy)
        X_train, X_test = data_cleaning.handle_data()
        
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error while cleaning the data {e}")
        raise e