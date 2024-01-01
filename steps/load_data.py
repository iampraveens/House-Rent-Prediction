import logging

import pandas as pd
from zenml import step

class LoadData:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def get_data(self):
        logging.info(f"Loading the data {self.data_path}")
        return pd.read_csv(self.data_path)
@step   
def load_df(data_path: str) -> pd.DataFrame:
    
    try:
        load_data = LoadData(data_path)
        df = load_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while loading the data {e}")
        raise e