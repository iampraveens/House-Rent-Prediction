import logging
import pandas as pd


class LoadData:
    """
    Load data from the specified data path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): path to the data
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Load data from the specified data path.
        
        Returns:
        A pandas DataFrame containing the loaded data.
        """
        logging.info(f"Loading data from {self.data_path}")
        return pd.read_csv(self.data_path)


def load_df(data_path: str) -> pd.DataFrame:
    """
    Load the data from data_path and convert it to a pandas DataFrame.
    
    Args:
        data_path (str): Path to the data file.
        
    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        load_data = LoadData(data_path)
        df = load_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while loading data {e}")
        raise e