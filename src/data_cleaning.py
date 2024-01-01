import logging
from typing import Union
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

label_encoder = LabelEncoder()

class DataStrategy(ABC):
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        try:
            data = data.drop(columns=['id', 'activation_date', 'latitude',
                         'longitude', 'gym', 'lift', 
                          'swimming_pool', 'property_size', 'property_age', 
                          'facing', 'total_floor', 'amenities'], axis=1)
            
            data.dropna(inplace=True)
            data.drop_duplicates(inplace=True)
            
            iqr = data['bathroom'].quantile(0.75) - data['bathroom'].quantile(0.25)
            upper_thersold = data['bathroom'].quantile(0.75) + (1.5 * iqr)
            lower_thersold = data['bathroom'].quantile(0.25) - (1.5 * iqr)
            data['bathroom'] = data['bathroom'].clip(lower_thersold, upper_thersold)
            
            iqr = data['cup_board'].quantile(0.75) - data['cup_board'].quantile(0.25)
            upper_thersold = data['cup_board'].quantile(0.75) + (1.5 * iqr)
            lower_thersold = data['cup_board'].quantile(0.25) - (1.5 * iqr)
            data['cup_board'] = data['cup_board'].clip(lower_thersold, upper_thersold)
            
            iqr = data['floor'].quantile(0.75) - data['floor'].quantile(0.25)
            upper_thersold = data['floor'].quantile(0.75) + (1.5 * iqr)
            lower_thersold = data['floor'].quantile(0.25) - (1.5 * iqr)
            data['floor'] = data['floor'].clip(lower_thersold, upper_thersold)
            
            iqr = data['balconies'].quantile(0.75) - data['balconies'].quantile(0.25)
            upper_thersold = data['balconies'].quantile(0.75) + (1.5 * iqr)
            lower_thersold = data['balconies'].quantile(0.25) - (1.5 * iqr)
            data['balconies'] = data['balconies'].clip(lower_thersold, upper_thersold)
            
            iqr = data['rent'].quantile(0.75) - data['rent'].quantile(0.25)
            upper_thersold = data['rent'].quantile(0.75) + (1.5 * iqr)
            lower_thersold = data['rent'].quantile(0.25) - (1.5 * iqr)
            data['rent'] = data['rent'].clip(lower_thersold, upper_thersold)
            
            data['type'] = data['type'].map({'1BHK1': 0, 'BHK1': 0, 'RK1': 0, 
                                 'bhk2': 1, 'BHK2': 2, 'bhk3': 3, 
                                 'BHK3': 3, 'BHK4': 4, 'BHK4PLUS': 4})
            data['locality'] = label_encoder.fit_transform(data['locality'])
            data['lease_type'] = data['lease_type'].map({'COMPANY': 0, 'BACHELOR': 1, 'ANYONE': 2, 'FAMILY': 3})
            data['furnishing'] = data['furnishing'].map({'NOT_FURNISHED': 0, 'FULLY_FURNISHED': 1, 'SEMI_FURNISHED': 2})
            data['parking'] = data['parking'].map({'NONE': 0, 'TWO_WHEELER': 1, 'FOUR_WHEELER': 2, 'BOTH': 3})
            data['water_supply'] = data['water_supply'].map({'BOREWELL': 0, 'CORPORATION': 1, 'CORP_BORE': 2})
            data['building_type'] = data['building_type'].map({'GC': 0, 'IH': 1, 'AP': 2, 'IF': 3})
            
            return data
        except Exception as e:
            logging.error(f"Error while preprocessing the data {e}")
            raise e
        
class DataSplitStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(columns=['rent'], axis=1)
            y = data['rent']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error while splitting the data {e}")
            raise e
        
class DataScaleStrategy(DataStrategy):
    
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(self.X_train)
            X_test = scaler.transform(self.X_test)
            
            X_train = pd.DataFrame(X_train, columns=self.X_train.columns)
            X_test = pd.DataFrame(X_test, columns=self.X_train.columns)
            
            return X_train, X_test
        except Exception as e:
            logging.error(f"Error while scaling the data {e}")
            raise e
        
class DataCleaning:
    
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self):
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error while handling the data")
            raise e
        