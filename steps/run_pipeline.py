import os

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    
    data_path = os.path.join('data', 'house_rent.csv')
    train_pipeline(data_path=data_path)

