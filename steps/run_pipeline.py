import os
 
from pipelines.training_pipeline import train_pipeline
# sys.path.append("../pipelines") 

if __name__ == "__main__":
    
    # Run the pipeline
    data_path = os.path.join('data', 'house_rent.csv')
    train_pipeline(data_path=data_path)

