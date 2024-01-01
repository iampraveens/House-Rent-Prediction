import os
import mlflow
from urllib.parse import urlparse 

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    
    with mlflow.start_run(nested=True):
        remote_server_uri = "https://dagshub.com/iampraveens/Car-Price-Prediction-MLOps.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_store_type = urlparse(mlflow.get_tracking_uri()).scheme
        
        data_path = os.path.join('data', 'car_data.csv')
        model = train_pipeline(data_path=data_path)
        
        if tracking_url_store_type != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="XGBoostRegressor")
        else:
            mlflow.sklearn.log_model(model, "model")
            