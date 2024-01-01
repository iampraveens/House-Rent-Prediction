import os
import logging

from src.utils import SaveModel
from pipelines.prediction_pipeline import predict_model

def main():
    """
    This function saves a trained model to a specified file path.
    """
    try:
        file_path = os.path.join('saved_models', 'model.pkl')
        trained_model = predict_model()
        SaveModel(model=trained_model, save_path=file_path)
        logging.info("Model Saved Successfully")
    except Exception as e:
        logging.error(f"Error in saving model {e}")
        raise e
    
if __name__ == "__main__":
    main()