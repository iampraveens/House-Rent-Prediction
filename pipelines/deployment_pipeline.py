import numpy as np
import pandas as pd

from steps.load_data import load_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.model_evaluate import evaluate_model
from steps.save_model import get_data_for_test


class DeploymentTriggerConfig():
    
    min_accuracy: float = 0.5
    
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    
    return accuracy >= config.min_accuracy

def continuous_deployment_pipeline(
    min_accuracy: float = 0.5,
):

    df = load_df(data_path= r"C:\Users\sprav\Desktop\My Projects\Customer Churn Prediction\data\telcoChurn.csv")
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy, f1score, recall = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(f1score)
    
    

    
    