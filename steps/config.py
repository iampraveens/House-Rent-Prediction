from zenml.steps import BaseParameters
# from pydantic import BaseModel

class ModelNameConfig(BaseParameters):
    """
    Configuration class for the model name.
    
    Attributes:
        model_name (str): The name of the model. Default is 'RandomForest'.
    """
    model_name: str = 'RandomForest'