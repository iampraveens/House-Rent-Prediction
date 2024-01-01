import joblib
 
def SaveModel(model, save_path):
    try:
        with open(save_path, 'wb') as file:
            joblib.dump(model, file)
    except Exception as e:
        raise e
    
def LoadModel(load_path):
    try:
        with open(load_path, 'rb') as file:
            return joblib.load(file)
    except Exception as e:
        raise e