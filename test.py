import pickle 
import os 
import numpy as np

model_path = os.path.join('saved_models', 'model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)
    
    
input_data = (0,228,3,1,2,1,1.0,1.0,1.0,1,1,0.0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction[0])