import os
import requests
from io import BytesIO
import streamlit as st 
import pandas as pd
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from src.utils import LoadModel

url = "https://cdn-icons-png.flaticon.com/512/13644/13644020.png"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
page_title = 'House Rent Prediction'
page_icon = image
layout = 'wide'

st.set_page_config(page_title=page_title,
                   page_icon=page_icon,
                   layout=layout
                   )

def locality_class():
    label_encoder = LabelEncoder()
    data = pd.read_csv(os.path.join('data', 'house_rent.csv'))
    data = data.dropna()
    data = data.drop_duplicates()
    data['locality'] = label_encoder.fit_transform(data['locality'])

    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # print(label_mapping)
    return label_mapping

type_key = {'BHK1': 0,'BHK2': 2, 
        'BHK3': 3, 'BHK4': 4}
location_key = locality_class()
lease_type_key = {'COMPANY': 0, 'BACHELOR': 1, 'ANYONE': 2, 'FAMILY': 3}
furnishing_key = {'NOT_FURNISHED': 0, 'FULLY_FURNISHED': 1, 'SEMI_FURNISHED': 2}
parking_key = {'NONE': 0, 'TWO_WHEELER': 1, 'FOUR_WHEELER': 2, 'BOTH': 3}
water_supply_key = {'BOREWELL': 0, 'CORPORATION': 1, 'CORP_BORE': 2}
building_type_key = {'GC': 0, 'IH': 1, 'AP': 2, 'IF': 3}



location = st.selectbox('Enter the location', locality_class().keys())

column_1, column_2, column_3, column_4, column_5 = st.columns([15,6,15,6,15])

with column_1:
    type = st.selectbox('Select house type', type_key.keys())
    lease_type = st.selectbox('Select lease type', lease_type_key.keys())
    negotiable = st.selectbox('Select Negotiable', ['Yes', 'No'])
    furnishing = st.selectbox('Select Furnishing', furnishing_key.keys())
    
    predict_button = st.button("Predict", type='primary')
    
with column_3:
    parking = st.selectbox('Select Parking', parking_key.keys())
    bathroom = st.number_input('Enter no of bathrooms')
    cup_board = st.number_input('Enter no of cup boards')
    floor = st.number_input('Enter no of floors')
    
with column_5:
    water_supply = st.selectbox('Select water supply', water_supply_key.keys())
    building_type = st.selectbox('Select building type', building_type_key.keys())
    balconies = st.number_input('Enter no of balconies')
    
user_input = pd.DataFrame({
    'type': [type],
    'locality': [location],
    'lease_type': [lease_type],
    'negotiable': [1 if negotiable == 'Yes' else 0],
    'furnishing': [furnishing],
    'parking': [parking],
    'bathroom': [bathroom],
    'cup_board': [cup_board],
    'floor': [floor],
    'water_supply': [water_supply],
    'building_type': [building_type],
    'balconies': [balconies]
})

load_path = os.path.join('saved_models', 'model.pkl')
loaded_model = LoadModel(load_path=load_path)

with column_1:
    if predict_button:
        user_input['type'].map(type_key)
        user_input['locality'].map(location_key)
        user_input['lease_type'].map(lease_type_key)
        user_input['furnishing'].map(furnishing_key)
        user_input['parking'].map(parking_key)
        user_input['water_supply'].map(water_supply_key)
        user_input['building_type'].map(building_type_key)
        
        user_input = user_input.fillna(0)
        user_input = user_input.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        prediction = loaded_model.predict(user_input)
        st.success(*prediction)