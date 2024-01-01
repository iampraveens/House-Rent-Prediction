import os
import requests
from io import BytesIO
import streamlit as st 
import pandas as pd
from PIL import Image

from src.utils import LoadModel

url = "https://cdn-icons-png.flaticon.com/512/2168/2168422.png"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
page_title = 'Car Price Prediction'
page_icon = image
layout = 'wide'

st.set_page_config(page_title=page_title,
                   page_icon=page_icon,
                   layout=layout
                   )
hide_style = '''
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <style>
            
            '''
header_style = '''
             <style>
             .navbar {
                 position: fixed;
                 top: 0;
                 left: 0;
                 width: 100%;
                 z-index: 1;
                 display: flex;
                 justify-content: center;
                 align-items: center;
                 height: 80px;
                 background-color: #273D59;
                 box-sizing: border-box;
             }
             
             .navbar-brand {
                 color: white !important;
                 font-size: 23px;
                 text-decoration: none;
                 margin-right: auto;
                 margin-left: 50px;
             }
             
             .navbar-brand img {
                margin-bottom: 6px;
                margin-right: 1px;
                width: 40px;
                height: 40px;
                justify-content: center;
            }
            
            /* Add the following CSS to change the color of the text */
            .navbar-brand span {
                color: #EF6E04;
                justify-content: center;
            }
            
             </style>
             
             <nav class="navbar">
                 <div class="navbar-brand">
                <img src="https://cdn-icons-png.flaticon.com/512/2168/2168422.png" alt="Logo">
                    Car Price Prediction
                 </div>
             </nav>
               '''
st.markdown(hide_style, unsafe_allow_html=True)
st.markdown(header_style, unsafe_allow_html=True)

column_1, column_2, column_3, column_4, column_5 = st.columns([15,17,8,17,15])

owner_key = {'First Owner': 3, 'Second Owner': 2, 'Third Owner': 1, 'Fourth & Above Owner': 0}

with column_2:
    year = st.number_input("Enter made year")
    km_driven = st.number_input("Enter Km driven")
    engine = st.number_input("Enter engine")
    fuel_diesel = st.selectbox('Diesel fuel', ["Yes", "No"])
    seller_individual = st.selectbox('Individual(Seller)', ["Yes", "No"])
    
    predict_button = st.button("Predict", type='primary')
    
with column_4:
    mileage = st.number_input("Enter mileage")
    owner = st.selectbox("Enter the owner", owner_key.keys())
    transmission = st.selectbox('Transmission(Mannual)', ["Yes", "No"])
    fuel_petrol = st.selectbox('Petrol fuel', ["Yes", "No"])
    seller_trustmark_dealer = st.selectbox('Trustmark dealer(Seller)', ["Yes", "No"])
    
user_input = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'owner': [owner],
    'mileage': [mileage],
    'engine': [engine],
    'fuel_Diesel': [1 if fuel_diesel == 'Yes' else 0],
    'fuel_Petrol':  [1 if fuel_petrol == 'Yes' else 0],
    'seller_type_Individual': [1 if seller_individual == 'Yes' else 0],
    'seller_type_Trustmark Dealer': [1 if seller_trustmark_dealer == 'Yes' else 0],
    'transmission_Manual': [1 if transmission == 'Yes' else 0]
})
    
load_path = os.path.join('saved_models', 'model.pkl')
loaded_model = LoadModel(load_path=load_path)

with column_2:
    
    if predict_button:
        user_input['owner'] = user_input['owner'].map(owner_key)
        prediction = loaded_model.predict(user_input)
        # Display the prediction result
        colored1 = f'<span style="color: #F2F2F2;font-size: 17px; font-weight: bold;">Predicted Selling Price: {prediction[0]:.2f} INR.</span>'
        st.markdown(colored1, unsafe_allow_html=True)
        # st.success(f'Predicted Selling Price: {prediction[0]:.2f} INR')