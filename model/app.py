import streamlit as st
import pickle
import json
import numpy as np

# Load model and columns
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location.lower() in data_columns:
        loc_index = data_columns.index(location.lower())
        x[loc_index] = 1
    return model.predict([x])[0]

# UI
st.title("Bangalore House Price Predictor")
location = st.selectbox("Location", sorted([x.title() for x in data_columns[3:]]))
sqft = st.number_input("Total Square Feet", value=1000)
bath = st.number_input("Number of Bathrooms", value=2)
bhk = st.number_input("BHK", value=2)

if st.button("Predict"):
    result = predict_price(location, sqft, bath, bhk)
    st.success(f"Estimated Price: ₹ {round(result, 2)} Lakhs")
