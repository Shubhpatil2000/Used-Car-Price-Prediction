import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load label encoder, preprocessor, and trained model
le_model = joblib.load("label_encoder_model.joblib")

# Ensure the path to preprocessor and model is correct

preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("best_model.joblib")

# App title
st.title("Used Car Price Predictor 🚗💰")

# Input form
st.header("Enter Car Details")

# model_input = st.text_input("Car Model (e.g., Swift, Fortuner, Alto,etc)")
model_input = st.selectbox("Car Model", ['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'i10', 'Venue',
       'Swift', 'Verna', 'Duster', 'Cooper', 'Ciaz', 'C-Class', 'Innova',
       'Baleno', 'Swift Dzire', 'Vento', 'Creta', 'City', 'Bolero',
       'Fortuner', 'KWID', 'Amaze', 'Santro', 'XUV500', 'KUV100', 'Ignis',
       'RediGO', 'Scorpio', 'Marazzo', 'Aspire', 'Figo', 'Vitara',
       'Tiago', 'Polo', 'Seltos', 'Celerio', 'GO', '5', 'CR-V',
       'Endeavour', 'KUV', 'Jazz', '3', 'A4', 'Tigor', 'Ertiga', 'Safari',
       'Thar', 'Hexa', 'Rover', 'Eeco', 'A6', 'E-Class', 'Q7', 'Z4', '6',
       'XF', 'X5', 'Hector', 'Civic', 'D-Max', 'Cayenne', 'X1', 'Rapid',
       'Freestyle', 'Superb', 'Nexon', 'XUV300', 'Dzire VXI', 'S90',
       'WR-V', 'XL6', 'Triber', 'ES', 'Wrangler', 'Camry', 'Elantra',
       'Yaris', 'GL-Class', '7', 'S-Presso', 'Dzire LXI', 'Aura', 'XC',
       'Ghibli', 'Continental', 'CR', 'Kicks', 'S-Class', 'Tucson',
       'Harrier', 'X3', 'Octavia', 'Compass', 'CLS', 'redi-GO', 'Glanza',
       'Macan', 'X4', 'Dzire ZXI', 'XC90', 'F-PACE', 'A8', 'MUX',
       'GTC4Lusso', 'GLS', 'X-Trail', 'XE', 'XC60', 'Panamera', 'Alturas',
       'Altroz', 'NX', 'Carnival', 'C', 'RX', 'Ghost', 'Quattroporte',
       'Gurkha'])
vehicle_age = st.number_input("Vehicle Age (in years)", min_value=0, max_value=50, value=5)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
mileage = st.number_input("Mileage (e.g., 18.0)", min_value=0.0, value=18.0)
engine = st.number_input("Engine CC (e.g., 1197)", min_value=0, value=1197)
max_power = st.number_input("Max Power (e.g., 82.0)", min_value=0.0, value=82.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

# Predict Button
if st.button("Predict Selling Price"):
    try:
        # Encode the model name
        model_encoded = le_model.transform([model_input])[0]

        # Create DataFrame for the input
        input_df = pd.DataFrame({
            'model': [model_encoded],
            'vehicle_age': [vehicle_age],
            'km_driven': [km_driven],
            'seller_type': [seller_type],
            'fuel_type': [fuel_type],
            'transmission_type': [transmission_type],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats]
        })

        # Apply preprocessing (note: preprocessor excludes 'model')
        input_preprocessed = preprocessor.transform(input_df)

        # Predict selling price
        prediction = model.predict(input_preprocessed)[0]
        st.success(f"Estimated Selling Price: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
"Price Reduction Corporation" 




