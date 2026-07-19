"""
app.py
Streamlit app to predict concrete compressive strength from mix design inputs.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler (saved by src/train_model.py)
model = joblib.load("src/best_model.pkl")
scaler = joblib.load("src/scaler.pkl")

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱", layout="centered")

st.title("🧱 Concrete Compressive Strength Predictor")
st.write("Enter the mix design values below to predict compressive strength (MPa).")

st.divider()

col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cement (kg/m³)", min_value=0.0, max_value=600.0, value=280.0)
    slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=400.0, value=0.0)
    fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=250.0, value=0.0)
    water = st.number_input("Water (kg/m³)", min_value=0.0, max_value=300.0, value=180.0)

with col2:
    superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=35.0, value=5.0)
    coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, max_value=1200.0, value=970.0)
    fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, max_value=1000.0, value=770.0)
    age = st.number_input("Age (days)", min_value=1, max_value=365, value=28)

st.divider()

if st.button("Predict Strength", type="primary"):
    # Build input in the same column order used during training
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"### Predicted Compressive Strength: {prediction:.2f} MPa")