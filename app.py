
import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("Breast Cancer Prediction System")
st.write("⚠️ Educational purposes only. Not a medical diagnostic tool.")

st.subheader("Enter Tumor Feature Values")

radius = st.number_input("Mean Radius", min_value=0.0, step=0.1)
texture = st.number_input("Mean Texture", min_value=0.0, step=0.1)
perimeter = st.number_input("Mean Perimeter", min_value=0.0, step=0.1)
area = st.number_input("Mean Area", min_value=0.0, step=1.0)
smoothness = st.number_input("Mean Smoothness", min_value=0.0, step=0.001)

if st.button("Predict"):
    input_data = np.array([[radius, texture, perimeter, area, smoothness]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")
