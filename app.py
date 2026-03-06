import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("weather_model.pkl")

st.title("🌦️ AI Prediksi Cuaca")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind_speed = st.number_input("Wind Speed")

if st.button("Prediksi Cuaca"):

    data = np.array([[temperature, humidity, wind_speed]])

    prediction = model.predict(data)

    st.success(f"Hasil Prediksi: {prediction}")