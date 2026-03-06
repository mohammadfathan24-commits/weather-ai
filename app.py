import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("weather_model.pkl")

# load encoder
encoder = joblib.load("label_encoder.pkl")

st.title("🌦️ AI Prediksi Cuaca")

st.write("Masukkan data cuaca")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind_speed = st.number_input("Wind Speed")
precipitation = st.number_input("Precipitation (%)")
pressure = st.number_input("Atmospheric Pressure")
uv_index = st.number_input("UV Index")
visibility = st.number_input("Visibility (km)")

if st.button("Prediksi Cuaca"):

    data = np.array([[temperature,
                      humidity,
                      wind_speed,
                      precipitation,
                      pressure,
                      uv_index,
                      visibility]])

    prediction = model.predict(data)

    hasil = encoder.inverse_transform(prediction)

    st.success(f"Hasil Prediksi Cuaca: {hasil[0]}")