import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler (3).pkl")
columns = joblib.load("columns (1).pkl")

st.title("❤️ Heart Disease Prediction")

age = st.number_input("Age",1,100)
bp = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
fast = st.selectbox("Fasting Blood Sugar",[0,1])
hr = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Old Peak",format="%.1f")

if st.button("Predict"):

    data = pd.DataFrame({
        "Age":[age],
        "RestingBP":[bp],
        "Cholesterol":[chol],
        "FastingBS":[fast],
        "MaxHR":[hr],
        "Oldpeak":[oldpeak]
    })

    data = pd.get_dummies(data)
    data = data.reindex(columns=columns, fill_value=0)
    data = scaler.transform(data)

    pred = model.predict(data)

    if pred[0]==1:
        st.success("❤️ Heart Disease : YES")
    else:
        st.error("💚 Heart Disease : NO")