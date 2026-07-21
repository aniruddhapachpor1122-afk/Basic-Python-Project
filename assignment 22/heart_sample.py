import streamlit as st

st.title("❤️ Heart Disease Prediction")

age = st.number_input("Age", 1, 100)
bp = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
fast = st.selectbox("Fasting Blood Sugar", [0,1])
hr = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Old Peak", format="%.1f")

if st.button("Predict"):
    st.write("Prediction Button Clicked")