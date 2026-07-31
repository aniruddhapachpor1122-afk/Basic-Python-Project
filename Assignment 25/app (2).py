import streamlit as st
import joblib
import numpy as np

model=joblib.load("best_model.pkl")

st.title("Iris Flower Prediction")

sl=st.number_input("Sepal Length")
sw=st.number_input("Sepal Width")
pl=st.number_input("Petal Length")
pw=st.number_input("Petal Width")

if st.button("Predict"):

    data=np.array([[sl,sw,pl,pw]])

    pred=model.predict(data)

    st.success(f"Prediction : {pred[0]}")