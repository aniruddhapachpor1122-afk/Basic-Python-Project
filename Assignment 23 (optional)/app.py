import streamlit as st
import pandas as pd
import joblib

#  LOAD FILES 
linear_model = joblib.load("linear_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")
knn_model = joblib.load("knn_model.pkl")
nb_model = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")   

#  UI 
st.title(" Cloth Prediction App")

#  MODEL SELECT 
model_choice = st.selectbox(
    "Select Model",
    ["Linear Regression", "Logistic Regression", "KNN", "Naive Bayes"]
)
st.subheader("Enter Details")

#  INPUT FIELDS 
age = st.number_input("Age", 0, 100, 25)
income = st.number_input("Income", 0, 100000, 5000)
price = st.number_input("Cloth Price", 0, 5000, 500)
rating = st.number_input("Rating", 0.0, 5.0, 3.0)
discount = st.number_input("Discount (%)", 0, 100, 10)
purchase_freq = st.number_input("Purchase Frequency", 0, 50, 5)
stock = st.number_input("Stock Available", 0, 1000, 100)

#  CREATE INPUT 
input_dict = {col: 0 for col in columns}   

# fill your values
if "Age" in input_dict: input_dict["Age"] = age
if "Income" in input_dict: input_dict["Income"] = income
if "Price" in input_dict: input_dict["Price"] = price
if "Rating" in input_dict: input_dict["Rating"] = rating
if "Discount" in input_dict: input_dict["Discount"] = discount
if "Purchase_Frequency" in input_dict: input_dict["Purchase_Frequency"] = purchase_freq
if "Stock" in input_dict: input_dict["Stock"] = stock

#  DATAFRAME 
input_data = pd.DataFrame([input_dict])

#  SCALE 
input_scaled = scaler.transform(input_data)

#  PREDICT 
if st.button(" Predict"):

    try:
        if model_choice == "Linear Regression":
            prediction = linear_model.predict(input_scaled)
            st.success(f" Prediction: {prediction[0]:.2f}")

        elif model_choice == "Logistic Regression":
            prediction = logistic_model.predict(input_scaled)
            st.success(f" Prediction: {prediction[0]}")

        elif model_choice == "KNN":
            prediction = knn_model.predict(input_scaled)
            st.success(f" Prediction: {prediction[0]}")

        elif model_choice == "Naive Bayes":
            prediction = nb_model.predict(input_scaled)
            st.success(f" Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f" Error: {e}")