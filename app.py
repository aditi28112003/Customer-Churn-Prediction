import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction")

model = joblib.load("model.pkl")
input_data = {
    'tenure': st.number_input("Tenure", min_value=0),
    'MonthlyCharges': st.number_input("Monthly Charges"),
    'TotalCharges': st.number_input("Total Charges"),
    # Add more inputs as per features used
}

if st.button("Predict"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    st.write("Churn" if prediction else "Not Churn")
