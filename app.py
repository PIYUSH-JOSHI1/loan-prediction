# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_processing import load_and_preprocess_data
from src.model import train_model

# Set page config
st.set_page_config(page_title="Loan Prediction App", page_icon="💰", layout="wide")

# Load data and train model
@st.cache(allow_output_mutation=True)
def load_data_and_train_model():
    data_path = os.path.join(os.path.dirname(__file__), "data", "trainloan.csv")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(data_path)
    model = train_model(X_train, y_train)
    return X_train, X_test, y_train, y_test, scaler, feature_names, model

X_train, X_test, y_train, y_test, scaler, feature_names, model = load_data_and_train_model()

st.title("Loan Prediction App 💰")
st.write("Enter your information to predict loan approval probability.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12, max_value=480, value=360)
    credit_history = st.selectbox("Credit History", ["1", "0"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create a dictionary with user inputs
user_input = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Preprocess user input
user_df_encoded = pd.get_dummies(user_df, drop_first=True)

# Ensure all columns from training data are present in user input
for col in feature_names:
    if col not in user_df_encoded.columns:
        user_df_encoded[col] = 0

# Reorder columns to match training data
user_df_encoded = user_df_encoded[feature_names]

# Scale user input
user_df_scaled = scaler.transform(user_df_encoded)

# Make prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(user_df_scaled)
    probability = model.predict_proba(user_df_scaled)[0][1]
    
    st.write("---")
    if prediction[0] == 1:
        st.success(f"Congratulations! Your loan is likely to be approved with a probability of {probability:.2%}")
    else:
        st.error(f"Sorry, your loan is likely to be rejected with a probability of {1-probability:.2%}")
    
    # Display feature importances
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False).head(10)
    
    st.write("---")
    st.subheader("Top 10 Features Influencing the Decision")
    st.bar_chart(feature_imp.set_index('feature'))

# Add some information about the model
st.sidebar.title("About")
st.sidebar.info("This app uses a Random Forest Classifier to predict loan approval based on the provided information. The model is trained on a dataset of previous loan applications.")

# Add model performance metrics
st.sidebar.title("Model Performance")
accuracy = model.score(X_test, y_test)
st.sidebar.metric("Accuracy", f"{accuracy:.2%}")

# Add a disclaimer
st.sidebar.title("Disclaimer")
st.sidebar.warning("This app is for educational purposes only and should not be used as financial advice. Please consult with a professional for actual loan decisions.")