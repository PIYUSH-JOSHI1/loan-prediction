import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Set page config
st.set_page_config(page_title="Loan Prediction App", page_icon="💰", layout="wide")

# Define color schemes
light_theme = {
    "bg_color": "#f5f7f9",
    "text_color": "#2c3e50",
    "accent_color": "#3498db",
    "success_color": "#2ecc71",
    "error_color": "#e74c3c"
}

dark_theme = {
    "bg_color": "#2c3e50",
    "text_color": "#ecf0f1",
    "accent_color": "#3498db",
    "success_color": "#2ecc71",
    "error_color": "#e74c3c"
}

# Function to get the current theme
def get_current_theme():
    return dark_theme if st.session_state.dark_mode else light_theme

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Toggle dark mode
if st.sidebar.checkbox("Dark Mode", value=st.session_state.dark_mode):
    st.session_state.dark_mode = True
else:
    st.session_state.dark_mode = False

# Get current theme
current_theme = get_current_theme()

# Custom CSS
st.markdown(f"""
<style>
    .main {{
        background-color: {current_theme["bg_color"]};
        color: {current_theme["text_color"]};
    }}
    .stApp {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    .st-bw {{
        background-color: {current_theme["bg_color"]};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .st-eb {{
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }}
    .stButton>button {{
        background-color: {current_theme["accent_color"]};
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        opacity: 0.8;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}
    h1, h2, h3 {{
        color: {current_theme["text_color"]};
    }}
    .stSidebar {{
        background-color: {current_theme["bg_color"]};
        padding: 20px;
    }}
    .sidebar-content {{
        background-color: {current_theme["bg_color"]};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache(allow_output_mutation=True)
def load_and_preprocess_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "trainloan.csv")
    df = pd.read_csv(data_path)
    df = df.drop(['Loan_ID'], axis=1)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('Loan_Status_Y', axis=1)
    y = df['Loan_Status_Y']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# Train the model
@st.cache(allow_output_mutation=True)
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load data and train model
X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
model = train_model(X_train, y_train)

st.title("🏦 Loan Prediction App")
st.markdown(f"<p style='text-align: center; font-size: 1.2em; color: {current_theme['text_color']};'>Enter your information to predict loan approval probability.</p>", unsafe_allow_html=True)

st.markdown("<div class='st-bw'>", unsafe_allow_html=True)
# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2>Personal Information</h2>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.markdown("<h2>Financial Information</h2>", unsafe_allow_html=True)
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12, max_value=480, value=360)
    credit_history = st.selectbox("Credit History", ["1", "0"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("</div>", unsafe_allow_html=True)

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
    
    st.markdown("<div class='st-bw'>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.markdown(f"<h2 style='color: {current_theme['success_color']};'>🎉 Congratulations!</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.2em;'>Your loan is likely to be approved with a probability of <strong>{probability:.2%}</strong></p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color: {current_theme['error_color']};'>😔 We're sorry</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.2em;'>Your loan is likely to be rejected with a probability of <strong>{1-probability:.2%}</strong></p>", unsafe_allow_html=True)
    
    # Display feature importances
    st.markdown("<h2>Top 10 Features Influencing the Decision</h2>", unsafe_allow_html=True)
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False).head(10)
    
    st.bar_chart(feature_imp.set_index('feature'))
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
st.sidebar.title("About")
st.sidebar.info("This app uses a Random Forest Classifier to predict loan approval based on the provided information. The model is trained on a dataset of previous loan applications.")

# Add model performance metrics
st.sidebar.title("Model Performance")
accuracy = model.score(X_test, y_test)
st.sidebar.metric("Accuracy", f"{accuracy:.2%}")

# Add a disclaimer
st.sidebar.title("Disclaimer")
st.sidebar.warning("This app is for educational purposes only and should not be used as financial advice. Please consult with a professional for actual loan decisions.")
st.sidebar.markdown("</div>", unsafe_allow_html=True)
