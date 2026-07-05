import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc

#Input Parameters
experiment_id = "14"
model_id = "m-49fa9768440d4b399a5821a30f3d2d0e"

#Page Configuration 
st.set_page_config(
    page_title="S25021301 | COM763 | Customer Churn Prediction",
)

st.title(":orange[Customer Churn Prediction]")
st.markdown("""
This Streamlit Application has been created for the Assignment Task - 1 of module COM736 - Advance Machine Learning\n
Student Id: S25021301
""")

#Load the Model from MLflow
@st.cache_resource
def load_mlflow_model(experiment_id: str, model_id: str):
    model_path = f"mlruns/mlartifacts/{experiment_id}/models/{model_id}/artifacts/"
    return mlflow.pyfunc.load_model(model_path)

try:
    model = load_mlflow_model(experiment_id, model_id)
except Exception as e:
    st.error(f"Model loading failed with error: {e}")
    st.stop()

# Input Form 
st.subheader("Enter the Customer Primary Details:")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col2:
    partner = st.selectbox("Has a Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

st.subheader("Services")
col3, col4 = st.columns(2)

with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
with col4:    
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.subheader("Financial Information")
col5, col6 = st.columns(2)

with col5:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
with col6:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=30.0, step=0.1)

binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

input_data = {
    'gender': binary_mapping[gender],
    'SeniorCitizen': int(senior_citizen),
    'Partner': binary_mapping[partner],
    'Dependents': binary_mapping[dependents],
    'tenure': int(tenure),
    'PhoneService': binary_mapping[phone_service],
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': binary_mapping[paperless_billing],
    'PaymentMethod': payment_method,
    'MonthlyCharges': float(monthly_charges),
    'TotalCharges': float(total_charges)
}

df = pd.DataFrame([input_data])

st.markdown("---")

if st.button("Predict Customer Churn", type="primary"):
    prediction = model.predict(df)[0]
    try:
        prediction_class = int(round(prediction)) if isinstance(prediction, (int, float, np.number)) else int(prediction)
    except:
        prediction_class = int(prediction)

    st.subheader("Result:")
    if prediction_class == 1:
        st.error(f"Customer is likely to Churn")
    else:
        st.success(f"Customer is un-likely to Churn")