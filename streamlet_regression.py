import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ==============================
# Load Model & Encoders/Scalers
# ==============================
MODEL_PATH = "salary_regression_model.h5"
ENCODERS_DIR = "."

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(f"{ENCODERS_DIR}/gender_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

with open(f"{ENCODERS_DIR}/geo_encoder.pkl", "rb") as f: 
    geo_encoder = pickle.load(f)

with open(f"{ENCODERS_DIR}/scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open(f"{ENCODERS_DIR}/scaler_Y.pkl", "rb") as f:
    scaler_Y = pickle.load(f)

# ==============================
# Streamlit App UI
# ==============================
st.set_page_config(page_title="Salary Prediction", page_icon="💰", layout="centered")
st.title("💰 Estimated Salary Prediction App")
st.markdown("Enter customer details to get an estimated salary prediction.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("🌍 Geography", geo_encoder.categories_[0])
    gender = st.selectbox("👤 Gender", gender_encoder.classes_)
    age = st.slider("📅 Age", 18, 92, 30)
    tenure = st.slider("📆 Tenure (Years)", 0, 10, 5)
    num_of_products = st.slider("🛍 Number of Products", 1, 4, 1)

with col2:
    balance = st.number_input("🏦 Balance", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=850, value=600, step=1)
    has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
    is_active_member = st.selectbox("✅ Is Active Member", [0, 1])
    exited = st.selectbox("🚪 Exited", [0, 1])

# ==============================
# Data Preprocessing
# ==============================
# Encode gender
gender_encoded = gender_encoder.transform([gender])[0]

# One-hot encode geography
geo_encoded = geo_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=geo_encoder.get_feature_names_out(["Geography"])
)

# Combine into DataFrame
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]
})

# Merge with geography one-hot columns
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# Ensure same feature order as training
try:
    input_data = input_data[scaler_X.feature_names_in_]
except AttributeError:
    st.error("Scaler does not contain feature names. Please check training code.")
    st.stop()

# Scale inputs
scaled_input = scaler_X.transform(input_data)

# ==============================
# Prediction
# ==============================
if st.button("🔮 Predict Salary"):
    pred_scaled = model.predict(scaled_input)
    pred_actual = scaler_Y.inverse_transform(pred_scaled)
    salary_value = pred_actual[0][0]

    st.success(f"💵 Predicted Estimated Salary: ₹{salary_value:,.2f}")
