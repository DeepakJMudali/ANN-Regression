import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained Keras model
model = tf.keras.models.load_model("salary_regression_model.keras", compile=False)

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f: 
    geo_encoder = pickle.load(f)

with open("scaler_X.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit title
st.title("Estimated Salary Prediction App")

# User inputs
geography = st.selectbox("Geography", geo_encoder.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
exited = st.selectbox("Exited", [0, 1])
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Encode gender
gender_encoded = gender_encoder.transform([gender])[0]

# One-hot encode geography
geo_encoded = geo_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=geo_encoder.get_feature_names_out(["Geography"])
)

# Create the input DataFrame (with all features EXCEPT EstimatedSalary)
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]  # ✅ required input feature
})

# Add one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# Match training column order
input_data = input_data[scaler.feature_names_in_]

# Scale the input features
scaled_input = scaler.transform(input_data)

# Predict salary
predicted_salary = model.predict(scaled_input)[0][0]

# Display result
st.success(f"Predicted Estimated Salary: ₹{predicted_salary:,.2f}")
