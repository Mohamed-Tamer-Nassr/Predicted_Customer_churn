import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model ,scaler, ohe,and label encoder
model = tf.keras.models.load_model("ann_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("onehot_encoder_geography.pkl", "rb") as f:
    ohe = pickle.load(f)


st.title("Customer Churn Prediction")

# user inputs
geography = st.selectbox("Geography", ohe.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (years)", 0, 10, 2)
balance = st.number_input(
    "Balance ($)", min_value=0.0, max_value=500000.0, value=50000.0
)
num_of_products = st.slider("Number of Products", 1, 4, 2)
is_active_member = st.selectbox("Is Active Member?", [1, 0])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

estimated_salary = st.number_input(
    "Estimated Salary ($)", min_value=1000.0, max_value=200000.0, value=50000.0
)
has_cr_card = st.selectbox("Has Credit Card?", [1, 0])


# Encode gender
gender_encoded = label_encoder.transform([gender])[0]

# Encode geography
geography_encoded = ohe.transform([[geography]])
if hasattr(geography_encoded, "toarray"):
    geography_encoded = geography_encoded.toarray()
geography_df = pd.DataFrame(
    geography_encoded, columns=ohe.get_feature_names_out(["Geography"])
)

# prepare the input data - matching the EXACT order used in training
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

# Combine with geography one-hot encoded columns
input_data = pd.concat([input_data, geography_df], axis=1)


# scale the input data
input_data_scaled = scaler.transform(input_data)


# make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    if churn_probability > 0.5:
        st.error(
            f"The customer is likely to churn with a probability of {churn_probability:.2f}"
        )
    else:
        st.success(
            f"The customer is unlikely to churn with a probability of {churn_probability:.2f}"
        )
