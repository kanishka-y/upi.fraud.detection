import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --------------------
# Step 1: Create synthetic data (same as your training project)
np.random.seed(42)
n_samples = 10000

data = {
    "transaction_id": np.arange(n_samples),
    "user_id": np.random.randint(1000, 5000, size=n_samples),
    "amount": np.random.exponential(scale=2000, size=n_samples).round(2),
    "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="min"),
    "merchant_id": np.random.choice(["M1", "M2", "M3", "M4"], size=n_samples),
    "location": np.random.choice(["Delhi", "Mumbai", "Bangalore", "Kolkata"], size=n_samples),
    "device_type": np.random.choice(["Android", "iOS", "Web"], size=n_samples),
    "is_fraud": np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
}

df = pd.DataFrame(data)

# Encode categorical columns
label_encoders = {}
for col in ['merchant_id', 'location', 'device_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature engineering
df['hour'] = df['timestamp'].dt.hour
df.drop(['transaction_id', 'user_id', 'timestamp'], axis=1, inplace=True)

# Split features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# --------------------
# Step 2: Streamlit UI
st.title("💳 UPI Fraud Detection System")

st.write("""
This is a simple AI model that predicts whether a UPI transaction is **fraudulent** or **legitimate** based on user inputs.
""")

st.header("🔎 Enter Transaction Details:")

amount = st.number_input("Transaction Amount (INR)", min_value=1.0, max_value=100000.0, value=1000.0, step=1.0)

merchant = st.selectbox("Merchant", label_encoders['merchant_id'].classes_)
location = st.selectbox("Location", label_encoders['location'].classes_)
device = st.selectbox("Device Type", label_encoders['device_type'].classes_)
hour = st.slider("Hour of Transaction (0-23)", 0, 23, 12)

if st.button("Predict Fraud"):
    # Encode inputs
    merchant_encoded = label_encoders['merchant_id'].transform([merchant])[0]
    location_encoded = label_encoders['location'].transform([location])[0]
    device_encoded = label_encoders['device_type'].transform([device])[0]

    # Create feature array
    input_data = np.array([[amount, merchant_encoded, location_encoded, device_encoded, hour]])
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.error("🚨 Alert! This transaction is likely **FRAUDULENT**.")
    else:
        st.success("✅ This transaction is likely **LEGITIMATE**.")
