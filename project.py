# project.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Create synthetic dataset
np.random.seed(42)
n_samples = 10000

data = {
    "transaction_id": np.arange(n_samples),
    "user_id": np.random.randint(1000, 5000, size=n_samples),
    "amount": np.random.exponential(scale=2000, size=n_samples).round(2),
    "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="min"),  # Updated 'T' to 'min'
    "merchant_id": np.random.choice(["M1", "M2", "M3", "M4"], size=n_samples),
    "location": np.random.choice(["Delhi", "Mumbai", "Bangalore", "Kolkata"], size=n_samples),
    "device_type": np.random.choice(["Android", "iOS", "Web"], size=n_samples),
    "is_fraud": np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
}

df = pd.DataFrame(data)

# Step 2: Preprocessing
label_encoders = {}
for col in ['merchant_id', 'location', 'device_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df['hour'] = df['timestamp'].dt.hour
df.drop(['transaction_id', 'user_id', 'timestamp'], axis=1, inplace=True)

# Step 3: Feature selection
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 7: Prediction and Evaluation
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Step 8: Output
print("✅ Model trained successfully!")
print("\n📊 Confusion Matrix:")
print(conf_matrix)
print("\n📈 Classification Report:")
print(report)




