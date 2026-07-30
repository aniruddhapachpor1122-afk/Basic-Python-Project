# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression   # safer default

# Load dataset
data = pd.read_csv("Colth.csv")

print("✅ Dataset Loaded Successfully")
print(data.head())

# 🔴 IMPORTANT: Check columns
print("\nColumns are:", data.columns)

# 👉 CHANGE THIS BASED ON YOUR DATASET
target_column = data.columns[-1]   # automatically takes last column

# Separate features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Handle categorical columns properly
X = pd.get_dummies(X, drop_first=True)

# Save columns
joblib.dump(X.columns.tolist(), "columns.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train model (Linear Regression for safety)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("\n🎉 SUCCESS!")
print("Files created:")
print("✔ model.pkl")
print("✔ scaler.pkl")
print("✔ columns.pkl")