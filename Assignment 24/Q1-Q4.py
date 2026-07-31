# ============================================
# Q1. Dataset Selection & Preprocessing
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Classification Dataset
# -----------------------------

# Load Heart Disease Dataset
heart = pd.read_csv("heart .csv")

print("Heart Dataset")
print(heart.head())

# Check Missing Values
print("\nMissing Values")
print(heart.isnull().sum())
print("Data Cleaning Successfully Completed")

# Independent Features (X)
X_cls = heart.drop("HeartDisease", axis=1)

# Dependent Feature (y)
y_cls = heart["HeartDisease"]

# One-Hot Encoding
X_cls = pd.get_dummies(X_cls, drop_first=True)
print("Encoding Successfully Completed")

# Train-Test Split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.20,
    random_state=42
)
print("Train-Test Split Successfully Completed")

# Feature Scaling
scaler_cls = StandardScaler()

X_train_cls = scaler_cls.fit_transform(X_train_cls)
X_test_cls = scaler_cls.transform(X_test_cls)
print("Feature Scaling Successfully Completed")

print("\nClassification Dataset Ready")
print("X_train Shape:", X_train_cls.shape)
print("X_test Shape :", X_test_cls.shape)
print("y_train Shape:", y_train_cls.shape)
print("y_test Shape :", y_test_cls.shape)

# ------------------------------------
# Regression Dataset
# ------------------------------------

# Load Car Dataset
car = pd.read_csv("Car.csv")

print("\nCar Dataset")
print(car.head())

# Check Missing Values
print("\nMissing Values")
print(car.isnull().sum())
print("Data Cleaning Successfully Completed")

# Independent Features
X_reg = car.drop("price", axis=1)

# Dependent Feature
y_reg = car["price"]

# One-Hot Encoding
X_reg = pd.get_dummies(X_reg, drop_first=True)
print("Encoding Successfully Completed")

# Train-Test Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)
print("Train-Test Split Successfully Completed")

# Feature Scaling
scaler_reg = StandardScaler()

X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)
print("Feature Scaling Successfully Completed")

print("\nRegression Dataset Ready")
print("X_train Shape:", X_train_reg.shape)
print("X_test Shape :", X_test_reg.shape)
print("y_train Shape:", y_train_reg.shape)
print("y_test Shape :", y_test_reg.shape)

# ============================================
# Q2. Classification Algorithms
# ============================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Dictionary of Classification Models

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Support Vector Machine": SVC(),

    "K-Nearest Neighbors": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB()

}

accuracy_list = []

best_accuracy = 0
best_model = None
best_model_name = ""

print("="*60)
print("CLASSIFICATION MODELS")
print("="*60)

for name, model in models.items():

    print("\n")

    print("="*60)
    print(name)
    print("="*60)

    # Train Model
    model.fit(X_train_cls, y_train_cls)
    print("Model Trained Successfully")

    # Prediction
    y_pred = model.predict(X_test_cls)
    print("Prediction Completed Successfully")

    # Accuracy
    accuracy = accuracy_score(y_test_cls, y_pred)

    accuracy_list.append([name, accuracy])

    print("Accuracy :", accuracy)

    # Confusion Matrix

    cm = confusion_matrix(y_test_cls, y_pred)

    print("\nConfusion Matrix")

    print(cm)

    # Classification Report

    print("\nClassification Report")

    print(classification_report(y_test_cls, y_pred))

    # Best Model

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

print("\n")

print("="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame(

    accuracy_list,

    columns=["Model", "Accuracy"]

)

print(comparison)

print("\nBest Classification Model :", best_model_name)

print("Best Accuracy :", best_accuracy)

# ============================================
# Q3. Regression Algorithms
# ============================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
import pandas as pd

# Dictionary of Regression Models

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),

    "Support Vector Regressor": SVR(),

    "K-Nearest Neighbors Regressor": KNeighborsRegressor()

}

r2_list = []

best_r2 = -999
best_regression_model = None
best_model_name = ""

print("="*60)
print("REGRESSION MODELS")
print("="*60)

for name, model in models.items():

    print("\n")
    print("="*60)
    print(name)
    print("="*60)

    # Train Model
    model.fit(X_train_reg, y_train_reg)
    print("Model Trained Successfully")

    # Prediction
    y_pred = model.predict(X_test_reg)
    print("Prediction Completed Successfully")

    # R² Score
    r2 = r2_score(y_test_reg, y_pred)

    r2_list.append([name, r2])

    print("R² Score :", r2)

    # Best Model
    if r2 > best_r2:

        best_r2 = r2
        best_regression_model = model
        best_model_name = name

print("\n")

print("="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame(

    r2_list,

    columns=["Model", "R² Score"]

)

print(comparison)

print("\nBest Regression Model :", best_model_name)
print("Best R² Score :", best_r2)


# ============================================
# Q4. Best Model Selection & Saving
# ============================================

import joblib

# Save Best Classification Model
joblib.dump(best_model, "best_classification_model.joblib")
joblib.dump(scaler_cls, "classification_scaler.joblib")

print("Best Classification Model Saved Successfully!")

# Save Best Regression Model
joblib.dump(best_regression_model, "best_regression_model.joblib")
joblib.dump(scaler_reg, "regression_scaler.joblib")

print("Best Regression Model Saved Successfully!")

print("\nAll Models Saved Successfully.")