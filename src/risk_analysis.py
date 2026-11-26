import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


print("========== PATIENT RISK ANALYSIS ==========")


# Load dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")


# Features and Target
X = df.drop("target", axis=1)
y = df["target"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Logistic Regression Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# =====================================================
# PATIENT INPUT
# =====================================================

print("\nEnter Patient Information")
print("--------------------------------")

age = float(input("Age: "))

sex = int(input("Sex (0 = Female, 1 = Male): "))

cp = int(input("Chest Pain Type (0-3): "))

trestbps = float(input("Resting Blood Pressure: "))

chol = float(input("Cholesterol: "))

fbs = int(input("Fasting Blood Sugar (0 = No, 1 = Yes): "))

restecg = int(input("Resting ECG Result (0-2): "))

thalach = float(input("Maximum Heart Rate: "))

exang = int(input(
    "Exercise Induced Angina (0 = No, 1 = Yes): "
))

oldpeak = float(input("ST Depression (Oldpeak): "))

slope = int(input("Slope (0-2): "))

ca = int(input("Number of Major Vessels (0-4): "))

thal = int(input("Thalassemia (0-3): "))


# =====================================================
# CREATE PATIENT DATA
# =====================================================

patient = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])


# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(patient)

probability = model.predict_proba(patient)

risk_score = probability[0][1] * 100


# =====================================================
# RISK CATEGORY
# =====================================================

if risk_score < 30:
    risk_category = "Low"

elif risk_score < 70:
    risk_category = "Medium"

else:
    risk_category = "High"


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n========================================")
print("       PATIENT RISK ANALYSIS")
print("========================================")

if prediction[0] == 1:
    print("Heart Disease Prediction: YES")
else:
    print("Heart Disease Prediction: NO")

print(f"Patient Risk Score: {risk_score:.2f}%")

print(f"Patient Risk Category: {risk_category}")

print("========================================")

print("\nRisk analysis completed successfully!")