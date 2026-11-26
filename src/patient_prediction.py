
import pandas as pd
import joblib
import os

# STEP 1: LOAD TRAINED MODEL AND SCALER
model = joblib.load("models/best_heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# STEP 2: LOAD ORIGINAL FEATURE NAMES
X = pd.read_csv("dataset/X_features.csv")
feature_names = X.columns.tolist()

print("========== HEART DISEASE PREDICTION SYSTEM ==========")

print("\nEnter Patient Information:")

# STEP 3: TAKE PATIENT INPUT
patient_data = {}

for feature in feature_names:
    while True:
        try:
            value = float(input(f"Enter {feature}: "))
            patient_data[feature] = value
            break
        except ValueError:
            print("Please enter a valid numeric value.")

# STEP 4: CONVERT INPUT INTO DATAFRAME
patient_df = pd.DataFrame(
    [patient_data],
    columns=feature_names
)

# STEP 5: SCALE PATIENT DATA
patient_scaled = scaler.transform(patient_df)

# STEP 6: PREDICT DISEASE STATUS
prediction = model.predict(patient_scaled)[0]

# STEP 7: CALCULATE PROBABILITY
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(patient_scaled)[0][1]
else:
    probability = None

# STEP 8: DISPLAY RESULTS
print("\n========== PREDICTION RESULT ==========")

if prediction == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease Detected")

if probability is not None:
    print("Estimated Model Probability: {:.2f}%".format(
        probability * 100
    ))

print("\nModel Used:", type(model).__name__)

print("\nNote:")
print("This is an educational machine learning prediction.")
print("It is not a medical diagnosis. Consult a qualified doctor.")

