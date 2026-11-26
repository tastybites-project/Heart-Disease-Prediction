
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# STEP 1: LOAD FEATURES AND TARGET
X = pd.read_csv("dataset/X_features.csv")
y = pd.read_csv("dataset/y_target.csv").squeeze("columns")

# STEP 2: SPLIT DATASET (80% TRAIN, 20% TEST)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 3: FEATURE SCALING
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 4: CONVERT SCALED DATA INTO DATAFRAMES
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

# STEP 5: CREATE MODELS FOLDER
os.makedirs("models", exist_ok=True)

# STEP 6: SAVE TRAINING AND TESTING DATA
X_train_scaled.to_csv("dataset/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("dataset/X_test_scaled.csv", index=False)

y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

# STEP 7: SAVE SCALER
joblib.dump(scaler, "models/scaler.pkl")

# STEP 8: DISPLAY RESULTS
print("========== TRAIN TEST SPLIT ==========")

print("\nOriginal Dataset:", X.shape)
print("Training Features:", X_train_scaled.shape)
print("Testing Features:", X_test_scaled.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting Target Distribution:")
print(y_test.value_counts().sort_index())

print("\nTrain-Test Split and Feature Scaling Completed!")

print("\nSaved Files:")
print("dataset/X_train_scaled.csv")
print("dataset/X_test_scaled.csv")
print("dataset/y_train.csv")
print("dataset/y_test.csv")
print("models/scaler.pkl")
