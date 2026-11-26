
import os
import pandas as pd
import joblib

print("========== FINAL PROJECT TESTING ==========")

# STEP 1: CHECK IMPORTANT FILES
required_files = [
    "dataset/heart_disease.csv",
    "dataset/heart_disease_cleaned.csv",
    "dataset/heart_disease_outlier_handled.csv",
    "dataset/heart_disease_encoded.csv",
    "dataset/X_features.csv",
    "dataset/y_target.csv",
    "dataset/X_train_scaled.csv",
    "dataset/X_test_scaled.csv",
    "dataset/y_train.csv",
    "dataset/y_test.csv",
    "models/scaler.pkl",
    "models/linear_regression_model.pkl",
    "models/logistic_regression_model.pkl",
    "models/knn_model.pkl",
    "models/kmeans_model.pkl",
    "models/best_heart_disease_model.pkl",
    "reports/model_optimization_results.csv",
    "reports/final_model_comparison.csv",
    "reports/best_model_summary.csv",
    "app.py"
]

missing_files = []

print("\n--- FILE CHECK ---")

for file in required_files:
    if os.path.exists(file):
        print("[OK]", file)
    else:
        print("[MISSING]", file)
        missing_files.append(file)

# STEP 2: CHECK DATASET SHAPES
print("\n--- DATASET CHECK ---")

X = pd.read_csv("dataset/X_features.csv")
y = pd.read_csv("dataset/y_target.csv").squeeze("columns")

X_train = pd.read_csv("dataset/X_train_scaled.csv")
X_test = pd.read_csv("dataset/X_test_scaled.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze("columns")
y_test = pd.read_csv("dataset/y_test.csv").squeeze("columns")

print("Feature Dataset:", X.shape)
print("Target Dataset:", y.shape)
print("Training Dataset:", X_train.shape)
print("Testing Dataset:", X_test.shape)

print("Total Split Rows:", len(X_train) + len(X_test))

# STEP 3: CHECK TRAIN-TEST SPLIT
assert len(X_train) == len(y_train), "Training rows mismatch"
assert len(X_test) == len(y_test), "Testing rows mismatch"
assert len(X_train) + len(X_test) == len(X), "Total rows mismatch"

print("\nTrain-Test Split: PASSED")

# STEP 4: LOAD SCALER AND BEST MODEL
scaler = joblib.load("models/scaler.pkl")
best_model = joblib.load("models/best_heart_disease_model.pkl")

# STEP 5: TEST MODEL PREDICTION
sample_patient = X.iloc[[0]]
sample_scaled = scaler.transform(sample_patient)

prediction = best_model.predict(sample_scaled)[0]

print("\n--- MODEL PREDICTION TEST ---")
print("Best Model:", type(best_model).__name__)
print("Sample Prediction:", prediction)

assert prediction in [0, 1], "Unexpected prediction value"

print("Model Prediction Test: PASSED")

# STEP 6: CHECK FEATURE COUNT
assert scaler.n_features_in_ == X.shape[1], "Scaler feature mismatch"

print("\nFeature Scaling Test: PASSED")

# STEP 7: FINAL TEST SUMMARY
print("\n========== FINAL TEST SUMMARY ==========")

if missing_files:
    print("Some required files are missing:")
    for file in missing_files:
        print("-", file)
    print("\nPlease check the missing files.")
else:
    print("All required files are available.")

print("Dataset and model checks completed.")
