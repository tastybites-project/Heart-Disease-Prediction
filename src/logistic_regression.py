
import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# STEP 1: LOAD TRAINING AND TESTING DATA
X_train = pd.read_csv("dataset/X_train_scaled.csv")
X_test = pd.read_csv("dataset/X_test_scaled.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze("columns")
y_test = pd.read_csv("dataset/y_test.csv").squeeze("columns")

# STEP 2: CREATE LOGISTIC REGRESSION MODEL
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# STEP 3: TRAIN MODEL
model.fit(X_train, y_train)

# STEP 4: PREDICT DISEASE STATUS
y_pred = model.predict(X_test)

# STEP 5: PREDICT PROBABILITY
probabilities = model.predict_proba(X_test)[:, 1]

# STEP 6: EVALUATE MODEL
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("========== LOGISTIC REGRESSION RESULTS ==========")

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# STEP 7: CREATE PREDICTION RESULTS
results = pd.DataFrame({
    "Actual Disease Status": y_test.values,
    "Predicted Disease Status": y_pred,
    "Disease Probability (%)": probabilities * 100
})

print("\nFirst 10 Patient Predictions:")
print(results.head(10).round(2))

# STEP 8: SAVE MODEL AND RESULTS
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/logistic_regression_model.pkl")

results.to_csv(
    "reports/logistic_regression_predictions.csv",
    index=False
)

# STEP 9: SAVE EVALUATION METRICS
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
})

metrics.to_csv(
    "reports/logistic_regression_metrics.csv",
    index=False
)

print("\nLogistic Regression Model Training Completed!")
print("Model saved: models/logistic_regression_model.pkl")
print("Predictions saved: reports/logistic_regression_predictions.csv")
print("Metrics saved: reports/logistic_regression_metrics.csv")
