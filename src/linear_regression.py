
import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# STEP 1: LOAD TRAINING AND TESTING DATA
X_train = pd.read_csv("dataset/X_train_scaled.csv")
X_test = pd.read_csv("dataset/X_test_scaled.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze("columns")
y_test = pd.read_csv("dataset/y_test.csv").squeeze("columns")

# STEP 2: CREATE LINEAR REGRESSION MODEL
model = LinearRegression()

# STEP 3: TRAIN MODEL
model.fit(X_train, y_train)

# STEP 4: MAKE PREDICTIONS
y_pred = model.predict(X_test)

# STEP 5: CONVERT OUTPUT INTO 0-100 RISK SCORE
risk_scores = y_pred.clip(0, 1) * 100

# STEP 6: EVALUATE MODEL
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("========== LINEAR REGRESSION RESULTS ==========")

print("\nMean Absolute Error:", round(mae, 4))
print("Mean Squared Error:", round(mse, 4))
print("R2 Score:", round(r2, 4))

# STEP 7: DISPLAY PATIENT RISK SCORES
results = pd.DataFrame({
    "Actual Disease Status": y_test.values,
    "Predicted Value": y_pred,
    "Estimated Risk Score": risk_scores
})

print("\nFirst 10 Patient Risk Scores:")
print(results.head(10).round(2))

# STEP 8: SAVE RESULTS
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/linear_regression_model.pkl")

results.to_csv(
    "reports/linear_regression_risk_scores.csv",
    index=False
)

# STEP 9: SAVE EVALUATION METRICS
metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "R2 Score"],
    "Value": [mae, mse, r2]
})

metrics.to_csv(
    "reports/linear_regression_metrics.csv",
    index=False
)

print("\nLinear Regression Model Training Completed!")
print("Model saved: models/linear_regression_model.pkl")
print("Risk scores saved: reports/linear_regression_risk_scores.csv")
print("Metrics saved: reports/linear_regression_metrics.csv")
