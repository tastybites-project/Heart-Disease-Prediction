
import pandas as pd
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
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

# STEP 2: CREATE KNN MODEL
model = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
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

print("========== KNN MODEL RESULTS ==========")

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# STEP 7: FIND SIMILAR PATIENTS
distances, indices = model.kneighbors(X_test)

similar_patients = []

for i in range(min(5, len(X_test))):
    for rank in range(5):
        train_index = indices[i][rank]

        similar_patients.append({
            "Test Patient Number": i + 1,
            "Similar Patient Rank": rank + 1,
            "Training Patient Index": int(train_index),
            "Distance": float(distances[i][rank]),
            "Disease Status": int(y_train.iloc[train_index])
        })

similar_df = pd.DataFrame(similar_patients)

print("\nSimilar Patients for First Test Patient:")
print(similar_df[similar_df["Test Patient Number"] == 1])

# STEP 8: CREATE PREDICTION RESULTS
results = pd.DataFrame({
    "Actual Disease Status": y_test.values,
    "Predicted Disease Status": y_pred,
    "Disease Probability (%)": probabilities * 100
})

# STEP 9: SAVE MODEL AND RESULTS
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/knn_model.pkl")

results.to_csv(
    "reports/knn_predictions.csv",
    index=False
)

similar_df.to_csv(
    "reports/knn_similar_patients.csv",
    index=False
)

# STEP 10: SAVE METRICS
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
})

metrics.to_csv(
    "reports/knn_metrics.csv",
    index=False
)

print("\nKNN Model Training Completed!")
print("Model saved: models/knn_model.pkl")
print("Predictions saved: reports/knn_predictions.csv")
print("Similar patients saved: reports/knn_similar_patients.csv")
print("Metrics saved: reports/knn_metrics.csv")
