import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("========== MODEL EVALUATION & COMPARISON ==========")


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


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# =====================================================
# 1. LOGISTIC REGRESSION
# =====================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)
logistic_f1 = f1_score(y_test, logistic_pred)


print("\n========== LOGISTIC REGRESSION ==========")

print(f"Accuracy: {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall: {logistic_recall:.4f}")
print(f"F1 Score: {logistic_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))


# =====================================================
# 2. OPTIMIZED KNN
# =====================================================

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier())
])


param_grid = {
    "classifier__n_neighbors": [3, 5, 7, 9, 11, 13, 15]
}


grid_search = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)


grid_search.fit(X_train, y_train)

knn_pred = grid_search.predict(X_test)


knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)


print("\n========== OPTIMIZED KNN ==========")

print("Best K:")
print(grid_search.best_params_)

print(f"Accuracy: {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall: {knn_recall:.4f}")
print(f"F1 Score: {knn_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, knn_pred))


# =====================================================
# 3. MODEL COMPARISON
# =====================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Optimized KNN"
    ],
    "Accuracy": [
        logistic_accuracy,
        knn_accuracy
    ],
    "Precision": [
        logistic_precision,
        knn_precision
    ],
    "Recall": [
        logistic_recall,
        knn_recall
    ],
    "F1 Score": [
        logistic_f1,
        knn_f1
    ]
})


print("\n========== MODEL COMPARISON ==========")

print(comparison.to_string(index=False))


# Save comparison
comparison.to_csv(
    "dataset/model_comparison.csv",
    index=False
)


print("\nModel comparison saved successfully!")

print("\nModel evaluation completed successfully!")
