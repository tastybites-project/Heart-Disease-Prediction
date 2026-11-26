import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


print("========== K-NEAREST NEIGHBORS (KNN) ==========")


# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("\nDataset Shape:")
print(df.shape)


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


# Create KNN Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])


# Train Model
model.fit(X_train, y_train)

print("\nKNN model trained successfully!")


# Prediction
y_pred = model.predict(X_test)


print("\nPredicted Values:")
print(y_pred[:10])

print("\nActual Values:")
print(y_test.values[:10])


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nKNN completed successfully!")