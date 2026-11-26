import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


print("========== MODEL OPTIMIZATION ==========")


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
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier())
])


# Different K values
param_grid = {
    "classifier__n_neighbors": [3, 5, 7, 9, 11, 13, 15]
}


# Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)


# Train Grid Search
grid_search.fit(X_train, y_train)


print("\nOptimization completed successfully!")


# Best parameter
print("\nBest K Value:")
print(grid_search.best_params_)


# Best cross-validation score
print("\nBest Cross-Validation Accuracy:")
print(f"{grid_search.best_score_:.4f}")


# Test set score
test_accuracy = grid_search.score(X_test, y_test)

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")


# Show all results
results = pd.DataFrame(grid_search.cv_results_)

print("\n========== ALL K VALUES ==========")

for k, score in zip(
    results["param_classifier__n_neighbors"],
    results["mean_test_score"]
):
    print(f"K = {k} → CV Accuracy = {score:.4f}")


print("\nModel optimization completed successfully!")
