
import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# STEP 1: LOAD TRAINING AND TESTING DATA
X_train = pd.read_csv("dataset/X_train_scaled.csv")
X_test = pd.read_csv("dataset/X_test_scaled.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze("columns")
y_test = pd.read_csv("dataset/y_test.csv").squeeze("columns")

# STEP 2: DEFINE CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# STEP 3: DEFINE MODELS AND PARAMETERS
models = {
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "params": {
            "C": [0.1, 1, 10]
        }
    },

    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=42
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    }
}

# STEP 4: TRAIN AND OPTIMIZE MODELS
results = []
best_models = {}

for name, config in models.items():

    print("\nOptimizing:", name)

    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_models[name] = best_model

    # Cross-validation score
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    # Test predictions
    y_pred = best_model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": name,
        "Best Parameters": str(grid.best_params_),
        "Best CV Accuracy": grid.best_score_,
        "Mean CV Accuracy": cv_scores.mean(),
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_, 4))
    print("Mean CV Accuracy:", round(cv_scores.mean(), 4))
    print("Test Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

# STEP 5: CREATE RESULTS DATAFRAME
results_df = pd.DataFrame(results)

# STEP 6: SAVE RESULTS
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "reports/model_optimization_results.csv",
    index=False
)

# STEP 7: SAVE OPTIMIZED MODELS
for name, model in best_models.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_optimized.pkl"
    )

    joblib.dump(
        model,
        os.path.join("models", filename)
    )

# STEP 8: DISPLAY FINAL COMPARISON
print("\n========== FINAL MODEL COMPARISON ==========")

print(
    results_df[
        ["Model", "Best CV Accuracy", "Test Accuracy",
         "Precision", "Recall", "F1 Score"]
    ].round(4).to_string(index=False)
)

print("\nOptimization Completed!")
print("Results saved: reports/model_optimization_results.csv")
print("Optimized models saved inside models/ folder.")
