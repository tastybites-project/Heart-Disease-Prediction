
import pandas as pd
import joblib
import os

# STEP 1: LOAD OPTIMIZATION RESULTS
df = pd.read_csv("reports/model_optimization_results.csv")

print("========== MODEL PERFORMANCE COMPARISON ==========")

print("\nComplete Model Comparison:")
print(df.round(4).to_string(index=False))

# STEP 2: RANK MODELS BY F1 SCORE
df = df.sort_values(
    by=["F1 Score", "Recall", "Test Accuracy"],
    ascending=False
).reset_index(drop=True)

df["Rank"] = range(1, len(df) + 1)

# STEP 3: DISPLAY RANKING
print("\n========== MODEL RANKING ==========")

print(
    df[
        ["Rank", "Model", "Test Accuracy",
         "Precision", "Recall", "F1 Score"]
    ].round(4).to_string(index=False)
)

# STEP 4: SELECT BEST MODEL
best_model_name = df.iloc[0]["Model"]

model_files = {
    "Logistic Regression":
        "models/logistic_regression_optimized.pkl",

    "KNN":
        "models/knn_optimized.pkl",

    "Random Forest":
        "models/random_forest_optimized.pkl"
}

best_model_path = model_files[best_model_name]

best_model = joblib.load(best_model_path)

# STEP 5: SAVE BEST MODEL
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_heart_disease_model.pkl"
)

# STEP 6: SAVE COMPARISON REPORT
df.to_csv(
    "reports/final_model_comparison.csv",
    index=False
)

# STEP 7: SAVE BEST MODEL INFORMATION
best_info = pd.DataFrame([{
    "Best Model": best_model_name,
    "Test Accuracy": df.iloc[0]["Test Accuracy"],
    "Precision": df.iloc[0]["Precision"],
    "Recall": df.iloc[0]["Recall"],
    "F1 Score": df.iloc[0]["F1 Score"]
}])

best_info.to_csv(
    "reports/best_model_summary.csv",
    index=False
)

print("\n========== BEST MODEL SELECTED ==========")
print("Best Model:", best_model_name)
print("Test Accuracy:", round(df.iloc[0]["Test Accuracy"], 4))
print("Precision:", round(df.iloc[0]["Precision"], 4))
print("Recall:", round(df.iloc[0]["Recall"], 4))
print("F1 Score:", round(df.iloc[0]["F1 Score"], 4))

print("\nBest model saved:")
print("models/best_heart_disease_model.pkl")

print("\nComparison report saved:")
print("reports/final_model_comparison.csv")

print("\nBest model summary saved:")
print("reports/best_model_summary.csv")
