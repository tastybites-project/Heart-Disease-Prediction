
import pandas as pd

# STEP 1: LOAD DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

# STEP 2: CREATE BINARY TARGET
df["target"] = (df["num"] > 0).astype(int)

# STEP 3: DATASET SUMMARY
total_patients = len(df)

no_disease = (df["target"] == 0).sum()
heart_disease = (df["target"] == 1).sum()

# STEP 4: AVERAGE HEALTH INDICATORS
average_age = df["age"].mean()
average_chol = df["chol"].mean()
average_bp = df["trestbps"].mean()
average_thalach = df["thalach"].mean()

# STEP 5: CREATE SUMMARY
summary = pd.DataFrame({
    "Metric": [
        "Total Patients",
        "No Heart Disease",
        "Heart Disease",
        "Heart Disease Percentage",
        "Average Age",
        "Average Cholesterol",
        "Average Blood Pressure",
        "Average Maximum Heart Rate"
    ],
    "Value": [
        total_patients,
        no_disease,
        heart_disease,
        round((heart_disease / total_patients) * 100, 2),
        round(average_age, 2),
        round(average_chol, 2),
        round(average_bp, 2),
        round(average_thalach, 2)
    ]
})

# STEP 6: DISPLAY SUMMARY
print("========== EDA SUMMARY ==========")
print(summary.to_string(index=False))

# STEP 7: SAVE SUMMARY
summary.to_csv(
    "reports/eda_summary.csv",
    index=False
)

print("\nEDA Summary completed successfully!")
print("Report saved: reports/eda_summary.csv")
