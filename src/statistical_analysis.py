
import pandas as pd

# STEP 1: LOAD ENCODED DATASET
df = pd.read_csv("dataset/heart_disease_encoded.csv")

print("========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)

# STEP 2: DESCRIPTIVE STATISTICS
print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe().round(2))

# STEP 3: MEDIAN VALUES
print("\n========== MEDIAN VALUES ==========")
print(df.median().round(2))

# STEP 4: STANDARD DEVIATION
print("\n========== STANDARD DEVIATION ==========")
print(df.std().round(2))

# STEP 5: HEART DISEASE STATUS
# Original num: 0 = No Disease, 1-4 = Disease
df["disease_status"] = (df["num"] > 0).astype(int)

disease_counts = df["disease_status"].value_counts().sort_index()

print("\n========== HEART DISEASE DISTRIBUTION ==========")
print("No Heart Disease (0):", disease_counts.get(0, 0))
print("Heart Disease (1):", disease_counts.get(1, 0))

# STEP 6: PERCENTAGE
print("\n========== DISEASE PERCENTAGE ==========")

percentages = (
    df["disease_status"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("No Heart Disease:", round(percentages.get(0, 0), 2), "%")
print("Heart Disease:", round(percentages.get(1, 0), 2), "%")

# STEP 7: SAVE STATISTICAL REPORT
df.describe().round(2).to_csv(
    "reports/statistical_summary.csv"
)

print("\nStatistical analysis completed!")
print("Report saved: reports/statistical_summary.csv")
