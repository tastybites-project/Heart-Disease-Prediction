
import pandas as pd

# STEP 1: LOAD CLEANED DATASET
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("========== BEFORE OUTLIER HANDLING ==========")
print("Dataset Shape:", df.shape)

# STEP 2: SELECT NUMERICAL COLUMNS
columns = ["trestbps", "chol", "thalach", "oldpeak"]

# STEP 3: DETECT AND CAP OUTLIERS USING IQR
print("\n========== OUTLIER HANDLING ==========")

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers_before = (
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ).sum()

    df[col] = df[col].clip(
        lower=lower_limit,
        upper=upper_limit
    )

    outliers_after = (
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ).sum()

    print("\nColumn:", col)
    print("Outliers Before:", outliers_before)
    print("Outliers After:", outliers_after)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

# STEP 4: SAVE PROCESSED DATASET
df.to_csv(
    "dataset/heart_disease_outlier_handled.csv",
    index=False
)

print("\n========== FINAL RESULT ==========")
print("Final Dataset Shape:", df.shape)
print("Outlier-handled dataset saved successfully!")
print("File: dataset/heart_disease_outlier_handled.csv")
