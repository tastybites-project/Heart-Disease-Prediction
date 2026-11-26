
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

# Numerical columns for outlier analysis
columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

print("========== OUTLIER DETECTION ==========")

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print("\nColumn:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Outliers:", len(outliers))

print("\nOutlier detection completed!")
