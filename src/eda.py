import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

print("========== EXPLORATORY DATA ANALYSIS ==========")

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Target distribution
print("\nTarget Distribution:")
print(df["target"].value_counts())

# Target percentage
print("\nTarget Percentage:")
print(df["target"].value_counts(normalize=True) * 100)

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))