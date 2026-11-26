import pandas as pd

# Load dataset
df = pd.read_csv("dataset/heart_disease.csv")

print("========== DATA CLEANING ==========")

# Original shape
print("\nOriginal Dataset Shape:")
print(df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate count
duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicate_count)

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# Save cleaned dataset
df.to_csv("dataset/heart_disease_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

print("\nFinal Dataset Information:")
print(df.info())

print("\nFinal Dataset Description:")
print(df.describe())