
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/heart_disease.csv")

# 1. First five records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# 2. Last five records
print("\n========== LAST 5 RECORDS ==========")
print(df.tail())

# 3. Dataset shape
print("\n========== DATASET SHAPE ==========")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# 4. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 5. Dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 6. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 7. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 8. Total missing values
print("\nTotal Missing Values:", df.isnull().sum().sum())

# 9. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Total Duplicates:", df.duplicated().sum())

# 10. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 11. Target distribution
print("\n========== HEART DISEASE DISTRIBUTION ==========")
print(df["num"].value_counts().sort_index())

print("\nDataset inspection completed successfully!")
