
import pandas as pd
import numpy as np

# ==========================================
# STEP 1: LOAD ORIGINAL DATASET
# ==========================================

df = pd.read_csv("dataset/heart_disease.csv")

print("========== ORIGINAL DATASET ==========")
print("Dataset Shape:", df.shape)


# ==========================================
# STEP 2: CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES BEFORE CLEANING ==========")
print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())


# ==========================================
# STEP 3: HANDLE MISSING VALUES
# ==========================================

# Handle ca using Median
ca_median = df["ca"].median()
df["ca"] = df["ca"].fillna(ca_median)

print("\nMedian used for ca:", ca_median)

# Handle thal using Mode
thal_mode = df["thal"].mode()[0]
df["thal"] = df["thal"].fillna(thal_mode)

print("Mode used for thal:", thal_mode)


# ==========================================
# STEP 4: CHECK MISSING VALUES AFTER CLEANING
# ==========================================

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())


# ==========================================
# STEP 5: CHECK DUPLICATE RECORDS
# ==========================================

duplicates_before = df.duplicated().sum()

print("\n========== DUPLICATE REMOVAL ==========")
print("Duplicates Before Removal:", duplicates_before)


# ==========================================
# STEP 6: REMOVE DUPLICATE RECORDS
# ==========================================

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

print("Duplicates After Removal:", duplicates_after)

print("Dataset Shape After Duplicate Removal:", df.shape)


# ==========================================
# STEP 7: FINAL DATASET INFORMATION
# ==========================================

print("\n========== FINAL DATASET INFORMATION ==========")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nRemaining Missing Values:",
      df.isnull().sum().sum())

print("Remaining Duplicates:",
      df.duplicated().sum())


# ==========================================
# STEP 8: SAVE CLEANED DATASET
# ==========================================

df.to_csv(
    "dataset/heart_disease_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File: dataset/heart_disease_cleaned.csv")

print("\n========== PREPROCESSING COMPLETED ==========")
