
import pandas as pd

# STEP 1: LOAD OUTLIER-HANDLED DATASET
df = pd.read_csv("dataset/heart_disease_outlier_handled.csv")

print("========== ORIGINAL DATASET ==========")
print("Dataset Shape:", df.shape)
print("\nOriginal Columns:")
print(df.columns.tolist())

# STEP 2: SELECT CATEGORICAL COLUMNS
categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

print("\n========== CATEGORICAL COLUMNS ==========")
print(categorical_columns)

# STEP 3: APPLY ONE-HOT ENCODING
df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)

# STEP 4: DISPLAY ENCODED DATA
print("\n========== ENCODED DATASET ==========")
print("Dataset Shape:", df_encoded.shape)

print("\nEncoded Columns:")
print(df_encoded.columns.tolist())

print("\nFirst 5 Encoded Records:")
print(df_encoded.head())

# STEP 5: SAVE ENCODED DATASET
df_encoded.to_csv(
    "dataset/heart_disease_encoded.csv",
    index=False
)

print("\n========== ENCODING COMPLETED ==========")
print("Encoded dataset saved successfully!")
print("File: dataset/heart_disease_encoded.csv")
